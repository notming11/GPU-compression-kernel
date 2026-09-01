import argparse
import importlib.util
import math
import os
import sys
import torch
import triton

from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.nvidia.hopper import TensorDescriptor
from triton.language.core import _aggregate as aggregate
from triton.experimental.gluon.language.nvidia.hopper import (
    tma,
    mbarrier,
    fence_async_shared,
)

from common import WGMMA, pick_wgmma_layout
from sparsifier import (
    compress_2_4_autotune, 
    ws_tma_compress_2_4_kernel, 
    compress_tma_set_block_size_hook,
)

# ---------------------------------------------------------------------------
# WORKSPACE & ENVIRONMENT OVERRIDES
# ---------------------------------------------------------------------------
SCRATCH_WORKSPACE = "compiler_scratch"
JOB_ID = str(os.getpid())

os.makedirs(SCRATCH_WORKSPACE, exist_ok=True)
os.makedirs(os.path.join(SCRATCH_WORKSPACE, f"triton_cache_{JOB_ID}"), exist_ok=True)
os.makedirs(os.path.join(SCRATCH_WORKSPACE, f"cuda_cache_{JOB_ID}"), exist_ok=True)

os.environ["TRITON_CACHE_DIR"] = os.path.join(SCRATCH_WORKSPACE, f"triton_cache_{JOB_ID}")
os.environ["TMPDIR"] = SCRATCH_WORKSPACE
os.environ["TMP"] = SCRATCH_WORKSPACE
os.environ["TEMP"] = SCRATCH_WORKSPACE
os.environ["CUDA_CACHE_PATH"] = os.path.join(SCRATCH_WORKSPACE, f"cuda_cache_{JOB_ID}")
os.environ["TORCH_HOME"] = os.path.join(SCRATCH_WORKSPACE, f"cuda_cache_{JOB_ID}")
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TRITON_CACHE_AUTOTUNING"] = "0"

# ---------------------------------------------------------------------------
# SHARED HELPERS & ARGS
# ---------------------------------------------------------------------------

def GroupedPersistentTileScheduler(GROUP_SIZE_M):
    GROUP_SIZE_M = gl.constexpr(GROUP_SIZE_M)

    @aggregate
    class GroupedPersistentTileSchedulerImpl:
        start_pid: gl.tensor
        num_pid_m: gl.tensor
        num_pid_in_group: gl.tensor
        num_pid: gl.tensor

        @gluon.constexpr_function
        def __init__(self, start_pid, num_pid_m, num_pid_in_group, num_pid):
            self.start_pid = start_pid
            self.num_pid_m = num_pid_m
            self.num_pid_in_group = num_pid_in_group
            self.num_pid = num_pid

        @gluon.jit
        def initialize(M, N, BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr):
            start_pid = gl.program_id(axis=0)
            num_pid_m = gl.cdiv(M, BLOCK_M)
            num_pid_n = gl.cdiv(N, BLOCK_N)
            num_pid_in_group = GROUP_SIZE_M * num_pid_n
            num_pid = num_pid_m * num_pid_n
            return GroupedPersistentTileSchedulerImpl(start_pid, num_pid_m, num_pid_in_group, num_pid)

        @gluon.jit
        def get_num_tiles(self):
            return gl.cdiv(self.num_pid - self.start_pid, gl.num_programs(axis=0))

        @gluon.jit
        def get_tile(self, idx, SEQ_LEN: gl.constexpr, BLOCK_M: gl.constexpr, NUM_HEADS: gl.constexpr):
            tile_id = self.start_pid + idx * gl.num_programs(axis=0)
            num_pid_m = SEQ_LEN // BLOCK_M
            pid_m = tile_id % num_pid_m
            batch_head_idx = tile_id // num_pid_m
            global_m_offset = (batch_head_idx * SEQ_LEN) + (pid_m * BLOCK_M)
            return pid_m, batch_head_idx, global_m_offset

    GroupedPersistentTileSchedulerImpl.__name__ = f"GroupedPersistentTileScheduler({GROUP_SIZE_M.value})"
    return GroupedPersistentTileSchedulerImpl

@aggregate 
class PartitionArgs:
    q0_desc: tma.tensor_descriptor
    q1_desc: tma.tensor_descriptor
    eq0_desc: tma.tensor_descriptor
    eq1_desc: tma.tensor_descriptor
    k_desc: tma.tensor_descriptor
    v_desc: tma.tensor_descriptor
    o0_desc: tma.tensor_descriptor
    o1_desc: tma.tensor_descriptor

    q0_buf: gl.shared_memory_descriptor
    q1_buf: gl.shared_memory_descriptor
    eq0_buf: gl.shared_memory_descriptor
    eq1_buf: gl.shared_memory_descriptor
    k_bufs: gl.shared_memory_descriptor
    v_bufs: gl.shared_memory_descriptor
    o0_bufs: gl.shared_memory_descriptor
    o1_bufs: gl.shared_memory_descriptor

    q_ready_bar: gl.shared_memory_descriptor
    q_empty_bar: gl.shared_memory_descriptor
    kv_empty_bars: gl.shared_memory_descriptor
    kv_ready_bars: gl.shared_memory_descriptor
    
    o0_empty_bars: gl.shared_memory_descriptor
    o0_ready_bars: gl.shared_memory_descriptor
    o1_empty_bars: gl.shared_memory_descriptor
    o1_ready_bars: gl.shared_memory_descriptor

    ping_bar: gl.shared_memory_descriptor
    pong_bar: gl.shared_memory_descriptor

    SUBTILE_FACTOR: gl.constexpr
    num_warps: gl.constexpr
    
    @gluon.constexpr_function
    def __init__(
        self, 
        q0_desc, q1_desc, eq0_desc, eq1_desc, k_desc, v_desc, o0_desc, o1_desc, 
        q0_buf, q1_buf, eq0_buf, eq1_buf, k_bufs, v_bufs, o0_bufs, o1_bufs, 
        q_ready_bar, q_empty_bar, 
        kv_empty_bars, kv_ready_bars,
        o0_empty_bars, o0_ready_bars,
        o1_empty_bars, o1_ready_bars,
        ping_bar, pong_bar,
        SUBTILE_FACTOR: gl.constexpr, 
        num_warps: gl.constexpr
    ):
        self.q0_desc = q0_desc
        self.q1_desc = q1_desc
        self.eq0_desc = eq0_desc
        self.eq1_desc = eq1_desc
        self.k_desc = k_desc
        self.v_desc = v_desc
        self.o0_desc = o0_desc
        self.o1_desc = o1_desc
        
        self.q0_buf = q0_buf
        self.q1_buf = q1_buf
        self.eq0_buf = eq0_buf
        self.eq1_buf = eq1_buf
        self.k_bufs = k_bufs
        self.v_bufs = v_bufs
        self.o0_bufs = o0_bufs
        self.o1_bufs = o1_bufs
        
        self.q_ready_bar = q_ready_bar
        self.q_empty_bar = q_empty_bar
        self.kv_empty_bars = kv_empty_bars
        self.kv_ready_bars = kv_ready_bars
        
        self.o0_empty_bars = o0_empty_bars
        self.o0_ready_bars = o0_ready_bars
        self.o1_empty_bars = o1_empty_bars
        self.o1_ready_bars = o1_ready_bars
        
        self.ping_bar = ping_bar
        self.pong_bar = pong_bar

        self.SUBTILE_FACTOR = gl.constexpr(SUBTILE_FACTOR)
        self.num_warps = gl.constexpr(num_warps)

@aggregate
class Counter:
    index: gl.tensor
    phase: gl.tensor
    num_barriers: gl.constexpr

    @gluon.constexpr_function
    def __init__(self, index, phase, num_barriers):
        self.index = index
        self.phase = phase
        self.num_barriers = gl.constexpr(num_barriers)

    @gluon.jit
    def create(phase, num_barriers: gl.constexpr):
        return Counter(gl.to_tensor(0), gl.to_tensor(phase), num_barriers)

    @gluon.must_use_result
    @gluon.jit
    def next(self, pred=True):
        incr = self.index + gl.where(pred, 1, 0)
        rollover = incr == self.num_barriers
        index = gl.where(rollover, 0, incr)
        phase = gl.where(rollover, self.phase ^ 1, self.phase)
        return Counter(index, phase, self.num_barriers)

@gluon.jit
def _split_n(x, SUBTILE_FACTOR: gl.constexpr):
    split_count: gl.constexpr = SUBTILE_FACTOR.bit_length() - 1
    xs = (x, )
    for _ in gl.static_range(split_count):
        next_xs = ()
        for j in gl.static_range(len(xs)):
            x = xs[j]
            next_xs += x.reshape(x.shape[0], 2, x.shape[1] // 2).permute(0, 2, 1).split()
        xs = next_xs
    return xs

@gluon.jit
def store_acc_to_smem_subtile(acc, o_bufs, o_empty_bars, o_ready_bars, acc_state, SUBTILE_FACTOR: gl.constexpr):
    accs = _split_n(acc, SUBTILE_FACTOR)

    for i in gl.static_range(SUBTILE_FACTOR):
        mbarrier.wait(o_empty_bars.index(acc_state.index), acc_state.phase)
        o_buf = o_bufs.index(acc_state.index)

        o_buf.store(accs[i])
        fence_async_shared()
        mbarrier.arrive(o_ready_bars.index(acc_state.index), count=1)
        acc_state = acc_state.next()

    return acc_state

# ---------------------------------------------------------------------------
# PARTITIONS
# ---------------------------------------------------------------------------

@gluon.jit
def fa3_producer_partition(p: PartitionArgs, SchedulerImpl: gl.constexpr, SEQ_LEN: gl.constexpr, NUM_HEADS: gl.constexpr, HEAD_DIM: gl.constexpr, p_layout: gl.constexpr, m_layout: gl.constexpr, s_layout: gl.constexpr):
    SUB_BM: gl.constexpr = p.q0_desc.block_type.shape[0]
    BLOCK_M: gl.constexpr = SUB_BM * 2
    BLOCK_N: gl.constexpr = p.k_desc.block_type.shape[0]
    BLOCK_K: gl.constexpr = p.k_desc.block_type.shape[1]

    scheduler = SchedulerImpl.initialize(p.o0_desc.shape[0], p.o0_desc.shape[1], BLOCK_M, BLOCK_K)

    kv_state = Counter.create(1, p.kv_empty_bars.shape[0])
    q_state = Counter.create(1, p.q_empty_bar.shape[0])

    for tile_idx in range(scheduler.get_num_tiles()):
        pid_m, bh_idx, global_m_offset = scheduler.get_tile(tile_idx, SEQ_LEN, BLOCK_M, NUM_HEADS)      
        
        mbarrier.wait(p.q_empty_bar.index(0), q_state.phase)
        q_bar = p.q_ready_bar.index(0)
        
        mbarrier.expect(
            q_bar, 
            p.q0_desc.block_type.nbytes + p.q1_desc.block_type.nbytes +
            p.eq0_desc.block_type.nbytes + p.eq1_desc.block_type.nbytes
        )
        
        tma.async_copy_global_to_shared(p.q0_desc, [global_m_offset, 0], q_bar, p.q0_buf)
        tma.async_copy_global_to_shared(p.eq0_desc, [global_m_offset // 16, 0], q_bar, p.eq0_buf)
        
        tma.async_copy_global_to_shared(p.q1_desc, [global_m_offset + SUB_BM, 0], q_bar, p.q1_buf)
        tma.async_copy_global_to_shared(p.eq1_desc, [(global_m_offset + SUB_BM) // 16, 0], q_bar, p.eq1_buf)
        
        kv_global_offset = bh_idx * SEQ_LEN
        num_steps = SEQ_LEN // BLOCK_N

        for step in range(num_steps):
            bar = p.kv_ready_bars.index(kv_state.index)
            mbarrier.wait(p.kv_empty_bars.index(kv_state.index), kv_state.phase)

            mbarrier.expect(bar, p.k_desc.block_type.nbytes + p.v_desc.block_type.nbytes)
            tma.async_copy_global_to_shared(p.k_desc, [kv_global_offset + step * BLOCK_N, 0], bar, p.k_bufs.index(kv_state.index))
            tma.async_copy_global_to_shared(p.v_desc, [kv_global_offset + step * BLOCK_N, 0], bar, p.v_bufs.index(kv_state.index))
            
            kv_state = kv_state.next()
            
        q_state = q_state.next()

@gluon.jit
def fa3_consumer_wg0(p: PartitionArgs, SchedulerImpl: gl.constexpr, SEQ_LEN: gl.constexpr, NUM_HEADS: gl.constexpr, HEAD_DIM: gl.constexpr, p_layout: gl.constexpr, m_layout: gl.constexpr, s_layout: gl.constexpr):
    SUB_BM: gl.constexpr = p.q0_desc.block_type.shape[0]
    BLOCK_M: gl.constexpr = SUB_BM * 2
    BLOCK_N: gl.constexpr = p.k_desc.block_type.shape[0]
    BLOCK_K: gl.constexpr = p.v_desc.block_type.shape[1]
    
    num_stages: gl.constexpr = p.kv_ready_bars.shape[0]
    dtype: gl.constexpr = p.q0_desc.dtype

    scheduler = SchedulerImpl.initialize(p.o0_desc.shape[0], p.o0_desc.shape[1], BLOCK_M, BLOCK_K)

    acc_state = Counter.create(1, p.o0_empty_bars.shape[0])
    q_state = Counter.create(0, p.q_empty_bar.shape[0])
    kv_state = Counter.create(0, num_stages)
    
    num_steps = SEQ_LEN // BLOCK_N
    LOG2E: gl.constexpr = 1.4426950408889634
    sm_scale_log2: gl.constexpr = (1.0 / math.sqrt(HEAD_DIM)) * LOG2E

    pong_phase = 0
    mma_s_base = WGMMA.initialize(dtype, SUB_BM, BLOCK_N, p.num_warps, sparse=True)

    for tile_idx in range(scheduler.get_num_tiles()):
        pid_m, bh_idx, global_m_offset = scheduler.get_tile(tile_idx, SEQ_LEN, BLOCK_M, NUM_HEADS)   
        
        mma_o = WGMMA.initialize(dtype, SUB_BM, BLOCK_K, p.num_warps)

        m_old = gl.full((SUB_BM,), -float('inf'), dtype=gl.float32, layout=s_layout)
        l_old = gl.zeros((SUB_BM,), dtype=gl.float32, layout=s_layout)

        mbarrier.wait(p.q_ready_bar.index(0), q_state.phase)
        e_reg = mma_s_base.issue_metadata_load(p.eq0_buf)

        mbarrier.wait(p.kv_ready_bars.index(kv_state.index), kv_state.phase)
        mma_s = mma_s_base.issue_async_sparse_mma(p.q0_buf, e_reg, p.k_bufs.index(kv_state.index).permute((1, 0)))

        mbarrier.arrive(p.ping_bar.index(0), count=1)

        S_tile, mma_s = mma_s.wait_num_outstanding(0).take_result()
        S_tile = S_tile * sm_scale_log2

        m_old = gl.max(S_tile, axis=1)
        S_tile = gl.exp2(S_tile - m_old[:, None])
        l_old = gl.sum(S_tile, axis=1)

        P_cur_permuted = gl.convert_layout(gl.cast(S_tile, dtype=dtype), p_layout)

        for step in range(1, num_steps - 1):
            next_kv_state = kv_state.next()
            
            mbarrier.wait(p.pong_bar.index(0), pong_phase)
            pong_phase ^= 1

            mma_o = WGMMA(mma_o.acc, gl.to_tensor(True), mma_o.layout, SUB_BM, BLOCK_K)
            mma_o = mma_o.issue_async_mma(P_cur_permuted, p.v_bufs.index(kv_state.index))
            
            mbarrier.arrive(p.kv_empty_bars.index(kv_state.index), count=1)
            kv_state = next_kv_state
            
            mbarrier.wait(p.kv_ready_bars.index(next_kv_state.index), next_kv_state.phase)
            mma_s = mma_s_base.issue_async_sparse_mma(p.q0_buf, e_reg, p.k_bufs.index(next_kv_state.index).permute((1, 0)))
            
            mbarrier.arrive(p.ping_bar.index(0), count=1)

            S_tile, _ = mma_s.wait_num_outstanding(0).take_result()
            S_tile = S_tile * sm_scale_log2

            m_new = gl.maximum(m_old, gl.max(S_tile, axis=1))
            rescale_factor = gl.exp2(m_old - m_new)
            
            S_tile = gl.exp2(S_tile - m_new[:, None])
            l_old = l_old * rescale_factor + gl.sum(S_tile, axis=1)
            m_old = m_new
            
            P_cur_permuted = gl.convert_layout(gl.cast(S_tile, dtype=dtype), p_layout)

            o_acc, _ = mma_o.wait_num_outstanding(0).take_result()
            o_acc = o_acc * gl.convert_layout(rescale_factor, m_layout)[:, None]
            mma_o = WGMMA(o_acc, gl.to_tensor(True), mma_o.layout, SUB_BM, BLOCK_K)
            
        next_kv_state = kv_state.next()
        mbarrier.wait(p.pong_bar.index(0), pong_phase)
        pong_phase ^= 1

        mma_o = WGMMA(mma_o.acc, gl.to_tensor(True), mma_o.layout, SUB_BM, BLOCK_K)
        mma_o = mma_o.issue_async_mma(P_cur_permuted, p.v_bufs.index(kv_state.index))

        mbarrier.arrive(p.kv_empty_bars.index(kv_state.index), count=1)
        kv_state = next_kv_state

        mbarrier.wait(p.kv_ready_bars.index(next_kv_state.index), next_kv_state.phase)
        mma_s = mma_s_base.issue_async_sparse_mma(p.q0_buf, e_reg, p.k_bufs.index(next_kv_state.index).permute((1, 0)))

        mbarrier.arrive(p.ping_bar.index(0), count=1)

        S_tile, _ = mma_s.wait_num_outstanding(0).take_result()
        S_tile = S_tile * sm_scale_log2
            
        mbarrier.arrive(p.q_empty_bar.index(0), count=1)
        
        m_new = gl.maximum(m_old, gl.max(S_tile, axis=1))
        rescale_factor = gl.exp2(m_old - m_new)
            
        S_tile = gl.exp2(S_tile - m_new[:, None])
        l_old = l_old * rescale_factor + gl.sum(S_tile, axis=1)
        m_old = m_new
            
        P_cur_permuted = gl.convert_layout(gl.cast(S_tile, dtype=dtype), p_layout)

        o_acc, _ = mma_o.wait_num_outstanding(0).take_result()
        o_acc = o_acc * gl.convert_layout(rescale_factor, m_layout)[:, None]
        mma_o = WGMMA(o_acc, gl.to_tensor(True), mma_o.layout, SUB_BM, BLOCK_K)

        mbarrier.wait(p.pong_bar.index(0), pong_phase)
        pong_phase ^= 1
        
        mma_o = WGMMA(mma_o.acc, gl.to_tensor(True), mma_o.layout, SUB_BM, BLOCK_K)
        mma_o = mma_o.issue_async_mma(P_cur_permuted, p.v_bufs.index(kv_state.index))
        
        mbarrier.arrive(p.ping_bar.index(0), count=1)
        mbarrier.arrive(p.kv_empty_bars.index(kv_state.index), count=1)
        kv_state = kv_state.next()
        q_state = q_state.next()

        o_acc, mma_o = mma_o.wait_num_outstanding(0).take_result()
        l_final_m = gl.convert_layout(l_old, m_layout)
        acc_final = (o_acc / l_final_m[:, None]).to(p.o0_desc.dtype)

        acc_state = store_acc_to_smem_subtile(acc_final, p.o0_bufs, p.o0_empty_bars, p.o0_ready_bars, acc_state, p.SUBTILE_FACTOR)

@gluon.jit
def fa3_consumer_wg1(p: PartitionArgs, SchedulerImpl: gl.constexpr, SEQ_LEN: gl.constexpr, NUM_HEADS: gl.constexpr, HEAD_DIM: gl.constexpr, p_layout: gl.constexpr, m_layout: gl.constexpr, s_layout: gl.constexpr):
    SUB_BM: gl.constexpr = p.q1_desc.block_type.shape[0]
    BLOCK_M: gl.constexpr = SUB_BM * 2
    BLOCK_N: gl.constexpr = p.k_desc.block_type.shape[0]
    BLOCK_K: gl.constexpr = p.v_desc.block_type.shape[1]

    num_stages: gl.constexpr = p.kv_ready_bars.shape[0]
    dtype: gl.constexpr = p.q1_desc.dtype

    scheduler = SchedulerImpl.initialize(p.o1_desc.shape[0], p.o1_desc.shape[1], BLOCK_M, BLOCK_K)

    acc_state = Counter.create(1, p.o1_empty_bars.shape[0])
    q_state = Counter.create(0, p.q_empty_bar.shape[0])
    kv_state = Counter.create(0, num_stages)

    num_steps = SEQ_LEN // BLOCK_N
    LOG2E: gl.constexpr = 1.4426950408889634
    sm_scale_log2: gl.constexpr = (1.0 / math.sqrt(HEAD_DIM)) * LOG2E

    ping_phase = 0
    mma_s_base = WGMMA.initialize(dtype, SUB_BM, BLOCK_N, p.num_warps, sparse=True)

    for tile_idx in range(scheduler.get_num_tiles()):
        pid_m, bh_idx, global_m_offset = scheduler.get_tile(tile_idx, SEQ_LEN, BLOCK_M, NUM_HEADS)

        mma_o = WGMMA.initialize(dtype, SUB_BM, BLOCK_K, p.num_warps)

        m_old = gl.full((SUB_BM,), -float("inf"), dtype=gl.float32, layout=s_layout)
        l_old = gl.zeros((SUB_BM,), dtype=gl.float32, layout=s_layout)

        mbarrier.wait(p.ping_bar.index(0), ping_phase)
        ping_phase ^= 1

        mbarrier.wait(p.q_ready_bar.index(0), q_state.phase)
        e_reg = mma_s_base.issue_metadata_load(p.eq1_buf)

        mbarrier.wait(p.kv_ready_bars.index(kv_state.index), kv_state.phase)
        mma_s = mma_s_base.issue_async_sparse_mma(p.q1_buf, e_reg, p.k_bufs.index(kv_state.index).permute((1, 0)))

        mbarrier.arrive(p.pong_bar.index(0), count=1)

        S_tile, mma_s = mma_s.wait_num_outstanding(0).take_result()
        S_tile = S_tile * sm_scale_log2

        m_old = gl.max(S_tile, axis=1)
        S_tile = gl.exp2(S_tile - m_old[:, None])
        l_old = gl.sum(S_tile, axis=1)

        P_cur_permuted = gl.convert_layout(gl.cast(S_tile, dtype=dtype), p_layout)

        for step in range(1, num_steps - 1):
            next_kv_state = kv_state.next()

            mbarrier.wait(p.ping_bar.index(0), ping_phase)
            ping_phase ^= 1

            mma_o = WGMMA(mma_o.acc, gl.to_tensor(True), mma_o.layout, SUB_BM, BLOCK_K)
            mma_o = mma_o.issue_async_mma(P_cur_permuted, p.v_bufs.index(kv_state.index))

            mbarrier.arrive(p.kv_empty_bars.index(kv_state.index), count=1)
            kv_state = next_kv_state

            mbarrier.wait(p.kv_ready_bars.index(next_kv_state.index), next_kv_state.phase)
            mma_s = mma_s_base.issue_async_sparse_mma(p.q1_buf, e_reg, p.k_bufs.index(next_kv_state.index).permute((1, 0)))

            mbarrier.arrive(p.pong_bar.index(0), count=1)

            S_tile, _ = mma_s.wait_num_outstanding(0).take_result()
            S_tile = S_tile * sm_scale_log2

            m_new = gl.maximum(m_old, gl.max(S_tile, axis=1))
            rescale_factor = gl.exp2(m_old - m_new)

            S_tile = gl.exp2(S_tile - m_new[:, None])
            l_old = l_old * rescale_factor + gl.sum(S_tile, axis=1)
            m_old = m_new

            P_cur_permuted = gl.convert_layout(gl.cast(S_tile, dtype=dtype), p_layout)

            o_acc, _ = mma_o.wait_num_outstanding(0).take_result()
            o_acc = o_acc * gl.convert_layout(rescale_factor, m_layout)[:, None]
            mma_o = WGMMA(o_acc, gl.to_tensor(True), mma_o.layout, SUB_BM, BLOCK_K)

        next_kv_state = kv_state.next()

        mbarrier.wait(p.ping_bar.index(0), ping_phase)
        ping_phase ^= 1

        mma_o = WGMMA(mma_o.acc, gl.to_tensor(True), mma_o.layout, SUB_BM, BLOCK_K)
        mma_o = mma_o.issue_async_mma(P_cur_permuted, p.v_bufs.index(kv_state.index))

        mbarrier.arrive(p.kv_empty_bars.index(kv_state.index), count=1)
        kv_state = next_kv_state

        mbarrier.wait(p.kv_ready_bars.index(next_kv_state.index), next_kv_state.phase)
        mma_s = mma_s_base.issue_async_sparse_mma(p.q1_buf, e_reg, p.k_bufs.index(next_kv_state.index).permute((1, 0)))

        mbarrier.arrive(p.pong_bar.index(0), count=1)

        S_tile, _ = mma_s.wait_num_outstanding(0).take_result()
        S_tile = S_tile * sm_scale_log2

        mbarrier.arrive(p.q_empty_bar.index(0), count=1)

        m_new = gl.maximum(m_old, gl.max(S_tile, axis=1))
        rescale_factor = gl.exp2(m_old - m_new)

        S_tile = gl.exp2(S_tile - m_new[:, None])
        l_old = l_old * rescale_factor + gl.sum(S_tile, axis=1)
        m_old = m_new

        P_cur_permuted = gl.convert_layout(gl.cast(S_tile, dtype=dtype), p_layout)

        o_acc, _ = mma_o.wait_num_outstanding(0).take_result()
        o_acc = o_acc * gl.convert_layout(rescale_factor, m_layout)[:, None]
        mma_o = WGMMA(o_acc, gl.to_tensor(True), mma_o.layout, SUB_BM, BLOCK_K)

        mbarrier.wait(p.ping_bar.index(0), ping_phase)
        ping_phase ^= 1

        mma_o = WGMMA(mma_o.acc, gl.to_tensor(True), mma_o.layout, SUB_BM, BLOCK_K)
        mma_o = mma_o.issue_async_mma(P_cur_permuted, p.v_bufs.index(kv_state.index))

        mbarrier.arrive(p.kv_empty_bars.index(kv_state.index), count=1)
        kv_state = kv_state.next()
        q_state = q_state.next()

        o_acc, mma_o = mma_o.wait_num_outstanding(0).take_result()
        l_final_m = gl.convert_layout(l_old, m_layout)
        acc_final = (o_acc / l_final_m[:, None]).to(p.o1_desc.dtype)

        acc_state = store_acc_to_smem_subtile(acc_final, p.o1_bufs, p.o1_empty_bars, p.o1_ready_bars, acc_state, p.SUBTILE_FACTOR)

@gluon.jit
def fa3_store_partition(p: PartitionArgs, SchedulerImpl: gl.constexpr, SEQ_LEN: gl.constexpr, NUM_HEADS: gl.constexpr, HEAD_DIM: gl.constexpr, p_layout: gl.constexpr, m_layout: gl.constexpr, s_layout: gl.constexpr):
    SUB_BM: gl.constexpr = p.o0_desc.block_type.shape[0]
    BLOCK_M: gl.constexpr = SUB_BM * 2
    SPLIT_K: gl.constexpr = p.o0_desc.block_type.shape[1]
    BLOCK_K: gl.constexpr = SPLIT_K * p.SUBTILE_FACTOR

    scheduler = SchedulerImpl.initialize(p.o0_desc.shape[0], p.o0_desc.shape[1], BLOCK_M, BLOCK_K)
    state = Counter.create(0, p.o0_empty_bars.shape[0])

    num_buffers: gl.constexpr = p.o0_bufs.shape[0]
    outstanding_stores: gl.constexpr = num_buffers - 1
    store_iter = 0

    for tile_idx in range(scheduler.get_num_tiles()):
        pid_m, bh_idx, global_m_offset = scheduler.get_tile(tile_idx, SEQ_LEN, BLOCK_M, NUM_HEADS)  

        for i in gl.static_range(p.SUBTILE_FACTOR):
            mbarrier.wait(p.o0_ready_bars.index(state.index), state.phase)
            mbarrier.wait(p.o1_ready_bars.index(state.index), state.phase)

            o0_buf = p.o0_bufs.index(state.index)
            o1_buf = p.o1_bufs.index(state.index)

            tma.async_copy_shared_to_global(p.o0_desc, [global_m_offset, i * SPLIT_K], o0_buf)
            tma.async_copy_shared_to_global(p.o1_desc, [global_m_offset + SUB_BM, i * SPLIT_K], o1_buf)

            if store_iter >= outstanding_stores:
                tma.store_wait(outstanding_stores)
                empty_idx = (store_iter - outstanding_stores) % num_buffers
                mbarrier.arrive(p.o0_empty_bars.index(empty_idx), count=1)
                mbarrier.arrive(p.o1_empty_bars.index(empty_idx), count=1)

            state = state.next()
            store_iter += 1

    tma.store_wait(0)
    if store_iter > 0:
        last_empty_idx = (store_iter - 1) % num_buffers
        mbarrier.arrive(p.o0_empty_bars.index(last_empty_idx), count=1)
        mbarrier.arrive(p.o1_empty_bars.index(last_empty_idx), count=1)

# ---------------------------------------------------------------------------
# KERNEL LAUNCHER
# ---------------------------------------------------------------------------

@gluon.jit
def fa3_warp_specialized_kernel(
    q0_desc, q1_desc, eq0_desc, eq1_desc, k_desc, v_desc, o0_desc, o1_desc,
    SchedulerImpl: gl.constexpr,
    SEQ_LEN: gl.constexpr, HEAD_DIM: gl.constexpr, NUM_HEADS: gl.constexpr, 
    BLOCK_SIZE_M: gl.constexpr, BLOCK_SIZE_N: gl.constexpr, BLOCK_SIZE_K: gl.constexpr,
    num_stages: gl.constexpr, SUBTILE_FACTOR: gl.constexpr, num_warps: gl.constexpr
):
    gl.static_print(f"BM: {BLOCK_SIZE_M}, BN: {BLOCK_SIZE_N}, BK: {BLOCK_SIZE_K}, buf: {num_stages}, SF: {SUBTILE_FACTOR}, warp: {num_warps}", flush=True)
    dtype: gl.constexpr = q0_desc.dtype
    SUB_BM: gl.constexpr = BLOCK_SIZE_M // 2

    q0_buf = gl.allocate_shared_memory(dtype, q0_desc.block_type.shape, q0_desc.layout)
    q1_buf = gl.allocate_shared_memory(dtype, q1_desc.block_type.shape, q1_desc.layout)
    eq0_buf = gl.allocate_shared_memory(eq0_desc.dtype, eq0_desc.block_type.shape, eq0_desc.layout)
    eq1_buf = gl.allocate_shared_memory(eq1_desc.dtype, eq1_desc.block_type.shape, eq1_desc.layout)
    
    k_bufs = gl.allocate_shared_memory(dtype, [num_stages] + k_desc.block_type.shape, k_desc.layout)
    v_bufs = gl.allocate_shared_memory(dtype, [num_stages] + v_desc.block_type.shape, v_desc.layout)
    
    o0_bufs = gl.allocate_shared_memory(dtype, [2] + o0_desc.block_type.shape, o0_desc.layout)
    o1_bufs = gl.allocate_shared_memory(dtype, [2] + o1_desc.block_type.shape, o1_desc.layout)

    q_ready_bar = gl.allocate_shared_memory(gl.int64, [1, 1], mbarrier.MBarrierLayout())
    q_empty_bar = gl.allocate_shared_memory(gl.int64, [1, 1], mbarrier.MBarrierLayout())
    
    kv_empty_bars = gl.allocate_shared_memory(gl.int64, [num_stages, 1], mbarrier.MBarrierLayout())
    kv_ready_bars = gl.allocate_shared_memory(gl.int64, [num_stages, 1], mbarrier.MBarrierLayout())

    o0_empty_bars = gl.allocate_shared_memory(gl.int64, [2, 1], mbarrier.MBarrierLayout())
    o0_ready_bars = gl.allocate_shared_memory(gl.int64, [2, 1], mbarrier.MBarrierLayout())
    o1_empty_bars = gl.allocate_shared_memory(gl.int64, [2, 1], mbarrier.MBarrierLayout())
    o1_ready_bars = gl.allocate_shared_memory(gl.int64, [2, 1], mbarrier.MBarrierLayout())

    ping_bar = gl.allocate_shared_memory(gl.int64, [1, 1], mbarrier.MBarrierLayout())
    pong_bar = gl.allocate_shared_memory(gl.int64, [1, 1], mbarrier.MBarrierLayout())

    mbarrier.init(q_ready_bar.index(0), count=1)
    mbarrier.init(q_empty_bar.index(0), count=2)

    mbarrier.init(ping_bar.index(0), count=1)
    mbarrier.init(pong_bar.index(0), count=1)

    for i in gl.static_range(num_stages):
        mbarrier.init(kv_ready_bars.index(i), count=1)
        mbarrier.init(kv_empty_bars.index(i), count=2)

    for i in gl.static_range(2):
        mbarrier.init(o0_ready_bars.index(i), count=1)
        mbarrier.init(o0_empty_bars.index(i), count=1)

        mbarrier.init(o1_ready_bars.index(i), count=1)
        mbarrier.init(o1_empty_bars.index(i), count=1)

    p = PartitionArgs(
        q0_desc, q1_desc, eq0_desc, eq1_desc, k_desc, v_desc, o0_desc, o1_desc,
        q0_buf, q1_buf, eq0_buf, eq1_buf, k_bufs, v_bufs, o0_bufs, o1_bufs,
        q_ready_bar, q_empty_bar, 
        kv_empty_bars, kv_ready_bars,
        o0_empty_bars, o0_ready_bars,
        o1_empty_bars, o1_ready_bars,
        ping_bar, pong_bar,
        SUBTILE_FACTOR, num_warps
    )
    
    p_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=0,
        parent=pick_wgmma_layout(dtype, SUB_BM, BLOCK_SIZE_K, num_warps),
        k_width=32 // dtype.primitive_bitwidth,
        meta=0,
    )
    
    m_layout: gl.constexpr = gl.SliceLayout(dim=1, parent=pick_wgmma_layout(dtype, SUB_BM, BLOCK_SIZE_K, num_warps))
    s_layout: gl.constexpr = gl.SliceLayout(dim=1, parent=pick_wgmma_layout(dtype, SUB_BM, BLOCK_SIZE_N, num_warps))

    gl.warp_specialize([
        (fa3_consumer_wg0, (p, SchedulerImpl, SEQ_LEN, NUM_HEADS, HEAD_DIM, p_layout, m_layout, s_layout)),
        (fa3_consumer_wg1, (p, SchedulerImpl, SEQ_LEN, NUM_HEADS, HEAD_DIM, p_layout, m_layout, s_layout)),
        (fa3_producer_partition, (p, SchedulerImpl, SEQ_LEN, NUM_HEADS, HEAD_DIM, p_layout, m_layout, s_layout)),
        (fa3_store_partition, (p, SchedulerImpl, SEQ_LEN, NUM_HEADS, HEAD_DIM, p_layout, m_layout, s_layout)),
    ], [num_warps, 1, 1], [240, 24, 24])

# ---------------------------------------------------------------------------
# AUTOTUNER & CONFIG HOOKS
# ---------------------------------------------------------------------------

def fa3_get_configs(pre_hook=None, tune=True):
    def valid(BM, BN, BK, warps, num_stages, SF):
        if BM == 256 and BN == 256:
            return False
        SUB_BM = BM // 2
        if SUB_BM % 64 != 0:
            return False

        fp16_elements = (
            (2 * SUB_BM * (BK // 2)) +            # Compressed Q0, Q1
            (2 * num_stages * BK * BN) +          # K, V buffers
            (2 * 2 * SUB_BM * (BK // SF))         # Output subtile buffers
        )
        fp16_smem_bytes = 2 * fp16_elements
        meta_bytes = 2 * (2 * (SUB_BM // 16) * BK) # Metadata EQ0, EQ1
        num_barriers = 2 + (2 * num_stages) + 8 + 2
        barrier_bytes = 8 * num_barriers

        total_smem_bytes = fp16_smem_bytes + meta_bytes + barrier_bytes
        if total_smem_bytes > 232448:
            return False

        if BK % SF != 0:
            return False

        split_k = BK // SF
        if split_k < 32:
            return False

        warps_m = 4
        warps_n = 1
        m = 16
        while (warps_m * warps_n) != warps:
            if SUB_BM > m * warps_m:
                warps_m *= 2
            else:
                warps_n *= 2

        if SF > 1 and warps_n > 1:
            return False
        if SUB_BM < warps_m * 16 or BN < warps_n * 16:
            return False

        elements_per_thread = (SUB_BM * max(BN, BK)) / (warps * 32)
        required_regs = elements_per_thread + 64 
        max_regs_per_thread = min(255, 65536 // (warps * 32))

        if required_regs > max_regs_per_thread or elements_per_thread < 16:
            return False
        
        return True

    configs = [
        triton.Config(
            {
                "BLOCK_SIZE_M": BM,
                "BLOCK_SIZE_N": BN,
                "BLOCK_SIZE_K": BK,
                "num_stages": num_stages,
                "SUBTILE_FACTOR": SF,
            },
            num_warps=warps,
            num_stages=num_stages,
            pre_hook=pre_hook,
        )
        for BM in (128, )
        for BN in (64, 128, )
        for BK in (64, 128, 256)
        for warps in (4, )
        for num_stages in (2, 3, 4, 5, 6)
        for SF in (1, 2, 4, 8)
        if valid(BM, BN, BK, warps, num_stages, SF)
    ]
    
    return configs if tune else configs[:1]

def fa3_tma_set_block_size_hook(nargs):
    block_m = nargs["BLOCK_SIZE_M"]
    sub_bm = block_m // 2
    block_n = nargs["BLOCK_SIZE_N"]
    block_k = nargs["BLOCK_SIZE_K"]
    split_k = nargs["BLOCK_SIZE_K"] // nargs["SUBTILE_FACTOR"]

    nargs["q0_desc"].block_shape = [sub_bm, block_k // 2]
    nargs["q1_desc"].block_shape = [sub_bm, block_k // 2]
    nargs["eq0_desc"].block_shape = [sub_bm // 16, block_k]
    nargs["eq1_desc"].block_shape = [sub_bm // 16, block_k]
    nargs["k_desc"].block_shape = [block_n, block_k]
    nargs["v_desc"].block_shape = [block_n, block_k]
    nargs["o0_desc"].block_shape = [sub_bm, split_k]
    nargs["o1_desc"].block_shape = [sub_bm, split_k]

    layout_q = gl.NVMMASharedLayout.get_default_for(nargs["q0_desc"].block_shape, gl.float16)
    layout_eq = gl.NVMMASharedLayout.get_default_for(nargs["eq0_desc"].block_shape, gl.int16)
    layout_o = gl.NVMMASharedLayout.get_default_for(nargs["o0_desc"].block_shape, gl.float16)

    nargs["q0_desc"].layout = layout_q
    nargs["q1_desc"].layout = layout_q
    nargs["eq0_desc"].layout = layout_eq
    nargs["eq1_desc"].layout = layout_eq
    nargs["k_desc"].layout = gl.NVMMASharedLayout.get_default_for(nargs["k_desc"].block_shape, gl.float16)
    nargs["v_desc"].layout = gl.NVMMASharedLayout.get_default_for(nargs["v_desc"].block_shape, gl.float16)
    nargs["o0_desc"].layout = layout_o
    nargs["o1_desc"].layout = layout_o

_autotune_cache = {}

def get_autotuned_kernel(head_dim: int):
    if head_dim not in _autotune_cache:
        configs = [
            config for config in fa3_get_configs(pre_hook=fa3_tma_set_block_size_hook, tune=True)
            if config.kwargs["BLOCK_SIZE_K"] == head_dim
        ]
        
        _autotune_cache[head_dim] = triton.autotune(
            configs=configs,
            key=["SEQ_LEN"],
            do_bench=lambda kernel_call, quantiles: triton.testing.do_bench_cudagraph(
                kernel_call, quantiles=quantiles
            ),
        )(fa3_warp_specialized_kernel)
        
    return _autotune_cache[head_dim]

# ---------------------------------------------------------------------------
# INTEGRATED HOST EXECUTION
# ---------------------------------------------------------------------------

def compress_q_tensor(Q_flat):
    """Executes 2:4 pruning and packing step on dense Q using autotuned sparsifier kernel."""
    M_total = Q_flat.shape[0]
    K_total = Q_flat.shape[1]

    Q_comp_flat = torch.empty((M_total, K_total // 2), device=Q_flat.device, dtype=Q_flat.dtype)
    E_flat = torch.empty((M_total // 16, K_total), device=Q_flat.device, dtype=torch.int16)

    dummy_block = [1, 1]
    dummy_layout = gl.NVMMASharedLayout.get_default_for(dummy_block, gl.float16)
    dummy_meta_layout = gl.NVMMASharedLayout.get_default_for(dummy_block, gl.int16)

    a_desc = TensorDescriptor.from_tensor(Q_flat, dummy_block, dummy_layout)
    a_compressed_desc = TensorDescriptor.from_tensor(Q_comp_flat, dummy_block, dummy_layout)
    e_desc = TensorDescriptor.from_tensor(E_flat, dummy_block, dummy_meta_layout)

    def grid_prune(meta):
        return (
            triton.cdiv(M_total, meta["BLOCK_SIZE_M"]),
            triton.cdiv(K_total, meta["BLOCK_SIZE_K"]),
        )

    compress_2_4_autotune[grid_prune](
        a_desc, a_compressed_desc, e_desc,
        M_total, K_total
    )

    return Q_comp_flat, E_flat


def run_fa3_sparse_q_kernel(Q_dense, K, V, tune=True, manual_config=None):
    BATCH, NUM_HEADS, SEQ_LEN, HEAD_DIM = Q_dense.shape
    O = torch.empty_like(Q_dense)
    
    # 1. Build TMA Descriptors
    Q_flat = Q_dense.reshape(-1, HEAD_DIM)
    K_flat = K.reshape(-1, HEAD_DIM)
    V_flat = V.reshape(-1, HEAD_DIM)
    O_flat = O.reshape(-1, HEAD_DIM)

    # 2. Prune and compress Q using the autotuned 2:4 sparsifier
    Q_comp, E_Q = compress_q_tensor(Q_flat)

    dummy_block = [1, 1]
    dummy_layout = gl.NVMMASharedLayout.get_default_for(dummy_block, gl.float16)
    dummy_meta_layout = gl.NVMMASharedLayout.get_default_for(dummy_block, gl.int16)

    q0_desc = TensorDescriptor.from_tensor(Q_comp, dummy_block, dummy_layout)
    q1_desc = TensorDescriptor.from_tensor(Q_comp, dummy_block, dummy_layout)
    eq0_desc = TensorDescriptor.from_tensor(E_Q, dummy_block, dummy_meta_layout)
    eq1_desc = TensorDescriptor.from_tensor(E_Q, dummy_block, dummy_meta_layout)

    k_desc = TensorDescriptor.from_tensor(K_flat, dummy_block, dummy_layout)
    v_desc = TensorDescriptor.from_tensor(V_flat, dummy_block, dummy_layout)
    o0_desc = TensorDescriptor.from_tensor(O_flat, dummy_block, dummy_layout)
    o1_desc = TensorDescriptor.from_tensor(O_flat, dummy_block, dummy_layout)

    # 3. Launch Sparse FA3 Kernel
    if tune:
        kernel = get_autotuned_kernel(HEAD_DIM)
        def grid(meta):
            num_sms = torch.cuda.get_device_properties("cuda").multi_processor_count
            num_pid = triton.cdiv(SEQ_LEN, meta["BLOCK_SIZE_M"])
            total_tiles = num_pid * BATCH * NUM_HEADS
            return (min(num_sms, total_tiles), )

        kernel[grid](
            q0_desc, q1_desc, eq0_desc, eq1_desc, k_desc, v_desc, o0_desc, o1_desc,
            GroupedPersistentTileScheduler(4),
            SEQ_LEN, HEAD_DIM, NUM_HEADS
        )
        return O, kernel.best_config
    else:
        hook_kwargs = {
            "BLOCK_SIZE_M": manual_config["BM"],
            "BLOCK_SIZE_N": manual_config["BN"],
            "BLOCK_SIZE_K": manual_config["BK"],
            "SUBTILE_FACTOR": manual_config["SF"],
            "q0_desc": q0_desc, "q1_desc": q1_desc,
            "eq0_desc": eq0_desc, "eq1_desc": eq1_desc,
            "k_desc": k_desc, "v_desc": v_desc,
            "o0_desc": o0_desc, "o1_desc": o1_desc
        }
        fa3_tma_set_block_size_hook(hook_kwargs)

        num_sms = torch.cuda.get_device_properties("cuda").multi_processor_count
        num_pid = triton.cdiv(SEQ_LEN, manual_config["BM"])
        total_tiles = num_pid * BATCH * NUM_HEADS
        grid = (min(num_sms, total_tiles), )

        fa3_warp_specialized_kernel[grid](
            q0_desc, q1_desc, eq0_desc, eq1_desc, k_desc, v_desc, o0_desc, o1_desc,
            GroupedPersistentTileScheduler(8),
            SEQ_LEN, HEAD_DIM, NUM_HEADS,
            BLOCK_SIZE_M=manual_config["BM"],
            BLOCK_SIZE_N=manual_config["BN"],
            BLOCK_SIZE_K=manual_config["BK"],
            num_stages=manual_config["num_stages"],
            SUBTILE_FACTOR=manual_config["SF"],
            num_warps=manual_config["warps"],
        )

        return O, manual_config
    
def prune_2_4_ref(Q: torch.Tensor) -> torch.Tensor:
    """Applies 2:4 pruning along the last dimension to create a dense reference Q with 2:4 sparsity pattern."""
    orig_shape = Q.shape
    q_grouped = Q.reshape(-1, 4)
    
    _, top2_idx = torch.topk(q_grouped, k=2, dim=-1)
    
    mask = torch.zeros_like(q_grouped, dtype=torch.bool)
    mask.scatter_(-1, top2_idx, True)
    
    return (q_grouped * mask).reshape(orig_shape)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Sparse FlashAttention-3 Ping-Pong + 2-Stage Async Kernel")
    parser.add_argument("--tune", action="store_true", help="Enable Triton autotuning")
    
    parser.add_argument("--bm", type=int, default=128, help="BLOCK_SIZE_M")
    parser.add_argument("--bn", type=int, default=128, help="BLOCK_SIZE_N")
    parser.add_argument("--bk", type=int, default=64, help="HEAD_DIM (BLOCK_SIZE_K)")
    parser.add_argument("--stages", type=int, default=2, help="Number of pipeline stages for KV")
    parser.add_argument("--sf", type=int, default=1, help="SUBTILE_FACTOR")
    parser.add_argument("--warps", type=int, default=4, help="Number of compute warps")
    
    args = parser.parse_args()

    manual_config = {
        "BM": args.bm,
        "BN": args.bn,
        "BK": args.bk,
        "num_stages": args.stages,
        "SF": args.sf,
        "warps": args.warps,
    }

    NUM_HEADS = 16
    sizes = [(4096, 64)]

    for SEQ_LEN, HEAD_DIM in sizes:
        BATCH = max(1, 16384 // SEQ_LEN)
        print(f"\nTesting Sparse Q FA3: BATCH={BATCH}, NUM_HEADS={NUM_HEADS}, SEQ_LEN={SEQ_LEN}, HEAD_DIM={HEAD_DIM}", flush=True)
    
        Q = torch.randn((BATCH, NUM_HEADS, SEQ_LEN, HEAD_DIM), device="cuda", dtype=torch.float16)
        K = torch.randn((BATCH, NUM_HEADS, SEQ_LEN, HEAD_DIM), device="cuda", dtype=torch.float16)
        V = torch.randn((BATCH, NUM_HEADS, SEQ_LEN, HEAD_DIM), device="cuda", dtype=torch.float16)
    
        # Run Triton Sparse Q FA3 Kernel
        O_triton, config = run_fa3_sparse_q_kernel(Q, K, V, tune=args.tune, manual_config=manual_config)
    
        # PyTorch Reference: Prune Q to 2:4 sparsity, then compute standard SDPA
        Q_sparse_ref = prune_2_4_ref(Q)
        O_torch = torch.nn.functional.scaled_dot_product_attention(Q_sparse_ref, K, V)
    
        # Validate result match
        torch.testing.assert_close(O_torch, O_triton, rtol=1e-2, atol=2.5e-2)
        print("PASS: PyTorch reference (2:4 pruned SDPA) matches Triton Gluon Sparse Q FA3!")
    
        if args.tune:
            print(f"best config: {config}")