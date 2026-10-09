import argparse
import math
import matplotlib.pyplot as plt
import numpy as np
import torch
import triton

from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.nvidia.hopper import TensorDescriptor
from triton.language.core import _aggregate as aggregate
from triton.experimental.gluon.language.nvidia.hopper import (
    tma,
    mbarrier,
)
# from triton.experimental.gluon.language import inline_asm

from common import WGMMA, pick_wgmma_layout
from gluon_attention_qk_sparse import (
    fa3_producer_partition,
    fa3_store_partition,
    store_acc_to_smem_subtile,
    compress_q_tensor,
    fa3_tma_set_block_size_hook,
    GroupedPersistentTileScheduler,
    Counter,
)

# ---------------------------------------------------------------------------
# INLINE PTX CLOCK & PROFILING HELPERS
# ---------------------------------------------------------------------------

@gluon.jit
def get_clock():
    """Reads the 64-bit hardware timer (%clock64) in Gluon."""
    return gl.inline_asm_elementwise(
        asm="mov.u64 $0, %clock64;",
        constraints="=l",
        args=[],
        dtype=gl.int64,
        is_pure=False,
        pack=1,
    )

@gluon.jit
def record_event(debug_ptr, wg_id, step, event_idx, timestamp):
    """Stores timestamp to global memory using native Gluon gl.store."""
    offset = (wg_id * 128 + step) * 6 + event_idx
    gl.store(debug_ptr + offset, timestamp)

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
    debug_ptr: gl.tensor

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
        ping_bar, pong_bar, debug_ptr,
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
        self.debug_ptr = debug_ptr

        self.SUBTILE_FACTOR = gl.constexpr(SUBTILE_FACTOR)
        self.num_warps = gl.constexpr(num_warps)

# ---------------------------------------------------------------------------
# INSTRUMENTED CONSUMER WARPGROUPS
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# INSTRUMENTED CONSUMER WARPGROUPS (CAPTURING TRUE ASYNC WGMMA1 COMPLETION)
# ---------------------------------------------------------------------------

@gluon.jit
def fa3_consumer_wg0_profiled(p: PartitionArgs, SchedulerImpl: gl.constexpr, SEQ_LEN: gl.constexpr, NUM_HEADS: gl.constexpr, HEAD_DIM: gl.constexpr, p_layout: gl.constexpr, m_layout: gl.constexpr, s_layout: gl.constexpr):
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

        # --- PROLOGUE (Step 0) ---
        mbarrier.wait(p.q_ready_bar.index(0), q_state.phase)
        e_reg = mma_s_base.issue_metadata_load(p.eq0_buf)

        mbarrier.wait(p.kv_ready_bars.index(kv_state.index), kv_state.phase)
        t_g0_start = get_clock()
        mma_s = mma_s_base.issue_async_sparse_mma(p.q0_buf, e_reg, p.k_bufs.index(kv_state.index).permute((1, 0)))

        mbarrier.arrive(p.ping_bar.index(0), count=1)

        S_tile, mma_s = mma_s.wait_num_outstanding(0).take_result()
        t_g0_end = get_clock()

        t_soft_start = get_clock()
        S_tile = S_tile * sm_scale_log2

        m_old = gl.max(S_tile, axis=1)
        S_tile = gl.exp2(S_tile - m_old[:, None])
        l_old = gl.sum(S_tile, axis=1)

        P_cur_permuted = gl.convert_layout(gl.cast(S_tile, dtype=dtype), p_layout)
        t_soft_end = get_clock()

        if tile_idx == 0 and gl.program_id(axis=0) == 0:
            record_event(p.debug_ptr, 0, 0, 0, t_g0_start)
            record_event(p.debug_ptr, 0, 0, 1, t_g0_end)
            record_event(p.debug_ptr, 0, 0, 2, t_soft_start)
            record_event(p.debug_ptr, 0, 0, 3, t_soft_end)
            record_event(p.debug_ptr, 0, 0, 4, gl.to_tensor(0))
            record_event(p.debug_ptr, 0, 0, 5, gl.to_tensor(0))

        # --- MAIN LOOP ---
        for step in range(1, num_steps - 1):
            next_kv_state = kv_state.next()
            
            # 1. Issue WGMMA1 (PV) Asynchronously
            mbarrier.wait(p.pong_bar.index(0), pong_phase)
            pong_phase ^= 1
            t_g1_start = get_clock()

            mma_o = mma_o.issue_async_mma(P_cur_permuted, p.v_bufs.index(kv_state.index))
            
            # 2. Issue WGMMA0 (Sparse QK) Asynchronously
            t_g0_start = get_clock()
            mbarrier.arrive(p.kv_empty_bars.index(kv_state.index), count=1)
            kv_state = next_kv_state
            
            mbarrier.wait(p.kv_ready_bars.index(next_kv_state.index), next_kv_state.phase)
            mma_s = mma_s_base.issue_async_sparse_mma(p.q0_buf, e_reg, p.k_bufs.index(next_kv_state.index).permute((1, 0)))
            mbarrier.arrive(p.ping_bar.index(0), count=1)

            S_tile, _ = mma_s.wait_num_outstanding(0).take_result()
            t_g0_end = get_clock()

            # 3. Softmax & Wait for WGMMA1 Completion
            t_soft_start = get_clock()
            S_tile = S_tile * sm_scale_log2

            m_new = gl.maximum(m_old, gl.max(S_tile, axis=1))
            rescale_factor = gl.exp2(m_old - m_new)
            
            S_tile = gl.exp2(S_tile - m_new[:, None])
            l_old = l_old * rescale_factor + gl.sum(S_tile, axis=1)
            m_old = m_new
            
            P_cur_permuted = gl.convert_layout(gl.cast(S_tile, dtype=dtype), p_layout)

            # --- TRUE WGMMA1 ASYNC COMPLETION POINT ---
            o_acc, _ = mma_o.wait_num_outstanding(0).take_result()
            t_g1_end = get_clock()  # WGMMA1 hardware execution finished HERE!

            o_acc = o_acc * gl.convert_layout(rescale_factor, m_layout)[:, None]
            mma_o = WGMMA(o_acc, gl.to_tensor(True), mma_o.layout, SUB_BM, BLOCK_K)
            t_soft_end = get_clock()

            if tile_idx == 0 and gl.program_id(axis=0) == 0:
                record_event(p.debug_ptr, 0, step, 0, t_g0_start)
                record_event(p.debug_ptr, 0, step, 1, t_g0_end)
                record_event(p.debug_ptr, 0, step, 2, t_soft_start)
                record_event(p.debug_ptr, 0, step, 3, t_soft_end)
                record_event(p.debug_ptr, 0, step, 4, t_g1_start)
                record_event(p.debug_ptr, 0, step, 5, t_g1_end)

        # --- EPILOGUE 1 ---
        ep1_step = num_steps - 1
        next_kv_state = kv_state.next()

        t_g1_start = get_clock()
        mbarrier.wait(p.pong_bar.index(0), pong_phase)
        pong_phase ^= 1

        mma_o = mma_o.issue_async_mma(P_cur_permuted, p.v_bufs.index(kv_state.index))

        t_g0_start = get_clock()
        mbarrier.arrive(p.kv_empty_bars.index(kv_state.index), count=1)
        kv_state = next_kv_state

        mbarrier.wait(p.kv_ready_bars.index(next_kv_state.index), next_kv_state.phase)
        mma_s = mma_s_base.issue_async_sparse_mma(p.q0_buf, e_reg, p.k_bufs.index(next_kv_state.index).permute((1, 0)))

        mbarrier.arrive(p.ping_bar.index(0), count=1)

        S_tile, _ = mma_s.wait_num_outstanding(0).take_result()
        t_g0_end = get_clock()

        t_soft_start = get_clock()
        S_tile = S_tile * sm_scale_log2
            
        mbarrier.arrive(p.q_empty_bar.index(0), count=1)
        
        m_new = gl.maximum(m_old, gl.max(S_tile, axis=1))
        rescale_factor = gl.exp2(m_old - m_new)
            
        S_tile = gl.exp2(S_tile - m_new[:, None])
        l_old = l_old * rescale_factor + gl.sum(S_tile, axis=1)
        m_old = m_new
            
        P_cur_permuted = gl.convert_layout(gl.cast(S_tile, dtype=dtype), p_layout)

        o_acc, _ = mma_o.wait_num_outstanding(0).take_result()
        t_g1_end = get_clock()

        o_acc = o_acc * gl.convert_layout(rescale_factor, m_layout)[:, None]
        mma_o = WGMMA(o_acc, gl.to_tensor(True), mma_o.layout, SUB_BM, BLOCK_K)
        t_soft_end = get_clock()

        if tile_idx == 0 and gl.program_id(axis=0) == 0:
            record_event(p.debug_ptr, 0, ep1_step, 0, t_g0_start)
            record_event(p.debug_ptr, 0, ep1_step, 1, t_g0_end)
            record_event(p.debug_ptr, 0, ep1_step, 2, t_soft_start)
            record_event(p.debug_ptr, 0, ep1_step, 3, t_soft_end)
            record_event(p.debug_ptr, 0, ep1_step, 4, t_g1_start)
            record_event(p.debug_ptr, 0, ep1_step, 5, t_g1_end)

        # --- EPILOGUE 2 ---
        ep2_step = num_steps

        t_g1_start = get_clock()
        mbarrier.wait(p.pong_bar.index(0), pong_phase)
        pong_phase ^= 1
        
        mma_o = mma_o.issue_async_mma(P_cur_permuted, p.v_bufs.index(kv_state.index))
        
        mbarrier.arrive(p.ping_bar.index(0), count=1)
        mbarrier.arrive(p.kv_empty_bars.index(kv_state.index), count=1)
        kv_state = kv_state.next()
        q_state = q_state.next()

        o_acc, mma_o = mma_o.wait_num_outstanding(0).take_result()
        t_g1_end = get_clock()

        l_final_m = gl.convert_layout(l_old, m_layout)
        acc_final = (o_acc / l_final_m[:, None]).to(p.o0_desc.dtype)

        acc_state = store_acc_to_smem_subtile(acc_final, p.o0_bufs, p.o0_empty_bars, p.o0_ready_bars, acc_state, p.SUBTILE_FACTOR)
        
        mbarrier.wait(p.pong_bar.index(0), pong_phase)
        pong_phase ^= 1

        if tile_idx == 0 and gl.program_id(axis=0) == 0:
            record_event(p.debug_ptr, 0, ep2_step, 0, gl.to_tensor(0))
            record_event(p.debug_ptr, 0, ep2_step, 1, gl.to_tensor(0))
            record_event(p.debug_ptr, 0, ep2_step, 2, gl.to_tensor(0))
            record_event(p.debug_ptr, 0, ep2_step, 3, gl.to_tensor(0))
            record_event(p.debug_ptr, 0, ep2_step, 4, t_g1_start)
            record_event(p.debug_ptr, 0, ep2_step, 5, t_g1_end)

@gluon.jit
def fa3_consumer_wg1_profiled(p: PartitionArgs, SchedulerImpl: gl.constexpr, SEQ_LEN: gl.constexpr, NUM_HEADS: gl.constexpr, HEAD_DIM: gl.constexpr, p_layout: gl.constexpr, m_layout: gl.constexpr, s_layout: gl.constexpr):
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

        # --- PROLOGUE (Step 0) ---
        mbarrier.wait(p.ping_bar.index(0), ping_phase)
        ping_phase ^= 1

        mbarrier.wait(p.q_ready_bar.index(0), q_state.phase)
        e_reg = mma_s_base.issue_metadata_load(p.eq1_buf)

        mbarrier.wait(p.kv_ready_bars.index(kv_state.index), kv_state.phase)
        t0 = get_clock()
        mma_s = mma_s_base.issue_async_sparse_mma(p.q1_buf, e_reg, p.k_bufs.index(kv_state.index).permute((1, 0)))

        mbarrier.arrive(p.pong_bar.index(0), count=1)

        S_tile, mma_s = mma_s.wait_num_outstanding(0).take_result()
        t_g0_end = get_clock()

        t_soft_start = get_clock()
        S_tile = S_tile * sm_scale_log2

        m_old = gl.max(S_tile, axis=1)
        S_tile = gl.exp2(S_tile - m_old[:, None])
        l_old = gl.sum(S_tile, axis=1)

        P_cur_permuted = gl.convert_layout(gl.cast(S_tile, dtype=dtype), p_layout)
        t_soft_end = get_clock()

        if tile_idx == 0 and gl.program_id(axis=0) == 0:
            record_event(p.debug_ptr, 1, 0, 0, t0)
            record_event(p.debug_ptr, 1, 0, 1, t_g0_end)
            record_event(p.debug_ptr, 1, 0, 2, t_soft_start)
            record_event(p.debug_ptr, 1, 0, 3, t_soft_end)
            record_event(p.debug_ptr, 1, 0, 4, gl.to_tensor(0))
            record_event(p.debug_ptr, 1, 0, 5, gl.to_tensor(0))

        # --- MAIN LOOP ---
        for step in range(1, num_steps - 1):
            next_kv_state = kv_state.next()

            mbarrier.wait(p.ping_bar.index(0), ping_phase)
            ping_phase ^= 1
            t_g1_start = get_clock()

            mma_o = mma_o.issue_async_mma(P_cur_permuted, p.v_bufs.index(kv_state.index))

            t_g0_start = get_clock()
            mbarrier.arrive(p.kv_empty_bars.index(kv_state.index), count=1)
            kv_state = next_kv_state

            mbarrier.wait(p.kv_ready_bars.index(next_kv_state.index), next_kv_state.phase)
            mma_s = mma_s_base.issue_async_sparse_mma(p.q1_buf, e_reg, p.k_bufs.index(next_kv_state.index).permute((1, 0)))

            mbarrier.arrive(p.pong_bar.index(0), count=1)

            S_tile, _ = mma_s.wait_num_outstanding(0).take_result()
            t_g0_end = get_clock()

            t_soft_start = get_clock()
            S_tile = S_tile * sm_scale_log2

            m_new = gl.maximum(m_old, gl.max(S_tile, axis=1))
            rescale_factor = gl.exp2(m_old - m_new)

            S_tile = gl.exp2(S_tile - m_new[:, None])
            l_old = l_old * rescale_factor + gl.sum(S_tile, axis=1)
            m_old = m_new

            P_cur_permuted = gl.convert_layout(gl.cast(S_tile, dtype=dtype), p_layout)

            o_acc, _ = mma_o.wait_num_outstanding(0).take_result()
            t_g1_end = get_clock()

            o_acc = o_acc * gl.convert_layout(rescale_factor, m_layout)[:, None]
            mma_o = WGMMA(o_acc, gl.to_tensor(True), mma_o.layout, SUB_BM, BLOCK_K)
            t_soft_end = get_clock()

            if tile_idx == 0 and gl.program_id(axis=0) == 0:
                record_event(p.debug_ptr, 1, step, 0, t_g0_start)
                record_event(p.debug_ptr, 1, step, 1, t_g0_end)
                record_event(p.debug_ptr, 1, step, 2, t_soft_start)
                record_event(p.debug_ptr, 1, step, 3, t_soft_end)
                record_event(p.debug_ptr, 1, step, 4, t_g1_start)
                record_event(p.debug_ptr, 1, step, 5, t_g1_end)

        # --- EPILOGUE 1 ---
        ep1_step = num_steps - 1
        next_kv_state = kv_state.next()

        t_g1_start = get_clock()
        mbarrier.wait(p.ping_bar.index(0), ping_phase)
        ping_phase ^= 1

        mma_o = mma_o.issue_async_mma(P_cur_permuted, p.v_bufs.index(kv_state.index))

        t_g0_start = get_clock()
        mbarrier.arrive(p.kv_empty_bars.index(kv_state.index), count=1)
        kv_state = next_kv_state

        mbarrier.wait(p.kv_ready_bars.index(next_kv_state.index), next_kv_state.phase)
        mma_s = mma_s_base.issue_async_sparse_mma(p.q1_buf, e_reg, p.k_bufs.index(next_kv_state.index).permute((1, 0)))

        mbarrier.arrive(p.pong_bar.index(0), count=1)

        S_tile, _ = mma_s.wait_num_outstanding(0).take_result()
        t_g0_end = get_clock()

        t_soft_start = get_clock()
        S_tile = S_tile * sm_scale_log2

        mbarrier.arrive(p.q_empty_bar.index(0), count=1)

        m_new = gl.maximum(m_old, gl.max(S_tile, axis=1))
        rescale_factor = gl.exp2(m_old - m_new)

        S_tile = gl.exp2(S_tile - m_new[:, None])
        l_old = l_old * rescale_factor + gl.sum(S_tile, axis=1)
        m_old = m_new

        P_cur_permuted = gl.convert_layout(gl.cast(S_tile, dtype=dtype), p_layout)

        o_acc, _ = mma_o.wait_num_outstanding(0).take_result()
        t_g1_end = get_clock()

        o_acc = o_acc * gl.convert_layout(rescale_factor, m_layout)[:, None]
        mma_o = WGMMA(o_acc, gl.to_tensor(True), mma_o.layout, SUB_BM, BLOCK_K)
        t_soft_end = get_clock()

        if tile_idx == 0 and gl.program_id(axis=0) == 0:
            record_event(p.debug_ptr, 1, ep1_step, 0, t_g0_start)
            record_event(p.debug_ptr, 1, ep1_step, 1, t_g0_end)
            record_event(p.debug_ptr, 1, ep1_step, 2, t_soft_start)
            record_event(p.debug_ptr, 1, ep1_step, 3, t_soft_end)
            record_event(p.debug_ptr, 1, ep1_step, 4, t_g1_start)
            record_event(p.debug_ptr, 1, ep1_step, 5, t_g1_end)

        # --- EPILOGUE 2 ---
        ep2_step = num_steps

        t_g1_start = get_clock()
        mbarrier.wait(p.ping_bar.index(0), ping_phase)
        ping_phase ^= 1

        mma_o = mma_o.issue_async_mma(P_cur_permuted, p.v_bufs.index(kv_state.index))

        mbarrier.arrive(p.pong_bar.index(0), count=1)
        
        mbarrier.arrive(p.kv_empty_bars.index(kv_state.index), count=1)
        kv_state = kv_state.next()
        q_state = q_state.next()

        o_acc, mma_o = mma_o.wait_num_outstanding(0).take_result()
        t_g1_end = get_clock()

        l_final_m = gl.convert_layout(l_old, m_layout)
        acc_final = (o_acc / l_final_m[:, None]).to(p.o1_desc.dtype)

        acc_state = store_acc_to_smem_subtile(acc_final, p.o1_bufs, p.o1_empty_bars, p.o1_ready_bars, acc_state, p.SUBTILE_FACTOR)

        if tile_idx == 0 and gl.program_id(axis=0) == 0:
            record_event(p.debug_ptr, 1, ep2_step, 0, gl.to_tensor(0))
            record_event(p.debug_ptr, 1, ep2_step, 1, gl.to_tensor(0))
            record_event(p.debug_ptr, 1, ep2_step, 2, gl.to_tensor(0))
            record_event(p.debug_ptr, 1, ep2_step, 3, gl.to_tensor(0))
            record_event(p.debug_ptr, 1, ep2_step, 4, t_g1_start)
            record_event(p.debug_ptr, 1, ep2_step, 5, t_g1_end)

@gluon.jit
def fa3_warp_specialized_kernel_profiled(
    q0_desc, q1_desc, eq0_desc, eq1_desc, k_desc, v_desc, o0_desc, o1_desc,
    debug_ptr, SchedulerImpl: gl.constexpr,
    SEQ_LEN: gl.constexpr, HEAD_DIM: gl.constexpr, NUM_HEADS: gl.constexpr, 
    BLOCK_SIZE_M: gl.constexpr, BLOCK_SIZE_N: gl.constexpr, BLOCK_SIZE_K: gl.constexpr,
    num_stages: gl.constexpr, SUBTILE_FACTOR: gl.constexpr, num_warps: gl.constexpr
):
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
        ping_bar, pong_bar, debug_ptr,
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
        (fa3_consumer_wg0_profiled, (p, SchedulerImpl, SEQ_LEN, NUM_HEADS, HEAD_DIM, p_layout, m_layout, s_layout)),
        (fa3_consumer_wg1_profiled, (p, SchedulerImpl, SEQ_LEN, NUM_HEADS, HEAD_DIM, p_layout, m_layout, s_layout)),
        (fa3_producer_partition, (p, SchedulerImpl, SEQ_LEN, NUM_HEADS, HEAD_DIM, p_layout, m_layout, s_layout)),
        (fa3_store_partition, (p, SchedulerImpl, SEQ_LEN, NUM_HEADS, HEAD_DIM, p_layout, m_layout, s_layout)),
    ], [num_warps, 1, 1], [240, 24, 24])

# ---------------------------------------------------------------------------
# HOST EXECUTION LAUNCHER (Identical to gluon_attention_qk_sparse.py)
# ---------------------------------------------------------------------------

def run_fa3_sparse_q_kernel_profiled(Q_dense, K, V, manual_config, debug_log):
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

    # 3. TMA Hook Setup & Launcher (Matching gluon_attention_qk_sparse.py)
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

    fa3_warp_specialized_kernel_profiled[grid](
        q0_desc, q1_desc, eq0_desc, eq1_desc, k_desc, v_desc, o0_desc, o1_desc,
        debug_log,
        GroupedPersistentTileScheduler(8),
        SEQ_LEN, HEAD_DIM, NUM_HEADS,
        BLOCK_SIZE_M=manual_config["BM"],
        BLOCK_SIZE_N=manual_config["BN"],
        BLOCK_SIZE_K=manual_config["BK"],
        num_stages=manual_config["num_stages"],
        SUBTILE_FACTOR=manual_config["SF"],
        num_warps=manual_config["warps"],
    )

    return O

# ---------------------------------------------------------------------------
# MATPLOTLIB TIMELINE PLOTTER
# ---------------------------------------------------------------------------

from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np

def plot_fa3_timeline(debug_log, seq_len, block_n, output_filename="fa3_recreated_timeline.png"):
    """Generates a clean, publication-ready Gantt chart with distinct sub-tracks per operation."""
    timestamps = debug_log.cpu().numpy()
    num_steps = seq_len // block_n

    valid_mask = timestamps > 0
    if not np.any(valid_mask):
        print("Error: No timestamps recorded!")
        return

    min_clock = timestamps[valid_mask].min()
    timestamps = np.where(valid_mask, timestamps - min_clock, 0)
    max_clock = timestamps.max()

    fig, ax = plt.subplots(figsize=(18, 8))
    
    colors = {
        'WGMMA0': '#e68a8a',   # Muted Red / Salmon (Sparse QK)
        'Softmax': '#b8d98d',  # Soft Green (Softmax)
        'WGMMA1': '#70a1d7'    # Soft Blue (Dense PV)
    }

    # Distinct Y-coordinates for each sub-track to eliminate ALL overlap:
    # WG1 (Warpgroup 2): y = 5.0 (QK), y = 4.1 (Softmax), y = 3.2 (PV)
    # WG0 (Warpgroup 1): y = 1.8 (QK), y = 0.9 (Softmax), y = 0.0 (PV)
    y_pos = {
        1: {'g0': 5.0, 'soft': 4.1, 'g1': 3.2},  # WG1
        0: {'g0': 1.8, 'soft': 0.9, 'g1': 0.0}   # WG0
    }
    bar_height = 0.65

    # 1. Background shading & divider line between Warpgroups
    ax.axhspan(-0.5, 2.4, color='#f8f9fa', zorder=0)
    ax.axhspan(2.6, 5.6, color='#f1f3f5', zorder=0)
    ax.axhline(2.5, color='#adb5bd', linestyle='-', linewidth=1.5, zorder=1)

    # 2. Plot Bars for WG0 and WG1
    for wg in range(2):
        for step in range(0, num_steps + 1):
            t_g0_s, t_g0_e, t_s_s, t_s_e, t_g1_s, t_g1_e = timestamps[wg, step]

            if step == 0:
                tag = "P"
            elif step == num_steps - 1:
                tag = "E1"
            elif step == num_steps:
                tag = "E2"
            else:
                tag = f"{step - 1}"

            min_text_width = max_clock * 0.012  # Don't render text inside tiny boxes

            # WGMMA0 (Sparse QK)
            if t_g0_e > t_g0_s:
                w = t_g0_e - t_g0_s
                ax.barh(y_pos[wg]['g0'], w, left=t_g0_s, color=colors['WGMMA0'],
                        edgecolor='black', height=bar_height, zorder=2)
                if w >= min_text_width:
                    ax.text(t_g0_s + w / 2, y_pos[wg]['g0'], tag,
                            va='center', ha='center', fontsize=8, fontweight='bold', color='black', zorder=3)

            # Softmax
            if t_s_e > t_s_s:
                w = t_s_e - t_s_s
                ax.barh(y_pos[wg]['soft'], w, left=t_s_s, color=colors['Softmax'],
                        edgecolor='black', height=bar_height, zorder=2)

            # WGMMA1 (Dense PV)
            if t_g1_e > t_g1_s:
                w = t_g1_e - t_g1_s
                ax.barh(y_pos[wg]['g1'], w, left=t_g1_s, color=colors['WGMMA1'],
                        edgecolor='black', height=bar_height, zorder=2)
                if w >= min_text_width:
                    ax.text(t_g1_s + w / 2, y_pos[wg]['g1'], tag,
                            va='center', ha='center', fontsize=8, fontweight='bold', color='black', zorder=3)

    # Corrected Ping-Pong Sync Lines
    for step in range(0, num_steps):
        # Ping Sync: WG0 arrives ping_bar at QK start (t_g0_s) -> WG1 unblocks at PV start (t_g1_s)
        t_ping_src = timestamps[0, step, 0]  # WG0 QK start / ping signal sent
        t_ping_dst = timestamps[1, step, 4]  # WG1 PV start / ping signal received

        if t_ping_src > 0 and t_ping_dst > 0:
            ax.annotate(
                '', xy=(t_ping_dst, y_pos[1]['g1'] - bar_height / 2),
                xytext=(t_ping_src, y_pos[0]['g0'] + bar_height / 2),
                arrowprops=dict(arrowstyle="->", color="#d9534f", lw=1.2, ls="--", alpha=0.85),
                zorder=4
            )

        # Pong Sync: WG1 arrives pong_bar at QK start (t_g0_s) -> WG0 unblocks at PV start (next step)
        if step < num_steps:
            t_pong_src = timestamps[1, step, 0]      # WG1 QK start / pong signal sent
            t_pong_dst = timestamps[0, step + 1, 4]  # WG0 next step PV start / pong signal received

            if t_pong_src > 0 and t_pong_dst > 0:
                ax.annotate(
                    '', xy=(t_pong_dst, y_pos[0]['g1'] + bar_height / 2),
                    xytext=(t_pong_src, y_pos[1]['g0'] - bar_height / 2),
                    arrowprops=dict(arrowstyle="->", color="#0275d8", lw=1.2, ls="--", alpha=0.85),
                    zorder=4
                )

    # 4. Y-Axis Customization
    y_ticks = [5.0, 4.1, 3.2, 1.8, 0.9, 0.0]
    y_tick_labels = [
        "WG1: WGMMA0 (QK)",
        "WG1: Softmax",
        "WG1: WGMMA1 (PV)",
        "WG0: WGMMA0 (QK)",
        "WG0: Softmax",
        "WG0: WGMMA1 (PV)"
    ]
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_tick_labels, fontsize=10, fontweight='bold')
    ax.set_ylim(-0.7, 5.8)
    ax.set_xlim(0, max_clock * 1.03)

    ax.set_xlabel("GPU Clock Cycles", fontsize=12, fontweight='bold')
    ax.set_title(f"Recreated FlashAttention-3 Parallel Pipeline Timeline (SEQ_LEN={seq_len})", fontsize=14, fontweight='bold')

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=colors['WGMMA0'], edgecolor='black', label='WGMMA0 (Sparse QK)'),
        Patch(facecolor=colors['Softmax'], edgecolor='black', label='Softmax (Vector ALU)'),
        Patch(facecolor=colors['WGMMA1'], edgecolor='black', label='WGMMA1 (Dense PV)'),
        Line2D([0], [0], color='#d9534f', lw=1.5, ls='--', label='Ping Sync (WG0 → WG1)'),
        Line2D([0], [0], color='#0275d8', lw=1.5, ls='--', label='Pong Sync (WG1 → WG0)'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', framealpha=0.95)
    ax.grid(axis='x', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_filename, dpi=300)
    print(f"Timeline plot saved to: {output_filename}")
# ---------------------------------------------------------------------------
# MAIN LAUNCHER
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Profile FA3 Sparse Q Timeline via %clock64")
    parser.add_argument("--bm", type=int, default=128, help="BLOCK_SIZE_M")
    parser.add_argument("--bn", type=int, default=128, help="BLOCK_SIZE_N")
    parser.add_argument("--bk", type=int, default=128, help="HEAD_DIM (BLOCK_SIZE_K)")
    parser.add_argument("--stages", type=int, default=2, help="Number of pipeline stages for KV")
    parser.add_argument("--sf", type=int, default=1, help="SUBTILE_FACTOR")
    parser.add_argument("--warps", type=int, default=4, help="Number of compute warps")
    parser.add_argument("--seq_len", type=int, default=2048, help="Sequence length")
    
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
    SEQ_LEN = args.seq_len
    HEAD_DIM = args.bk
    BATCH = max(1, 16384 // SEQ_LEN)

    print(f"\nProfiling Sparse Q FA3: BATCH={BATCH}, NUM_HEADS={NUM_HEADS}, SEQ_LEN={SEQ_LEN}, HEAD_DIM={HEAD_DIM}", flush=True)

    Q = torch.randn((BATCH, NUM_HEADS, SEQ_LEN, HEAD_DIM), device="cuda", dtype=torch.float16)
    K = torch.randn((BATCH, NUM_HEADS, SEQ_LEN, HEAD_DIM), device="cuda", dtype=torch.float16)
    V = torch.randn((BATCH, NUM_HEADS, SEQ_LEN, HEAD_DIM), device="cuda", dtype=torch.float16)

    debug_log = torch.zeros((2, 128, 6), dtype=torch.int64, device="cuda")

    run_fa3_sparse_q_kernel_profiled(Q, K, V, manual_config=manual_config, debug_log=debug_log)

    plot_fa3_timeline(debug_log, SEQ_LEN, manual_config["BN"], output_filename=f"fa3_timeline_seq{SEQ_LEN}.png")