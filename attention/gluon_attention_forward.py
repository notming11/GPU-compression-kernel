import argparse
import importlib.util
import os
import sys
import math
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

from common import (
    WGMMA,
    pick_wgmma_layout,
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
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

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
    q_desc: tma.tensor_descriptor
    k_desc: tma.tensor_descriptor
    v_desc: tma.tensor_descriptor
    o_desc: tma.tensor_descriptor

    q_buf: gl.shared_memory_descriptor
    k_bufs: gl.shared_memory_descriptor
    v_bufs: gl.shared_memory_descriptor
    o_bufs: gl.shared_memory_descriptor

    q_ready_bar: gl.shared_memory_descriptor
    q_empty_bar: gl.shared_memory_descriptor
    kv_empty_bars: gl.shared_memory_descriptor
    kv_ready_bars: gl.shared_memory_descriptor
    o_empty_bars: gl.shared_memory_descriptor
    o_ready_bars: gl.shared_memory_descriptor

    SUBTILE_FACTOR: gl.constexpr
    num_warps: gl.constexpr
    
    @gluon.constexpr_function
    def __init__(
        self, 
        q_desc, k_desc, v_desc, o_desc, 
        q_buf, k_bufs, v_bufs, o_bufs, 
        q_ready_bar, q_empty_bar, 
        kv_empty_bars, kv_ready_bars,
        o_empty_bars, o_ready_bars, 
        SUBTILE_FACTOR: gl.constexpr, 
        num_warps: gl.constexpr
    ):
        self.q_desc = q_desc
        self.k_desc = k_desc
        self.v_desc = v_desc
        self.o_desc = o_desc
        self.q_buf = q_buf
        self.k_bufs = k_bufs
        self.v_bufs = v_bufs
        self.o_bufs = o_bufs
        self.q_ready_bar = q_ready_bar
        self.q_empty_bar = q_empty_bar
        self.kv_empty_bars = kv_empty_bars
        self.kv_ready_bars = kv_ready_bars
        self.o_empty_bars = o_empty_bars
        self.o_ready_bars = o_ready_bars
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
        next_index = (self.index + 1) & (self.num_barriers - 1)
        next_phase = gl.where(next_index == 0, self.phase ^ 1, self.phase)
        return Counter(next_index, next_phase, self.num_barriers)

@gluon.jit
def _split_n(x, SUBTILE_FACTOR: gl.constexpr):
    split_count: gl.constexpr = SUBTILE_FACTOR.bit_length() - 1  # log2
    xs = (x, )
    for _ in gl.static_range(split_count):
        next_xs = ()
        for j in gl.static_range(len(xs)):
            x = xs[j]
            next_xs += x.reshape(x.shape[0], 2, x.shape[1] // 2).permute(0, 2, 1).split()
        xs = next_xs
    return xs

@gluon.jit
def store_acc_to_smem_subtile(p, acc, acc_state):
    accs = _split_n(acc, p.SUBTILE_FACTOR)

    for i in gl.static_range(p.SUBTILE_FACTOR):
        mbarrier.wait(p.o_empty_bars.index(acc_state.index), acc_state.phase)
        o_buf = p.o_bufs.index(acc_state.index)

        o_buf.store(accs[i])
        fence_async_shared()
        mbarrier.arrive(p.o_ready_bars.index(acc_state.index), count=1)
        acc_state = acc_state.next()

    return acc_state

# ---------------------------------------------------------------------------
# ATTENTION PARTITIONS
# ---------------------------------------------------------------------------

@gluon.jit
def fa3_producer_partition(p: PartitionArgs, SchedulerImpl: gl.constexpr, SEQ_LEN: gl.constexpr, NUM_HEADS: gl.constexpr, HEAD_DIM: gl.constexpr):
    BLOCK_M: gl.constexpr = p.q_desc.block_type.shape[0]
    BLOCK_N: gl.constexpr = p.k_desc.block_type.shape[1]
    BLOCK_K: gl.constexpr = p.q_desc.block_type.shape[1]

    scheduler = SchedulerImpl.initialize(p.o_desc.shape[0], p.o_desc.shape[1], BLOCK_M, BLOCK_K)

    kv_state = Counter.create(1, p.kv_empty_bars.shape[0])
    q_state = Counter.create(1, p.q_empty_bar.shape[0])

    for tile_idx in range(scheduler.get_num_tiles()):
        pid_m, bh_idx, global_m_offset = scheduler.get_tile(tile_idx, SEQ_LEN, BLOCK_M, NUM_HEADS)      
          
        mbarrier.wait(p.q_empty_bar.index(0), q_state.phase)
        
        q_bar = p.q_ready_bar.index(0)
        mbarrier.expect(q_bar, p.q_desc.block_type.nbytes)
        tma.async_copy_global_to_shared(p.q_desc, [global_m_offset, 0], q_bar, p.q_buf)
        
        kv_global_offset = bh_idx * SEQ_LEN
        num_steps = SEQ_LEN // BLOCK_N

        for step in range(num_steps):
            bar = p.kv_ready_bars.index(kv_state.index)
            mbarrier.wait(p.kv_empty_bars.index(kv_state.index), kv_state.phase)

            mbarrier.expect(bar, p.k_desc.block_type.nbytes + p.v_desc.block_type.nbytes)

            tma.async_copy_global_to_shared(p.k_desc, [0, kv_global_offset + step * BLOCK_N], bar, p.k_bufs.index(kv_state.index))
            tma.async_copy_global_to_shared(p.v_desc, [kv_global_offset + step * BLOCK_N, 0], bar, p.v_bufs.index(kv_state.index))
            
            kv_state = kv_state.next()
            
        q_state = q_state.next()

@gluon.jit
def fa3_consumer_partition(p: PartitionArgs, SchedulerImpl: gl.constexpr, SEQ_LEN: gl.constexpr, NUM_HEADS: gl.constexpr, HEAD_DIM: gl.constexpr, p_layout: gl.constexpr):
    BLOCK_M: gl.constexpr = p.q_desc.block_type.shape[0]
    BLOCK_N: gl.constexpr = p.k_desc.block_type.shape[1]
    BLOCK_K: gl.constexpr = p.q_desc.block_type.shape[1]  # Dynamically equals HEAD_DIM
    dtype: gl.constexpr = p.q_desc.dtype

    scheduler = SchedulerImpl.initialize(p.o_desc.shape[0], p.o_desc.shape[1], BLOCK_M, BLOCK_K)

    kv_state = Counter.create(0, p.kv_ready_bars.shape[0])
    acc_state = Counter.create(1, p.o_empty_bars.shape[0])
    q_state = Counter.create(0, p.q_empty_bar.shape[0])
    
    num_steps = SEQ_LEN // BLOCK_N

    # Pre-scale with log2(e) for fast hardware MUFU.EX2 (exp2) intrinsics
    LOG2E: gl.constexpr = 1.4426950408889634
    sm_scale_log2: gl.constexpr = (1.0 / math.sqrt(HEAD_DIM)) * LOG2E
    
    for tile_idx in range(scheduler.get_num_tiles()):
        pid_m, bh_idx, global_m_offset = scheduler.get_tile(tile_idx, SEQ_LEN, BLOCK_M, NUM_HEADS)  
        
        mma_o = WGMMA.initialize(dtype, BLOCK_M, BLOCK_K, p.num_warps)
        mma_s_dummy = WGMMA.initialize(dtype, BLOCK_M, BLOCK_N, p.num_warps)

        m_layout: gl.constexpr = gl.SliceLayout(dim=1, parent=mma_o.layout)
        s_layout: gl.constexpr = gl.SliceLayout(dim=1, parent=mma_s_dummy.layout)

        m_old = gl.full((BLOCK_M,), -float('inf'), dtype=gl.float32, layout=s_layout)
        l_old = gl.zeros((BLOCK_M,), dtype=gl.float32, layout=s_layout)
        
        mbarrier.wait(p.q_ready_bar.index(0), q_state.phase)

        for step in range(num_steps):
            mbarrier.wait(p.kv_ready_bars.index(kv_state.index), kv_state.phase)

            # 1. First WGMMA (S = Q * K^T)
            mma_s = WGMMA.initialize(dtype, BLOCK_M, BLOCK_N, p.num_warps)
            mma_s = mma_s.issue_async_mma(p.q_buf, p.k_bufs.index(kv_state.index))
            mma_s = mma_s.wait_num_outstanding(0)
            
            S_tile, mma_s = mma_s.take_result()
            S_tile = S_tile * sm_scale_log2
            
            # 2. Online Softmax Max Update & Rescaling
            m_new_tile = gl.max(S_tile, axis=1)
            m_new = gl.maximum(m_old, m_new_tile)
            
            rescale_factor = gl.exp2(m_old - m_new)
            rescale_factor_m = gl.convert_layout(rescale_factor, m_layout)
            
            # Unconditional accumulator scaling (on step 0, 0 * rescale = 0)
            mma_o = mma_o.wait_num_outstanding(0)
            o_acc, mma_o = mma_o.take_result()
            o_acc = o_acc * rescale_factor_m[:, None]
            mma_o = WGMMA(o_acc, gl.constexpr(True), mma_o.layout, BLOCK_M, BLOCK_K)
            
            # 3. Softmax Exponentiation & Sum
            P_tile_f32 = gl.exp2(S_tile - m_new[:, None])
            
            p_sum = gl.sum(P_tile_f32, axis=1)
            l_old = l_old * rescale_factor + p_sum
            m_old = m_new

            P_tile_f16 = gl.cast(P_tile_f32, dtype=dtype)
            P_tile_permuted = gl.convert_layout(P_tile_f16, p_layout)
            # P_tile_permuted = P_tile_f16

            # 4. Second WGMMA (O += P * V)
            mma_o = mma_o.issue_async_mma(P_tile_permuted, p.v_bufs.index(kv_state.index))

            mbarrier.arrive(p.kv_empty_bars.index(kv_state.index), count=1)
            kv_state = kv_state.next()
    
        mbarrier.arrive(p.q_empty_bar.index(0), count=1)
        q_state = q_state.next()

        # Output normalization
        mma_o = mma_o.wait_num_outstanding(0)
        acc, mma_o = mma_o.take_result()

        l_final_m = gl.convert_layout(l_old, m_layout)
        acc = acc / l_final_m[:, None]
        acc = acc.to(p.o_desc.dtype)

        acc_state = store_acc_to_smem_subtile(p, acc, acc_state)

@gluon.jit
def fa3_store_partition(p: PartitionArgs, SchedulerImpl: gl.constexpr, SEQ_LEN: gl.constexpr, NUM_HEADS: gl.constexpr, HEAD_DIM: gl.constexpr):
    BLOCK_M: gl.constexpr = p.o_desc.block_type.shape[0]
    SPLIT_K: gl.constexpr = p.o_desc.block_type.shape[1]
    BLOCK_K: gl.constexpr = SPLIT_K * p.SUBTILE_FACTOR

    scheduler = SchedulerImpl.initialize(p.o_desc.shape[0], p.o_desc.shape[1], BLOCK_M, BLOCK_K)
    state = Counter.create(0, p.o_empty_bars.shape[0])

    num_buffers: gl.constexpr = p.o_bufs.shape[0]
    outstanding_stores: gl.constexpr = num_buffers - 1
    store_iter = 0

    for tile_idx in range(scheduler.get_num_tiles()):
        pid_m, bh_idx, global_m_offset = scheduler.get_tile(tile_idx, SEQ_LEN, BLOCK_M, NUM_HEADS)  

        for i in gl.static_range(p.SUBTILE_FACTOR):
            mbarrier.wait(p.o_ready_bars.index(state.index), state.phase)
            o_buf = p.o_bufs.index(state.index)

            tma.async_copy_shared_to_global(p.o_desc, [global_m_offset, i * SPLIT_K], o_buf)

            if store_iter >= outstanding_stores:
                tma.store_wait(outstanding_stores)
                empty_idx = (store_iter - outstanding_stores) % num_buffers
                mbarrier.arrive(p.o_empty_bars.index(empty_idx), count=1)

            state = state.next()
            store_iter += 1

    tma.store_wait(0)

# ---------------------------------------------------------------------------
# KERNEL LAUNCHER
# ---------------------------------------------------------------------------
@gluon.jit
def fa3_warp_specialized_kernel(
    q_desc, k_desc, v_desc, o_desc, 
    SchedulerImpl: gl.constexpr,
    SEQ_LEN: gl.constexpr, HEAD_DIM: gl.constexpr, NUM_HEADS: gl.constexpr, 
    BLOCK_SIZE_M: gl.constexpr, BLOCK_SIZE_N: gl.constexpr, BLOCK_SIZE_K: gl.constexpr,
    num_stages: gl.constexpr, SUBTILE_FACTOR: gl.constexpr, num_warps: gl.constexpr
):
    dtype: gl.constexpr = q_desc.dtype

    q_buf = gl.allocate_shared_memory(dtype, q_desc.block_type.shape, q_desc.layout)
    k_bufs = gl.allocate_shared_memory(dtype, [num_stages] + k_desc.block_type.shape, k_desc.layout)
    v_bufs = gl.allocate_shared_memory(dtype, [num_stages] + v_desc.block_type.shape, v_desc.layout)
    o_bufs = gl.allocate_shared_memory(dtype, [2] + o_desc.block_type.shape, o_desc.layout)

    q_ready_bar = gl.allocate_shared_memory(gl.int64, [1, 1], mbarrier.MBarrierLayout())
    q_empty_bar = gl.allocate_shared_memory(gl.int64, [1, 1], mbarrier.MBarrierLayout())
    
    kv_empty_bars = gl.allocate_shared_memory(gl.int64, [num_stages, 1], mbarrier.MBarrierLayout())
    kv_ready_bars = gl.allocate_shared_memory(gl.int64, [num_stages, 1], mbarrier.MBarrierLayout())

    o_empty_bars = gl.allocate_shared_memory(gl.int64, [2, 1], mbarrier.MBarrierLayout())
    o_ready_bars = gl.allocate_shared_memory(gl.int64, [2, 1], mbarrier.MBarrierLayout())

    mbarrier.init(q_ready_bar.index(0), count=1)
    mbarrier.init(q_empty_bar.index(0), count=1)

    for i in gl.static_range(num_stages):
        mbarrier.init(kv_empty_bars.index(i), count=1)
        mbarrier.init(kv_ready_bars.index(i), count=1)

    for i in gl.static_range(2):
        mbarrier.init(o_empty_bars.index(i), count=1)
        mbarrier.init(o_ready_bars.index(i), count=1)

    p = PartitionArgs(
        q_desc, k_desc, v_desc, o_desc,
        q_buf, k_bufs, v_bufs, o_bufs,
        q_ready_bar, q_empty_bar, 
        kv_empty_bars, kv_ready_bars,
        o_empty_bars, o_ready_bars,
        SUBTILE_FACTOR, num_warps
    )
    
    p_layout: gl.constexpr = gl.DotOperandLayout(
                operand_index=0,
                parent=pick_wgmma_layout(dtype, BLOCK_SIZE_M, BLOCK_SIZE_K, num_warps),
                k_width=32 // dtype.primitive_bitwidth,
                meta=0,
            )

    gl.warp_specialize([
        (fa3_consumer_partition, (p, SchedulerImpl, SEQ_LEN, NUM_HEADS, HEAD_DIM, p_layout)),
        (fa3_producer_partition, (p, SchedulerImpl, SEQ_LEN, NUM_HEADS, HEAD_DIM)),
        (fa3_store_partition, (p, SchedulerImpl, SEQ_LEN, NUM_HEADS, HEAD_DIM)),
    ], [1, 1], [24, 24])


def fa3_get_configs(pre_hook=None, tune=True):
    def valid(BM, BN, BK, warps, num_stages, SF):
        # if BM == 128 and BN == 256:
        #     return False

        # Shared Memory Calculation for 3-partition layout
        fp16_elements = (
            (1 * BM * BK) +                # q_buf (single resident tile)
            (num_stages * BK * BN) +       # k_bufs (circular)
            (num_stages * BN * BK) +       # v_bufs (circular)
            (2 * BM * (BK // SF))          # o_bufs (subtitled output)
        )
        fp16_smem_bytes = 2 * fp16_elements
        barrier_bytes = 8 * (1 + (2 * num_stages) + 4)

        total_smem_bytes = fp16_smem_bytes + barrier_bytes
        if total_smem_bytes > 232448:      # Hopper SMEM Ceiling (~227 KB)
            return False

        if BK % SF != 0:
            return False

        split_k = BK // SF
        if split_k < 16:
            return False

        # Warp allocation bounds
        warps_m = 4
        warps_n = 1
        m = 16
        while (warps_m * warps_n) != warps:
            if BM > m * warps_m:
                warps_m *= 2
            else:
                warps_n *= 2

        if SF > 1 and warps_n > 1:
            return False
        if BM < warps_m * 16 or BN < warps_n * 16:
            return False

        # Register pressure check
        elements_per_thread = (BM * max(BN, BK)) / (warps * 32)
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
        for BM in (64, 128, 256)
        for BN in (64, 128, 256)
        for BK in (64, 128, 256)
        for warps in (4, 8, )
        for num_stages in (2, 4)
        for SF in (1, 2, 4, 8, )
        if valid(BM, BN, BK, warps, num_stages, SF)
    ]
    
    return configs if tune else configs[:1]


def fa3_tma_set_block_size_hook(nargs):
    block_m = nargs["BLOCK_SIZE_M"]
    block_n = nargs["BLOCK_SIZE_N"]
    block_k = nargs["BLOCK_SIZE_K"]
    split_k = nargs["BLOCK_SIZE_K"] // nargs["SUBTILE_FACTOR"]

    nargs["q_desc"].block_shape = [block_m, block_k]
    nargs["k_desc"].block_shape = [block_k, block_n]
    nargs["v_desc"].block_shape = [block_n, block_k]
    nargs["o_desc"].block_shape = [block_m, split_k]

    nargs["q_desc"].layout = gl.NVMMASharedLayout.get_default_for(nargs["q_desc"].block_shape, gl.float16)
    nargs["k_desc"].layout = gl.NVMMASharedLayout.get_default_for(nargs["k_desc"].block_shape, gl.float16)
    nargs["v_desc"].layout = gl.NVMMASharedLayout.get_default_for(nargs["v_desc"].block_shape, gl.float16)
    nargs["o_desc"].layout = gl.NVMMASharedLayout.get_default_for(nargs["o_desc"].block_shape, gl.float16)


_autotune_cache = {}

def get_autotuned_kernel(head_dim: int):
    if head_dim not in _autotune_cache:
        # Filter configurations matching the dynamic HEAD_DIM
        configs = [
            config for config in fa3_get_configs(pre_hook=fa3_tma_set_block_size_hook, tune=True)
            if config.kwargs["BLOCK_SIZE_K"] == head_dim
        ]
        
        _autotune_cache[head_dim] = triton.autotune(
            configs=configs,
            key=["SEQ_LEN"],
            do_bench=lambda kernel_call, quantiles: triton.testing.do_bench_cudagraph(
                kernel_call, rep=100, quantiles=quantiles
            ),
        )(fa3_warp_specialized_kernel)
        
    return _autotune_cache[head_dim]


def run_fa3_kernel(Q, K, V, tune=True, manual_config=None):
    BATCH, NUM_HEADS, SEQ_LEN, HEAD_DIM = Q.shape
    O = torch.empty_like(Q)
    
    Q_flat = Q.reshape(-1, HEAD_DIM)
    K_flat = K.reshape(-1, HEAD_DIM)
    V_flat = V.reshape(-1, HEAD_DIM)
    O_flat = O.reshape(-1, HEAD_DIM)
    K_T = K_flat.transpose(0, 1).contiguous()

    dummy_block = [1, 1]
    dummy_layout = gl.NVMMASharedLayout.get_default_for(dummy_block, gl.float16)

    q_desc = TensorDescriptor.from_tensor(Q_flat, dummy_block, dummy_layout)
    k_desc = TensorDescriptor.from_tensor(K_T, dummy_block, dummy_layout)
    v_desc = TensorDescriptor.from_tensor(V_flat, dummy_block, dummy_layout)
    o_desc = TensorDescriptor.from_tensor(O_flat, dummy_block, dummy_layout)

    if tune:
        kernel = get_autotuned_kernel(HEAD_DIM)
        def grid(meta):
            num_sms = torch.cuda.get_device_properties("cuda").multi_processor_count
            num_pid = triton.cdiv(SEQ_LEN, meta["BLOCK_SIZE_M"])
            total_tiles = num_pid * BATCH * NUM_HEADS
            return (min(num_sms, total_tiles), )

        kernel[grid](
            q_desc, k_desc, v_desc, o_desc,
            GroupedPersistentTileScheduler(8),
            SEQ_LEN, HEAD_DIM, NUM_HEADS
        )
    else:
        manual_config["BK"] = HEAD_DIM
        hook_kwargs = {
            "BLOCK_SIZE_M": manual_config["BM"],
            "BLOCK_SIZE_N": manual_config["BN"],
            "BLOCK_SIZE_K": manual_config["BK"],
            "SUBTILE_FACTOR": manual_config["SF"],
            "q_desc": q_desc, "k_desc": k_desc, "v_desc": v_desc, "o_desc": o_desc
        }
        fa3_tma_set_block_size_hook(hook_kwargs)

        num_sms = torch.cuda.get_device_properties("cuda").multi_processor_count
        num_pid = triton.cdiv(SEQ_LEN, manual_config["BM"])
        total_tiles = num_pid * BATCH * NUM_HEADS
        grid = (min(num_sms, total_tiles), )

        fa3_warp_specialized_kernel[grid](
            q_desc, k_desc, v_desc, o_desc,
            GroupedPersistentTileScheduler(8),
            SEQ_LEN, HEAD_DIM, NUM_HEADS,
            BLOCK_SIZE_M=manual_config["BM"],
            BLOCK_SIZE_N=manual_config["BN"],
            BLOCK_SIZE_K=manual_config["BK"],
            num_stages=manual_config["num_stages"],
            SUBTILE_FACTOR=manual_config["SF"],
            num_warps=manual_config["warps"]
        )

    return O

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run FlashAttention-3 Warp-Specialized Kernel")
    parser.add_argument("--tune", action="store_true", help="Enable Triton autotuning")
    
    parser.add_argument("--bm", type=int, default=128, help="BLOCK_SIZE_M")
    parser.add_argument("--bn", type=int, default=128, help="BLOCK_SIZE_N")
    parser.add_argument("--bk", type=int, default=128, help="BLOCK_SIZE_N")
    parser.add_argument("--stages", type=int, default=2, help="Number of pipeline stages for KV")
    parser.add_argument("--sf", type=int, default=2, help="SUBTILE_FACTOR")
    parser.add_argument("--warps", type=int, default=8, help="Number of compute warps")
    
    args = parser.parse_args()

    manual_config = {
        "BM": args.bm,
        "BN": args.bn,
        "BK": args.bk,
        "warps": args.warps,
        "num_stages": args.stages,
        "SF": args.sf
    }

    if args.tune:
        print("Running FlashAttention-3. Autotuning enabled.")
    else:
        print(f"Running FlashAttention-3 with manual config: {manual_config}")
        
    BATCH, NUM_HEADS = 2, 16
    sizes = [
        (4096, 128),
        # (256, 64),
        # (512, 128),
        # (8192, 256)
    ]
    
    torch.set_printoptions(profile="full")
    torch.set_printoptions(linewidth=20000)
    
    os.environ["MLIR_ENABLE_DUMP"]="1"
    os.environ["MLIR_DUMP_PATH"] = "./MLIR_DUMP/3_partition_4096_128"
    os.makedirs(os.path.dirname(os.environ["MLIR_DUMP_PATH"]), exist_ok=True)

    for SEQ_LEN, HEAD_DIM in sizes:
        BATCH = max(1, 16384//SEQ_LEN)
        print(f"Testing BATCH={BATCH}, NUM_HEADS={NUM_HEADS}, SEQ_LEN={SEQ_LEN}, HEAD_DIM={HEAD_DIM}")
        
        Q = torch.randn((BATCH, NUM_HEADS, SEQ_LEN, HEAD_DIM), device="cuda", dtype=torch.float16)
        K = torch.randn((BATCH, NUM_HEADS, SEQ_LEN, HEAD_DIM), device="cuda", dtype=torch.float16)
        V = torch.randn((BATCH, NUM_HEADS, SEQ_LEN, HEAD_DIM), device="cuda", dtype=torch.float16)
        
        O_triton = run_fa3_kernel(Q, K, V, tune=args.tune, manual_config=manual_config)
        O_torch = torch.nn.functional.scaled_dot_product_attention(Q, K, V)
        
        torch.testing.assert_close(O_torch, O_triton, rtol=1e-2, atol=1e-2)
    
    print("Done. PyTorch reference matches Triton Gluon FA3 across dynamic head dimensions!")