import argparse
import torch
import triton
import math

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
)

import os
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
    # Bind this as a constexpr so it can be captured.
    GROUP_SIZE_M = gl.constexpr(GROUP_SIZE_M)

    # Like C++ templates!
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

            # Extract sequence block index and batch/head index
            pid_m = tile_id % num_pid_m
            batch_head_idx = tile_id // num_pid_m
    
            # Calculate global matrix offset in flattened DRAM (BATCH * NUM_HEADS * SEQ_LEN, HEAD_DIM)
            global_m_offset = (batch_head_idx * SEQ_LEN) + (pid_m * BLOCK_M)
    
            return pid_m, batch_head_idx, global_m_offset

    GroupedPersistentTileSchedulerImpl.__name__ = f"GroupedPersistentTileScheduler({GROUP_SIZE_M.value})"
    return GroupedPersistentTileSchedulerImpl

@aggregate 
class PartitionArgs:
    # ---------------------------------------------------------
    # 1. Swap Descriptors (TMA Configurations)
    # ---------------------------------------------------------
    # Note: Host-side TMA creation for k_desc must account for K^T layout mapping
    q_desc: tma.tensor_descriptor
    k_desc: tma.tensor_descriptor
    v_desc: tma.tensor_descriptor
    o_desc: tma.tensor_descriptor

    # ---------------------------------------------------------
    # 2. Buffer Allocation (Shared Memory)
    # ---------------------------------------------------------
    # Single resident buffer for Q (stays in SMEM for the inner loop)
    q_buf: gl.shared_memory_descriptor
    
    # Circular multi-staged buffers (3-5 stages) for K and V
    k_bufs: gl.shared_memory_descriptor
    v_bufs: gl.shared_memory_descriptor
    
    # Final output buffer for normalized O before TMA store
    o_bufs: gl.shared_memory_descriptor

    # ---------------------------------------------------------
    # 3. MBarrier Setup (Async Transaction Barriers)
    # ---------------------------------------------------------
    # Single barrier for the resident Q tile
    q_ready_bar: gl.shared_memory_descriptor
    q_empty_bar: gl.shared_memory_descriptor
    
    # Transaction barriers for the K and V circular pipelines
    # Grouped together since K and V stages advance synchronously 
    kv_empty_bars: gl.shared_memory_descriptor
    kv_ready_bars: gl.shared_memory_descriptor
    o_empty_bars: gl.shared_memory_descriptor
    o_ready_bars: gl.shared_memory_descriptor

    # ---------------------------------------------------------
    # Core Constants
    # ---------------------------------------------------------
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
        # ---------------------------------------------------------
        # Bind TMA Descriptors & Constants
        # ---------------------------------------------------------
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
        incr = self.index + gl.where(pred, 1, 0)
        rollover = incr == self.num_barriers
        index = gl.where(rollover, 0, incr)
        phase = gl.where(rollover, self.phase ^ 1, self.phase)
        return Counter(index, phase, self.num_barriers)

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
def fa3_producer_partition(p: PartitionArgs, SchedulerImpl: gl.constexpr, SEQ_LEN: gl.constexpr, NUM_HEADS: gl.constexpr):
    # Extract Block Dimensions
    BLOCK_M: gl.constexpr = p.q_desc.block_type.shape[0]
    BLOCK_N: gl.constexpr = p.k_desc.block_type.shape[1]
    BLOCK_K: gl.constexpr = p.q_desc.block_type.shape[1]

    # Initialize the scheduler using the final output shape and block dims
    scheduler = SchedulerImpl.initialize(p.o_desc.shape[0], p.o_desc.shape[1], BLOCK_M, BLOCK_K)

    # Initialize Phase/Index Counter for the KV pipeline empty barriers
    # Phase starts at 1 because empty barriers are pre-initialized (already "arrived")
    kv_state = Counter.create(1, p.kv_empty_bars.shape[0])
    q_state = Counter.create(1, p.q_empty_bar.shape[0])

    for tile_idx in range(scheduler.get_num_tiles()):
        # ---------------------------------------------------------
        # 1. Outer Loop Initialization
        # ---------------------------------------------------------
        pid_m, bh_idx, global_m_offset = scheduler.get_tile(tile_idx, SEQ_LEN, BLOCK_M, NUM_HEADS)      
          
        mbarrier.wait(p.q_empty_bar.index(0), q_state.phase)
        
        # Issue a single TMA load for the resident Q tile for this block
        q_bar = p.q_ready_bar.index(0)
        mbarrier.expect(q_bar, p.q_desc.block_type.nbytes)
        tma.async_copy_global_to_shared(p.q_desc, [global_m_offset, 0], q_bar, p.q_buf)
        
        kv_global_offset = bh_idx * SEQ_LEN

        # ---------------------------------------------------------
        # 2. Inner Loop (Sequence Length)
        # ---------------------------------------------------------
        num_steps = SEQ_LEN // BLOCK_N

        for step in range(num_steps):
            # Wait for the current stage's buffer to be empty/released by the Consumer
            bar = p.kv_ready_bars.index(kv_state.index)
            mbarrier.wait(p.kv_empty_bars.index(kv_state.index), kv_state.phase)

            # Tell the barrier how many bytes to expect (K + V TMA loads)
            mbarrier.expect(bar, p.k_desc.block_type.nbytes + p.v_desc.block_type.nbytes)

            # ---------------------------------------------------------
            # 3. Async Fetches
            # ---------------------------------------------------------
            # Issue TMA load for K_j into the circular buffer slot
            tma.async_copy_global_to_shared(p.k_desc, [0, kv_global_offset + step * BLOCK_N], bar, p.k_bufs.index(kv_state.index))
            
            # Issue TMA load for V_j into the circular buffer slot
            tma.async_copy_global_to_shared(p.v_desc, [kv_global_offset + step * BLOCK_N, 0], bar, p.v_bufs.index(kv_state.index))
            
            # Advance the Counter for the next stage
            kv_state = kv_state.next()
            
        q_state = q_state.next()

@gluon.jit
def fa3_consumer_partition(p: PartitionArgs, SchedulerImpl: gl.constexpr, SEQ_LEN: gl.constexpr, NUM_HEADS: gl.constexpr):
    # Extract Block Dimensions
    BLOCK_M: gl.constexpr = p.q_desc.block_type.shape[0]
    BLOCK_N: gl.constexpr = p.k_desc.block_type.shape[1]
    BLOCK_K: gl.constexpr = p.q_desc.block_type.shape[1]
    SPLIT_K: gl.constexpr = p.o_desc.block_type.shape[1]
    SEQ_LEN_N = p.k_desc.shape[1]
    dtype: gl.constexpr = p.q_desc.dtype

    # Initialize the scheduler
    scheduler = SchedulerImpl.initialize(p.o_desc.shape[0], p.o_desc.shape[1], BLOCK_M, BLOCK_K)

    # Initialize Phase/Index Counter for the circular K/V buffers
    kv_state = Counter.create(0, p.kv_ready_bars.shape[0])
    # Initialize output state counter (phase 1 because o_empty_bars are pre-initialized)
    acc_state = Counter.create(1, p.o_empty_bars.shape[0])
    q_state = Counter.create(0, p.q_empty_bar.shape[0])
    
    num_steps = SEQ_LEN // BLOCK_N

    for tile_idx in range(scheduler.get_num_tiles()):
        pid_m, bh_idx, global_m_offset = scheduler.get_tile(tile_idx, SEQ_LEN, BLOCK_M, NUM_HEADS)  
        # ---------------------------------------------------------
        # 1. Outer Loop Initialization
        # ---------------------------------------------------------
        # Initialize dummy WGMMA objects to extract both 1D slice layouts
        mma_o = WGMMA.initialize(dtype, BLOCK_M, BLOCK_K, p.num_warps)
        mma_s_dummy = WGMMA.initialize(dtype, BLOCK_M, BLOCK_N, p.num_warps)

        # m_layout is for O (M x K), s_layout is for S (M x N)
        m_layout: gl.constexpr = gl.SliceLayout(dim=1, parent=mma_o.layout)
        s_layout: gl.constexpr = gl.SliceLayout(dim=1, parent=mma_s_dummy.layout)

        # Initialize stats natively in s_layout
        m_old = gl.full((BLOCK_M,), -float('inf'), dtype=gl.float32, layout=s_layout)
        l_old = gl.zeros((BLOCK_M,), dtype=gl.float32, layout=s_layout)
        
        mbarrier.wait(p.q_ready_bar.index(0), q_state.phase)

        # ---------------------------------------------------------
        # 2. Inner Loop (Sequence Length)
        # ---------------------------------------------------------
        for step in range(num_steps):
            
            # Wait for K_j and V_j to be fully loaded into the current circular buffer slot
            mbarrier.wait(p.kv_ready_bars.index(kv_state.index), kv_state.phase)

            # --- First WGMMA (S = Q * K^T) ---
            # Initialize mma_s inside the loop because S is computed fresh per N-block
            mma_s = WGMMA.initialize(dtype, BLOCK_M, BLOCK_N, p.num_warps)
            mma_s = mma_s.issue_async_mma(p.q_buf, p.k_bufs.index(kv_state.index))
            mma_s = mma_s.wait_num_outstanding(0)
            
            
            # Extract accumulator registers from mma_s via take_result
            S_tile, mma_s = mma_s.take_result()
            S_tile = S_tile / gl.sqrt(gl.cast(p.q_desc.shape[1], gl.float32))
            
            # # Compute true M-block index for the tile
            # off_m = pid_m * BLOCK_M + gl.arange(0, BLOCK_M)
            # off_n = step * BLOCK_N + gl.arange(0, BLOCK_N)

            # # Stride by SEQ_LEN_N (256) instead of BLOCK_N (128)
            # s_ptrs = s_ptr + (off_m[:, None] * SEQ_LEN_N) + off_n[None, :]
            # gl.store(s_ptrs, gl.cast(S_tile, dtype))
            
            # --- Online Softmax & Rescaling (SIMT) ---
            m_new_tile = gl.max(S_tile, axis=1)
            # m_new_tile = gl.convert_layout(m_new_tile, m_layout)
            m_new = gl.maximum(m_old, m_new_tile)
            
            # Rescale the existing O accumulators
            rescale_factor = gl.exp(m_old - m_new)
            rescale_factor_m = gl.convert_layout(rescale_factor, m_layout)
            
            mma_o = mma_o.wait_num_outstanding(0)
            
            o_acc, mma_o = mma_o.take_result()
            o_acc = o_acc * rescale_factor_m[:, None]
            mma_o = WGMMA(o_acc, gl.to_tensor(True), mma_o.layout, BLOCK_M, BLOCK_K)
            
            # Compute unnormalized probabilities
            P_tile_f32 = gl.exp(S_tile - m_new[:, None])
            
            # Update running denominator
            l_old = l_old * rescale_factor + gl.sum(P_tile_f32, axis=1)
            m_old = m_new

            # --- The Layout Permutation ---
            P_tile_f16 = gl.cast(P_tile_f32, dtype=dtype)
            
            
            p_layout: gl.constexpr = gl.DotOperandLayout(
                operand_index=0,
                parent=mma_o.layout,
                k_width=32 // dtype.primitive_bitwidth,
                meta=0,
            )
            P_tile_permuted = gl.convert_layout(P_tile_f16, p_layout)
            

            # --- Second WGMMA (O += P * V) ---
            # Issue the second WGMMA using the permuted P registers and V_j from SMEM.
            # This inherently accumulates into mma_o's existing state.
            mma_o = mma_o.issue_async_mma(P_tile_permuted, p.v_bufs.index(kv_state.index))

            # --- Buffer Release & State Advancement ---
            # Arrive on the empty barrier to signal the producer that this stage is free
            mbarrier.arrive(p.kv_empty_bars.index(kv_state.index), count=1)
            
            # Advance circular buffer phase/index tracker
            kv_state = kv_state.next()
    
        mbarrier.arrive(p.q_empty_bar.index(0), count=1)
        q_state = q_state.next()
        # ---------------------------------------------------------
        # 3. Output: Normalize, cast, store to SMEM, signal store partition
        # ---------------------------------------------------------
        mma_o = mma_o.wait_num_outstanding(0)
        acc, mma_o = mma_o.take_result()

        # Normalize by the softmax denominator and cast to output dtype
        l_final_m = gl.convert_layout(l_old, m_layout)
        acc = acc / l_final_m[:, None]
        acc = acc.to(p.o_desc.dtype)

        acc_state = store_acc_to_smem_subtile(p, acc, acc_state)

@gluon.jit
def fa3_store_partition(p: PartitionArgs, SchedulerImpl: gl.constexpr, SEQ_LEN: gl.constexpr, NUM_HEADS: gl.constexpr):
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
        off_m = pid_m * BLOCK_M

        for i in gl.static_range(p.SUBTILE_FACTOR):
            # Wait for Consumer to finish math and store to this SMEM buffer
            mbarrier.wait(p.o_ready_bars.index(state.index), state.phase)
            o_buf = p.o_bufs.index(state.index)

            # Issue TMA Store
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
    # gl.static_print(f"BM: {BLOCK_SIZE_M}, BN: {BLOCK_SIZE_N}, BK: {BLOCK_SIZE_K}, warps: {num_warps}, buf: {num_stages}")
    dtype: gl.constexpr = q_desc.dtype

    # ---------------------------------------------------------
    # 1. Allocate Shared Memory Buffers
    # ---------------------------------------------------------
    # Q Buffer (Single stage, resident)
    q_buf = gl.allocate_shared_memory(dtype, q_desc.block_type.shape, q_desc.layout)
    
    # K and V Buffers (Circular, `num_stages`)
    k_bufs = gl.allocate_shared_memory(dtype, [num_stages] + k_desc.block_type.shape, k_desc.layout)
    v_bufs = gl.allocate_shared_memory(dtype, [num_stages] + v_desc.block_type.shape, v_desc.layout)
    
    # Output Buffers (Double-buffered for the decoupled Store partition)
    o_bufs = gl.allocate_shared_memory(dtype, [2] + o_desc.block_type.shape, o_desc.layout)

    # ---------------------------------------------------------
    # 2. Allocate & Initialize Transaction Barriers
    # ---------------------------------------------------------
    q_ready_bar = gl.allocate_shared_memory(gl.int64, [1, 1], mbarrier.MBarrierLayout())
    q_empty_bar = gl.allocate_shared_memory(gl.int64, [1, 1], mbarrier.MBarrierLayout())
    
    kv_empty_bars = gl.allocate_shared_memory(gl.int64, [num_stages, 1], mbarrier.MBarrierLayout())
    kv_ready_bars = gl.allocate_shared_memory(gl.int64, [num_stages, 1], mbarrier.MBarrierLayout())

    o_empty_bars = gl.allocate_shared_memory(gl.int64, [2, 1], mbarrier.MBarrierLayout())
    o_ready_bars = gl.allocate_shared_memory(gl.int64, [2, 1], mbarrier.MBarrierLayout())

    # Initialize Q ready barrier
    mbarrier.init(q_ready_bar.index(0), count=1)
    mbarrier.init(q_empty_bar.index(0), count=1)

    # Initialize KV pipeline barriers
    for i in gl.static_range(num_stages):
        mbarrier.init(kv_empty_bars.index(i), count=1)
        mbarrier.init(kv_ready_bars.index(i), count=1)

    # Initialize output pipeline barriers
    for i in gl.static_range(2):
        mbarrier.init(o_empty_bars.index(i), count=1)
        mbarrier.init(o_ready_bars.index(i), count=1)

    # ---------------------------------------------------------
    # 3. Bind Partition Arguments
    # ---------------------------------------------------------
    # Make sure your @aggregate class PartitionArgs matches this exact order!
    p = PartitionArgs(
        q_desc, k_desc, v_desc, o_desc,
        q_buf, k_bufs, v_bufs, o_bufs,
        q_ready_bar, q_empty_bar, 
        kv_empty_bars, kv_ready_bars,
        o_empty_bars, o_ready_bars,
        SUBTILE_FACTOR, num_warps
    )

    # ---------------------------------------------------------
    # 4. Launch Warp-Specialized Partitions
    # ---------------------------------------------------------
    # This assigns the warps across your 3 defined functions.
    # The Consumer (math) will natively claim the bulk of the warps you configured,
    # while the Producer (load) and Store (store) sit in the remaining WG slots.
    gl.warp_specialize([
        (fa3_consumer_partition, (p, SchedulerImpl, SEQ_LEN, NUM_HEADS)),
        (fa3_producer_partition, (p, SchedulerImpl, SEQ_LEN, NUM_HEADS)),
        (fa3_store_partition, (p, SchedulerImpl, SEQ_LEN, NUM_HEADS)),
    ], [1, 1], [24, 24])

def fa3_get_configs(pre_hook=None, tune=True):
    def valid(BM, BN, BK, warps, num_stages, SF):
        # ---------------------------------------------------------
        # 1. Shared Memory Footprint Calculation (FP16/BF16 = 2 bytes)
        # ---------------------------------------------------------
        # q_buf: 1 stage of size (BM * BK)
        # k_bufs: `num_stages` of size (BK * BN)
        # v_bufs: `num_stages` of size (BN * BK)
        # o_bufs: 2 stages of size (BM * BK) for the decoupled Store Partition
        tensor_smem_bytes = 2 * (
            (1 * BM * BK) +               # q_buf: single resident tile
            (num_stages * BK * BN) +       # k_bufs: circular pipeline
            (num_stages * BN * BK) +       # v_bufs: circular pipeline
            (2 * BM * (BK // SF))          # o_bufs: 2 subtile-sized output slots
        )
        
        # MBarriers: q_ready(1) + kv_empty(num_stages) + kv_ready(num_stages) + o_empty(2) + o_ready(2)
        # Each mbarrier is 8 bytes in Triton
        barrier_bytes = 8 * (1 + (2 * num_stages) + 4)
        
        smem_bytes = tensor_smem_bytes + barrier_bytes
        
        # Hopper max shared memory per block is ~227KB (232448 bytes)
        if smem_bytes > 232448: return False

        # ---------------------------------------------------------
        # 2. Warp Specialization & Subtile Constraints
        # ---------------------------------------------------------
        warps_m = 4
        warps_n = 1
        m = 16
        while (warps_m * warps_n) != warps:
            if BM > m * warps_m:
                warps_m *= 2
            else:
                warps_n *= 2

        if SF > 1 and warps_n > 1: return False
        if (BN // SF) < 16: return False
        if BM < warps_m * 16 or BN < warps_n * 16: return False

        # ---------------------------------------------------------
        # 3. Register Pressure Heuristic
        # ---------------------------------------------------------
        # FA3 has high register pressure in the Consumer WG due to holding
        # O accumulators (BM x BK), S transient accumulators (BM x BN), and SIMT softmax math.
        elements_per_thread = (BM * max(BN, BK)) / (warps * 32)
        
        # Bumped base required registers slightly (+64) for online softmax f32 overhead
        required_regs = elements_per_thread + 64 
        
        max_regs_per_thread = 65536 // (warps * 32)
        max_regs_per_thread = min(255, max_regs_per_thread)
        
        if required_regs > max_regs_per_thread: return False
        if elements_per_thread < 16: return False
        
        return True

    # ---------------------------------------------------------
    # Generate Config Space
    # ---------------------------------------------------------
    configs = [
        triton.Config(
            {
                "BLOCK_SIZE_M": BM,
                "BLOCK_SIZE_N": BN,
                "BLOCK_SIZE_K": BK,          # In FA, this is the Head Dimension
                "num_stages": num_stages,    # Defines K and V circular buffers
                "SUBTILE_FACTOR": SF,
            },
            num_warps=warps,
            pre_hook=pre_hook,
        )
        for BM in (64, 128, 256)
        for BN in (64, 128, 256)
        # Head dimensions (BK) are usually fixed based on model architecture (e.g., Llama is 128)
        for BK in (64, 128) 
        for warps in (4, 8)
        # Tested stages for K/V inner loop pipelining
        for num_stages in (2, 3, 4, 5) 
        for SF in (1, 2, 4, 8)
        if valid(BM, BN, BK, warps, num_stages, SF)
    ]
    
    return configs if tune else configs[:1]

def fa3_tma_set_block_size_hook(nargs):
    block_m = nargs["BLOCK_SIZE_M"]
    block_n = nargs["BLOCK_SIZE_N"]
    block_k = nargs["BLOCK_SIZE_K"]  # Head Dimension (e.g., 64 or 128)
    split_k = nargs["BLOCK_SIZE_K"] // nargs["SUBTILE_FACTOR"]

    # Configure Block Shapes for Q, K, V, and O descriptors
    # Note: K descriptor block shape is transposed [head_dim, block_n] for Q * K^T layout mapping
    nargs["q_desc"].block_shape = [block_m, block_k]
    nargs["k_desc"].block_shape = [block_k, block_n]
    nargs["v_desc"].block_shape = [block_n, block_k]
    nargs["o_desc"].block_shape = [block_m, split_k]

    # Assign NVMMA Shared Layouts optimized for Hopper WGMMA instructions
    nargs["q_desc"].layout = gl.NVMMASharedLayout.get_default_for(nargs["q_desc"].block_shape, gl.float16)
    nargs["k_desc"].layout = gl.NVMMASharedLayout.get_default_for(nargs["k_desc"].block_shape, gl.float16)
    nargs["v_desc"].layout = gl.NVMMASharedLayout.get_default_for(nargs["v_desc"].block_shape, gl.float16)
    nargs["o_desc"].layout = gl.NVMMASharedLayout.get_default_for(nargs["o_desc"].block_shape, gl.float16)

# ---------------------------------------------------------------------------
# AUTOTUNING WRAPPERS
# ---------------------------------------------------------------------------
fa3_kernel_autotune = triton.autotune(
    configs=fa3_get_configs(pre_hook=fa3_tma_set_block_size_hook, tune=True),
    key=["SEQ_LEN", "HEAD_DIM"],
    do_bench=lambda kernel_call, quantiles: triton.testing.do_bench_cudagraph(
        kernel_call, rep=100, quantiles=quantiles),
)(fa3_warp_specialized_kernel)


def run_fa3_kernel(Q, K, V, tune=True, manual_config=None):
    # Assuming 2D unbatched for simplicity: (SEQ_LEN, HEAD_DIM)
    BATCH, NUM_HEADS, SEQ_LEN, HEAD_DIM = Q.shape
    
    O = torch.empty_like(Q)
    
    Q_flat = Q.reshape(-1, HEAD_DIM)
    K_flat = K.reshape(-1, HEAD_DIM)
    V_flat = V.reshape(-1, HEAD_DIM)
    O_flat = O.reshape(-1, HEAD_DIM)
    
    # Create Dummy Descriptors (Actual block shapes are injected by the pre_hook)
    dummy_block = [1, 1]
    dummy_layout_f16 = gl.NVMMASharedLayout.get_default_for(dummy_block, gl.float16)
    
    K_T = K_flat.transpose(0,1).contiguous()
    
    q_desc = TensorDescriptor.from_tensor(Q_flat, dummy_block, dummy_layout_f16)
    k_desc = TensorDescriptor.from_tensor(K_T, dummy_block, dummy_layout_f16)
    v_desc = TensorDescriptor.from_tensor(V_flat, dummy_block, dummy_layout_f16)
    o_desc = TensorDescriptor.from_tensor(O_flat, dummy_block, dummy_layout_f16)

    if tune:
        def grid(meta):
            num_sms = torch.cuda.get_device_properties("cuda").multi_processor_count
            # FA3 partitions the grid only across the Q Sequence Length (M dimension)
            num_pid = triton.cdiv(SEQ_LEN, meta["BLOCK_SIZE_M"])
            total_tiles = num_pid * BATCH * NUM_HEADS
            return (min(num_sms, total_tiles), )
            
        fa3_kernel_autotune[grid](
            q_desc, k_desc, v_desc, o_desc, 
            GroupedPersistentTileScheduler(8), 
            SEQ_LEN, HEAD_DIM, NUM_HEADS
        )
        
    else:
        # 1. Setup the TMA Block Shapes using the manual config
        hook_kwargs = {
            "BLOCK_SIZE_M": manual_config["BM"],
            "BLOCK_SIZE_N": manual_config["BN"],
            "BLOCK_SIZE_K": manual_config["BK"],
            "SUBTILE_FACTOR": manual_config["SF"],
            "q_desc": q_desc, "k_desc": k_desc, "v_desc": v_desc, "o_desc": o_desc
        }
        fa3_tma_set_block_size_hook(hook_kwargs)
        
        # 2. Grid Calculation
        num_sms = torch.cuda.get_device_properties("cuda").multi_processor_count
        num_pid = triton.cdiv(SEQ_LEN, manual_config["BM"])
        total_tiles = num_pid * BATCH * NUM_HEADS
        grid = (min(num_sms, total_tiles), )
        # grid = (1, )
        
        # 3. Launch the Kernel
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
    
    # Manual config arguments 
    parser.add_argument("--bm", type=int, default=128, help="BLOCK_SIZE_M")
    parser.add_argument("--bn", type=int, default=128, help="BLOCK_SIZE_N")
    parser.add_argument("--bk", type=int, default=128, help="HEAD_DIM (BLOCK_SIZE_K)")
    parser.add_argument("--warps", type=int, default=8, help="Number of compute warps")
    parser.add_argument("--stages", type=int, default=2, help="Number of pipeline stages for KV")
    parser.add_argument("--sf", type=int, default=4, help="SUBTILE_FACTOR")
    
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
        
    # Standard FA sizes: (SEQ_LEN, HEAD_DIM)
    BATCH, NUM_HEADS = 2, 16
    sizes = [
        (256, 128),
        (8192, 128)
    ]
    
    torch.set_printoptions(profile="full")
    torch.set_printoptions(linewidth=20000)

    for SEQ_LEN, HEAD_DIM in sizes:
        print(f"Testing BATCH={BATCH}, NUM_HEADS={NUM_HEADS}, SEQ_LEN={SEQ_LEN}, HEAD_DIM={HEAD_DIM}")
        
        # 4D Inputs
        Q = torch.randn((BATCH, NUM_HEADS, SEQ_LEN, HEAD_DIM), device="cuda", dtype=torch.float16)
        K = torch.randn((BATCH, NUM_HEADS, SEQ_LEN, HEAD_DIM), device="cuda", dtype=torch.float16)
        V = torch.randn((BATCH, NUM_HEADS, SEQ_LEN, HEAD_DIM), device="cuda", dtype=torch.float16)
        
        O_triton = run_fa3_kernel(Q, K, V, tune=args.tune, manual_config=manual_config)
        O_torch = torch.nn.functional.scaled_dot_product_attention(Q, K, V)
        
        torch.testing.assert_close(O_torch, O_triton, rtol=1e-2, atol=1e-2)
    
    print("Done. PyTorch reference matches Triton Gluon FA3!")