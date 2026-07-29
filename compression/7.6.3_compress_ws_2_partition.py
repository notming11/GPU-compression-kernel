import argparse
import os
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
    warpgroup_mma,
    warpgroup_mma_wait,
    warpgroup_mma_accumulator,
)

from common import WGMMA, GroupedPersistentTileScheduler
from prune import prune_2_4
from compress_2_4 import compress_dense_to_sparse

from typing import Union

# ---------------------------------------------------------------------------
# WGMMA HELPERS
# ---------------------------------------------------------------------------

@gluon.constexpr_function
def get_warps_per_cta(BLOCK_M, BLOCK_N, num_warps):
    warps_per_cta = [4, 1]
    m = 16
    while warps_per_cta[0] * warps_per_cta[1] != num_warps:
        if BLOCK_M > m * warps_per_cta[0]:
            warps_per_cta[0] *= 2
        else:
            warps_per_cta[1] *= 2
    return warps_per_cta

@gluon.constexpr_function
def get_instr_shape_n(BLOCK_M, BLOCK_N, num_warps):
    m = 16
    mReps = triton.cdiv(BLOCK_M, m)
    nReps = triton.cdiv(num_warps, mReps)
    maxN = max(BLOCK_N // nReps, 8)
    n = 256
    while n > maxN or BLOCK_N % n != 0:
        n -= 8
    assert n >= 8, "expected to find a valid n"
    return n

@gluon.constexpr_function
def pick_sparse_wgmma_layout(dtype, BLOCK_M, BLOCK_N, num_warps):
    m = 16
    k = 32
    n = get_instr_shape_n(BLOCK_M, BLOCK_N, num_warps)
    warps_per_cta = get_warps_per_cta(BLOCK_M, BLOCK_N, num_warps)
    return gl.NVMMADistributedLayout(
        version=[3, 0],
        warps_per_cta=warps_per_cta,
        instr_shape=[m, n, k],
    )
    
@gluon.jit
def create_metadata(meta_1, meta_2):
    return meta_1 | (meta_2 << 4)


@gluon.jit
def create_metadata_8(meta_1, meta_2):
    return meta_1 | (meta_2 << 8)

@aggregate
class SparseWGMMA:
    acc: Union[warpgroup_mma_accumulator, gl.tensor]
    use_acc: gl.tensor
    layout: gl.constexpr

    @gluon.constexpr_function
    def __init__(self, acc, use_acc, layout):
        self.acc = acc
        self.use_acc = use_acc
        self.layout = gl.constexpr(layout)

    @gluon.jit
    def initialize(dtype: gl.constexpr, BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr, num_warps: gl.constexpr):
        mma_layout: gl.constexpr = pick_sparse_wgmma_layout(dtype, BLOCK_M, BLOCK_N, num_warps)
        acc = gl.zeros((BLOCK_M, BLOCK_N), dtype=gl.float32, layout=mma_layout)
        return SparseWGMMA(acc, gl.to_tensor(False), mma_layout)
    
    @gluon.jit
    def generate_compressed_and_meta(self, a_pruned, BLOCK_M : gl.constexpr, BLOCK_K: gl.constexpr, a_compressed_layout: gl.constexpr):
        # --- Extract groups of 4 consecutive columns using reshape + split ---
        a_grouped = a_pruned.reshape(BLOCK_M, BLOCK_K // 4, 2, 2)
        a_even, a_odd = a_grouped.split()

        # split again to separate the pairs
        a0, a2 = a_even.split()  # a0 = col 4g+0, a2 = col 4g+2
        a1, a3 = a_odd.split()   # a1 = col 4g+1, a3 = col 4g+3

        # OPTIMIZATION 1: Cache the non-zero checks.
        b0 = a0 != 0
        b1 = a1 != 0
        b2 = a2 != 0
        # OPTIMIZATION 2: Streamlined value extraction.
        nz0 = gl.where(b0, a0, gl.where(b1, a1, a2))

        nz1 = gl.where(b0 & b1, a1, gl.where(b2 & (b0 | b1), a2, a3))

        a_compressed = gl.join(nz0, nz1)
        a_compressed = a_compressed.reshape(BLOCK_M, BLOCK_K // 2)

        # OPTIMIZATION 3: Direct metadata generation.
        meta_4 = gl.where(b0,
             gl.where(b1, 4, gl.where(b2, 8, 12)),
             gl.where(b1, gl.where(b2, 9, 13), 14))

        # To lower register usage, we do the reshape and permute BEFORE the reduction!
        meta_4_reshaped = meta_4.reshape(BLOCK_M // 16, 2, 8, BLOCK_K // 64, 4, 2, 2)
        meta_4_permuted = meta_4_reshaped.permute(0, 3, 2, 4, 1, 5, 6)
        meta_4_ready = meta_4_permuted.reshape(BLOCK_M // 16, BLOCK_K, 2, 2)
        
        meta_reordered = gl.reduce(
            gl.reduce(meta_4_ready, 3, create_metadata), 2, create_metadata_8
        ).to(gl.int16)

        e_layout: gl.constexpr = gl.DotOperandLayout(
            operand_index=0,
            parent=self.layout,
            k_width=32 // gl.int16.primitive_bitwidth,
            meta=1,
        )

        a_compressed = gl.convert_layout(
            a_compressed, a_compressed_layout, assert_trivial=False
        )
        e = gl.convert_layout(meta_reordered, e_layout)
        
        return a_compressed, e

    @gluon.jit
    def generate_compressed_and_meta_raw(self, a_pruned, BLOCK_M : gl.constexpr, BLOCK_K: gl.constexpr):
        """Like generate_compressed_and_meta but returns raw tensors without layout conversion.
        Used by the load+compress partition which writes results to SMEM."""
        # --- Extract groups of 4 consecutive columns using reshape + split ---
        a_grouped = a_pruned.reshape(BLOCK_M, BLOCK_K // 4, 2, 2)
        a_even, a_odd = a_grouped.split()

        a0, a2 = a_even.split()
        a1, a3 = a_odd.split()

        b0 = a0 != 0
        b1 = a1 != 0
        b2 = a2 != 0
        nz0 = gl.where(b0, a0, gl.where(b1, a1, a2))
        nz1 = gl.where(b0 & b1, a1, gl.where(b2 & (b0 | b1), a2, a3))

        a_compressed = gl.join(nz0, nz1)
        a_compressed = a_compressed.reshape(BLOCK_M, BLOCK_K // 2)

        meta_4 = gl.where(b0,
             gl.where(b1, 4, gl.where(b2, 8, 12)),
             gl.where(b1, gl.where(b2, 9, 13), 14))

        meta_4_reshaped = meta_4.reshape(BLOCK_M // 16, 2, 8, BLOCK_K // 64, 4, 2, 2)
        meta_4_permuted = meta_4_reshaped.permute(0, 3, 2, 4, 1, 5, 6)
        meta_4_ready = meta_4_permuted.reshape(BLOCK_M // 16, BLOCK_K, 2, 2)
        
        meta_reordered = gl.reduce(
            gl.reduce(meta_4_ready, 3, create_metadata), 2, create_metadata_8
        ).to(gl.int16)
        
        return a_compressed, meta_reordered

    @gluon.jit
    def issue_precompressed_async_mma(
        self,
        a_compressed,
        e,
        b
    ):
        acc = warpgroup_mma(
            a_compressed,
            b,
            self.acc,
            e=e,
            is_async=True,
            use_acc=self.use_acc,
        )
        return SparseWGMMA(acc, gl.to_tensor(True), self.layout)

    @gluon.jit
    def wait_num_outstanding(self, num_outstanding: gl.constexpr):
        acc = warpgroup_mma_wait(num_outstanding, (self.acc,))
        return SparseWGMMA(acc, self.use_acc, self.layout)

    @gluon.jit
    def flush_num_outstanding(self):
        acc = warpgroup_mma_wait(0, (self.acc,))
        return SparseWGMMA(acc, self.use_acc, self.layout)

    # Take the result and reset the accumulator.
    @gluon.jit
    def take_result(self):
        return self.acc, SparseWGMMA(self.acc, gl.to_tensor(False), self.layout)


# ---------------------------------------------------------------------------
# SHARED ARGS & COUNTER
# ---------------------------------------------------------------------------

@aggregate
class SparsePartitionArgs:
    a_pruned_desc: tma.tensor_descriptor
    b_desc: tma.tensor_descriptor
    c_desc: tma.tensor_descriptor
    a_pruned_bufs: gl.shared_memory_descriptor
    b_bufs: gl.shared_memory_descriptor
    a_comp_bufs: gl.shared_memory_descriptor
    e_bufs: gl.shared_memory_descriptor
    
    a_empty_bars: gl.shared_memory_descriptor
    a_ready_bars: gl.shared_memory_descriptor
    b_empty_bars: gl.shared_memory_descriptor
    b_ready_bars: gl.shared_memory_descriptor
    comp_empty_bars: gl.shared_memory_descriptor
    comp_ready_bars: gl.shared_memory_descriptor
    
    acc_bufs: gl.shared_memory_descriptor # Still needed for TMA store drops
    
    SUBTILE_FACTOR: gl.constexpr
    num_warps: gl.constexpr
    num_buffers: gl.constexpr

    @gluon.constexpr_function
    def __init__(self, a_pruned_desc, b_desc, c_desc, 
                 a_pruned_bufs, b_bufs, a_comp_bufs, e_bufs,
                 a_empty_bars, a_ready_bars, b_empty_bars, b_ready_bars,
                 comp_empty_bars, comp_ready_bars,
                 acc_bufs, SUBTILE_FACTOR, num_warps, num_buffers):
        self.a_pruned_desc = a_pruned_desc
        self.b_desc = b_desc
        self.c_desc = c_desc
        self.a_pruned_bufs = a_pruned_bufs
        self.b_bufs = b_bufs
        self.a_comp_bufs = a_comp_bufs
        self.e_bufs = e_bufs
        
        self.a_empty_bars = a_empty_bars
        self.a_ready_bars = a_ready_bars
        self.b_empty_bars = b_empty_bars
        self.b_ready_bars = b_ready_bars
        self.comp_empty_bars = comp_empty_bars
        self.comp_ready_bars = comp_ready_bars
        
        self.acc_bufs = acc_bufs
        
        self.SUBTILE_FACTOR = gl.constexpr(SUBTILE_FACTOR)
        self.num_warps = gl.constexpr(num_warps)
        self.num_buffers = gl.constexpr(num_buffers)


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


# ---------------------------------------------------------------------------
# PIPELINE PARTITIONS (2-Stage Fused Topology)
# ---------------------------------------------------------------------------

@gluon.jit
def sparse_matmul_load_and_compress_partition(p, SchedulerImpl: gl.constexpr):
    BLOCK_M: gl.constexpr = p.a_pruned_desc.block_type.shape[0]
    BLOCK_N: gl.constexpr = p.b_desc.block_type.shape[1]
    BLOCK_K: gl.constexpr = p.a_pruned_desc.block_type.shape[1]
    K = p.a_pruned_desc.shape[1]
    num_buffers: gl.constexpr = p.num_buffers

    issue_state = Counter.create(1, num_buffers)
    process_state = Counter.create(0, num_buffers)
    
    # Producer warp footprint lock (4 warps = 128 threads)
    compress_warps: gl.constexpr = 4
    a_warp_bases: gl.constexpr = [[16, 0], [32, 0]]
    
    a_pruned_reg_layout: gl.constexpr = gl.DistributedLinearLayout(
        reg_bases=[[0, 1], [0, 2], [8, 0], [0, 4], [0, 8]],
        lane_bases=[[0, 16], [0, 32], [1, 0], [2, 0], [4, 0]],
        warp_bases=a_warp_bases,
        block_bases=[],
        shape=[16 * compress_warps, 64],
    )

    # Create a dummy SparseWGMMA to access the compression method
    compress_helper = SparseWGMMA.initialize(p.a_pruned_desc.dtype, BLOCK_M, BLOCK_N, compress_warps)

    scheduler = SchedulerImpl.initialize(p.c_desc.shape[0], p.c_desc.shape[1], BLOCK_M, BLOCK_N)
    total_k_iters = (K + BLOCK_K - 1) // BLOCK_K

    for idx in range(scheduler.get_num_tiles()):
        pid_m, pid_n = scheduler.get_tile(idx)
        off_m, off_n = pid_m * BLOCK_M, pid_n * BLOCK_N
        
        # --- PROLOGUE: Pre-fetch ---
        for k_issue in range(0, (num_buffers - 1) * BLOCK_K, BLOCK_K):
            if k_issue < K:
                a_bar = p.a_ready_bars.index(issue_state.index)
                b_bar = p.b_ready_bars.index(issue_state.index)
                
                mbarrier.wait(p.a_empty_bars.index(issue_state.index), issue_state.phase)
                mbarrier.wait(p.b_empty_bars.index(issue_state.index), issue_state.phase)
                
                mbarrier.expect(a_bar, p.a_pruned_desc.block_type.nbytes)
                tma.async_copy_global_to_shared(p.a_pruned_desc, [off_m, k_issue], a_bar, p.a_pruned_bufs.index(issue_state.index))
                
                mbarrier.expect(b_bar, p.b_desc.block_type.nbytes)
                tma.async_copy_global_to_shared(p.b_desc, [k_issue, off_n], b_bar, p.b_bufs.index(issue_state.index))
                issue_state = issue_state.next()

        # --- STEADY STATE ---
        for k_process in range(0, K, BLOCK_K):
            # 2. Wait for Load and Compute Handoff
            mbarrier.wait(p.a_ready_bars.index(process_state.index), process_state.phase)
            mbarrier.wait(p.comp_empty_bars.index(process_state.index), process_state.phase ^ 1)

            # 3. Load Tile to Registers
            a_pruned = p.a_pruned_bufs.index(process_state.index).load(a_pruned_reg_layout)

            # 4. Compress and generate metadata via SparseWGMMA class method
            a_compressed, meta_reordered = compress_helper.generate_compressed_and_meta_raw(
                a_pruned, BLOCK_M, BLOCK_K
            )

            # 5. Write back to SMEM
            p.a_comp_bufs.index(process_state.index).store(a_compressed)
            p.e_bufs.index(process_state.index).store(meta_reordered)
            fence_async_shared()
            
            # 6. Hand-off Signals
            mbarrier.arrive(p.comp_ready_bars.index(process_state.index), count=1)
            mbarrier.arrive(p.a_empty_bars.index(process_state.index), count=1)

            process_state = process_state.next()
            
            # 1. Lookahead Issue
            k_future = k_process + (num_buffers - 1) * BLOCK_K
            if k_future < K:
                a_bar = p.a_ready_bars.index(issue_state.index)
                b_bar = p.b_ready_bars.index(issue_state.index)
                
                mbarrier.wait(p.a_empty_bars.index(issue_state.index), issue_state.phase)
                mbarrier.wait(p.b_empty_bars.index(issue_state.index), issue_state.phase)
                
                mbarrier.expect(a_bar, p.a_pruned_desc.block_type.nbytes)
                tma.async_copy_global_to_shared(p.a_pruned_desc, [off_m, k_future], a_bar, p.a_pruned_bufs.index(issue_state.index))
                
                mbarrier.expect(b_bar, p.b_desc.block_type.nbytes)
                tma.async_copy_global_to_shared(p.b_desc, [k_future, off_n], b_bar, p.b_bufs.index(issue_state.index))
                issue_state = issue_state.next()
            


@gluon.jit
def sparse_matmul_compute_iteration(
    p, state, mma, k_iter, outstanding_mmas: gl.constexpr,
    e_layout: gl.constexpr, num_buffers: gl.constexpr
):
    mbarrier.wait(p.comp_ready_bars.index(state.index), state.phase)
    mbarrier.wait(p.b_ready_bars.index(state.index), state.phase)

    e = p.e_bufs.index(state.index).load(e_layout)
    
    mma = mma.issue_precompressed_async_mma(
        p.a_comp_bufs.index(state.index),
        e,
        p.b_bufs.index(state.index)
    )
    state = state.next()
    
    mma = mma.wait_num_outstanding(outstanding_mmas)

    idx_to_arrive = (k_iter - outstanding_mmas) % num_buffers
    mbarrier.arrive(
        p.comp_empty_bars.index(idx_to_arrive),
        count=1, pred=k_iter >= outstanding_mmas
    )
    mbarrier.arrive(
        p.b_empty_bars.index(idx_to_arrive),
        count=1, pred=k_iter >= outstanding_mmas
    )

    return state, mma, k_iter + 1

@gluon.jit
def sparse_matmul_compute_drain(
    p, state, mma, k_iter, limit, current_mma: gl.constexpr, num_mmas: gl.constexpr,
    e_layout: gl.constexpr, num_buffers: gl.constexpr
):
    if current_mma >= num_mmas: 
        return state, mma, k_iter
    else:
        if k_iter >= limit: 
            return state, mma, k_iter
        else:
            state, mma, k_iter = sparse_matmul_compute_iteration(
                p, state, mma, k_iter, num_mmas, e_layout, num_buffers
            )
            return sparse_matmul_compute_drain(
                p, state, mma, k_iter, limit, gl.constexpr(current_mma + 1), num_mmas, e_layout, num_buffers
            )

@gluon.jit
def sparse_matmul_compute_and_store_partition(p, SchedulerImpl: gl.constexpr):
    BLOCK_M: gl.constexpr = p.a_pruned_desc.block_type.shape[0]
    BLOCK_N: gl.constexpr = p.b_desc.block_type.shape[1]
    BLOCK_K: gl.constexpr = p.a_pruned_desc.block_type.shape[1]
    SPLIT_N: gl.constexpr = p.c_desc.block_type.shape[1]
    K = p.a_pruned_desc.shape[1]
    dtype: gl.constexpr = p.a_pruned_desc.dtype
    num_buffers: gl.constexpr = p.num_buffers

    state = Counter.create(0, num_buffers)
    scheduler = SchedulerImpl.initialize(p.c_desc.shape[0], p.c_desc.shape[1], BLOCK_M, BLOCK_N)
    
    num_mmas: gl.constexpr = 2
    k_iter = 0

    for idx in range(scheduler.get_num_tiles()):
        mma = SparseWGMMA.initialize(dtype, BLOCK_M, BLOCK_N, p.num_warps)
        
        e_layout: gl.constexpr = gl.DotOperandLayout(
            operand_index=0, parent=mma.layout,
            k_width=32 // dtype.primitive_bitwidth, meta=1,
        )

        total_k_iters = (K + BLOCK_K - 1) // BLOCK_K

        for _ in range(total_k_iters // num_mmas):
            for _ in gl.static_range(num_mmas):
                state, mma, k_iter = sparse_matmul_compute_iteration(
                    p, state, mma, k_iter, num_mmas - 1, e_layout, num_buffers
                )

        state, mma, k_iter = sparse_matmul_compute_drain(
            p, state, mma, k_iter,
            (total_k_iters % num_mmas) + k_iter,
            0, num_mmas - 1, e_layout, num_buffers
        )

        # -----------------------------------------------------------
        # FUSED STORE EPILOGUE (Direct TMA issue)
        # -----------------------------------------------------------
        pid_m, pid_n = scheduler.get_tile(idx)
        off_m, off_n = pid_m * BLOCK_M, pid_n * BLOCK_N
        
        mma = mma.wait_num_outstanding(0)
        acc, mma = mma.take_result()
        accs = _split_n(acc, p.SUBTILE_FACTOR)

        for i in gl.static_range(p.SUBTILE_FACTOR):
            # Ping-pong between the 2 SMEM buffers to prevent pipeline stalls
            buf_idx = i % 2
            c_buf = p.acc_bufs.index(buf_idx)
            
            if i >= 2:
                # Wait until the previous TMA store from this specific buffer finishes
                tma.store_wait(1) 
                
            c_buf.store(accs[i].to(p.c_desc.dtype))
            fence_async_shared()
            
            tma.async_copy_shared_to_global(p.c_desc, [off_m, off_n + i * SPLIT_N], c_buf)

        # Ensure all chunk stores finish before moving to the next primary tile
        tma.store_wait(0)


# ---------------------------------------------------------------------------
# KERNEL LAUNCHER
# ---------------------------------------------------------------------------

@gluon.jit
def sparse_matmul_warp_specialized_kernel(
    a_pruned_desc, a_comp_desc, e_desc, b_desc, c_desc, SchedulerImpl: gl.constexpr,
    M, N, K, BLOCK_SIZE_M: gl.constexpr, BLOCK_SIZE_N: gl.constexpr, BLOCK_SIZE_K: gl.constexpr,
    num_buffers: gl.constexpr, SUBTILE_FACTOR: gl.constexpr, num_warps: gl.constexpr
):
    dtype: gl.constexpr = a_pruned_desc.dtype

    a_pruned_bufs = gl.allocate_shared_memory(dtype, [num_buffers] + a_pruned_desc.block_type.shape, a_pruned_desc.layout)
    b_bufs = gl.allocate_shared_memory(dtype, [num_buffers] + b_desc.block_type.shape, b_desc.layout)
    
    a_comp_bufs = gl.allocate_shared_memory(dtype, [num_buffers] + a_comp_desc.block_type.shape, a_comp_desc.layout)
    e_bufs = gl.allocate_shared_memory(gl.int16, [num_buffers] + e_desc.block_type.shape, e_desc.layout)

    a_empty_bars = gl.allocate_shared_memory(gl.int64, [num_buffers, 1], mbarrier.MBarrierLayout())
    a_ready_bars = gl.allocate_shared_memory(gl.int64, [num_buffers, 1], mbarrier.MBarrierLayout())
    b_empty_bars = gl.allocate_shared_memory(gl.int64, [num_buffers, 1], mbarrier.MBarrierLayout())
    b_ready_bars = gl.allocate_shared_memory(gl.int64, [num_buffers, 1], mbarrier.MBarrierLayout())
    comp_empty_bars = gl.allocate_shared_memory(gl.int64, [num_buffers, 1], mbarrier.MBarrierLayout())
    comp_ready_bars = gl.allocate_shared_memory(gl.int64, [num_buffers, 1], mbarrier.MBarrierLayout())

    for i in gl.static_range(num_buffers):
        mbarrier.init(a_empty_bars.index(i), count=1)
        mbarrier.init(a_ready_bars.index(i), count=1)
        mbarrier.init(b_empty_bars.index(i), count=1)
        mbarrier.init(b_ready_bars.index(i), count=1)
        mbarrier.init(comp_empty_bars.index(i), count=1)
        mbarrier.init(comp_ready_bars.index(i), count=1)

    # 2 buffers for ping-pong Store drops
    acc_bufs = gl.allocate_shared_memory(dtype, [2] + c_desc.block_type.shape, c_desc.layout)

    p = SparsePartitionArgs(
        a_pruned_desc, b_desc, c_desc, 
        a_pruned_bufs, b_bufs, a_comp_bufs, e_bufs,
        a_empty_bars, a_ready_bars, b_empty_bars, b_ready_bars,
        comp_empty_bars, comp_ready_bars,
        acc_bufs, SUBTILE_FACTOR, num_warps, num_buffers
    )

    # 2-Stage Topology: Default (Compute+Store) | Worker 1 (Load+Compress)
    # Default partition inherits `num_warps` (e.g. 8). Worker 1 gets 4 warps.
    gl.warp_specialize([
        (sparse_matmul_compute_and_store_partition, (p, SchedulerImpl)),
        (sparse_matmul_load_and_compress_partition, (p, SchedulerImpl)),
    ], [4], [255])


def sparse_matmul_get_configs(pre_hook=None, tune=True):
    def valid(BM, BN, BK, warps, buffers, SF):
        # 1. Total warps check (Compute + 4 Producer) <= 32 Max
        if warps + 4 > 32: return False
        
        # 2. SMEM check (Limit 227KB)
        smem_bytes = 2 * ((buffers * BM * BK) + (buffers * BK * BN))
        smem_bytes += 2 * (buffers * BM * (BK // 2))
        smem_bytes += 2 * (buffers * (BM // 16) * BK)
        smem_bytes += 2 * (2 * BM * (BN // SF)) # Acc buffers
        smem_bytes += 8 * (6 * buffers) # Acc barriers eliminated
        if smem_bytes > 232448: return False

        warps_m = 4
        warps_n = 1
        m = 16
        while (warps_m * warps_n) != warps:
            if BM > m * warps_m: warps_m *= 2
            else: warps_n *= 2

        if SF > 1 and warps_n > 1: return False
        if (BN // SF) < 16: return False
        if BM < warps_m * 16 or BN < warps_n * 16: return False

        # 3. Dynamic Pipeline Register Math
        elements_per_thread = (BM * BN) / (warps * 32)
        required_regs = elements_per_thread + 48
        
        worker_regs = (4 * 32 * 128) # 4 Producer warps @ 128 regs
        max_regs_per_thread = (65536 - worker_regs) // (warps * 32)
        
        if required_regs > max_regs_per_thread: return False
        if elements_per_thread < 16: return False
        return True

    configs = [
        triton.Config(
            {
                "BLOCK_SIZE_M": BM, "BLOCK_SIZE_N": BN, "BLOCK_SIZE_K": BK,
                "num_buffers": buffers, "SUBTILE_FACTOR": SF,
            },
            num_warps=warps,
            pre_hook=pre_hook,
            maxnreg=255,
        )
        for BM in (64, 128, 256)
        for BN in (64, 128, 256)
        for BK in (64, 128, 256)
        for warps in (4, 8, 16)
        for buffers in (3, 4, 5, 6, 7)
        for SF in (1, 2, 4, 8)
        if valid(BM, BN, BK, warps, buffers, SF)
    ]
    return configs if tune else configs[:1]

def sparse_matmul_tma_set_block_size_hook(nargs):
    block_m = nargs["BLOCK_SIZE_M"]
    block_n = nargs["BLOCK_SIZE_N"]
    block_k = nargs["BLOCK_SIZE_K"]
    split_n = nargs["BLOCK_SIZE_N"] // nargs["SUBTILE_FACTOR"]

    nargs["a_pruned_desc"].block_shape = [block_m, block_k]
    nargs["a_comp_desc"].block_shape = [block_m, block_k // 2]
    nargs["e_desc"].block_shape = [block_m // 16, block_k]
    nargs["b_desc"].block_shape = [block_k, block_n]
    nargs["c_desc"].block_shape = [block_m, split_n]

    nargs["a_pruned_desc"].layout = gl.NVMMASharedLayout.get_default_for(nargs["a_pruned_desc"].block_shape, gl.float16)
    nargs["a_comp_desc"].layout = gl.NVMMASharedLayout.get_default_for(nargs["a_comp_desc"].block_shape, gl.float16)
    nargs["e_desc"].layout = gl.NVMMASharedLayout.get_default_for(nargs["e_desc"].block_shape, gl.int16)
    nargs["b_desc"].layout = gl.NVMMASharedLayout.get_default_for(nargs["b_desc"].block_shape, gl.float16)
    nargs["c_desc"].layout = gl.NVMMASharedLayout.get_default_for(nargs["c_desc"].block_shape, gl.float16)

sparse_ws_kernel_autotune = triton.autotune(
    configs=sparse_matmul_get_configs(pre_hook=sparse_matmul_tma_set_block_size_hook, tune=True),
    key=["M", "N", "K"],
    do_bench = lambda kernel_call, quantiles: triton.testing.do_bench_cudagraph(kernel_call, rep=100, quantiles=quantiles),
)(sparse_matmul_warp_specialized_kernel)

def run_sparse_ws_matmul(A_pruned, B, tune=True, manual_config=None):
    M, K = A_pruned.shape[0], A_pruned.shape[1]
    N = B.shape[1]

    c = torch.empty((M, N), device=A_pruned.device, dtype=torch.float16)
    dummy_block = [1, 1]
    dummy_layout_f16 = gl.NVMMASharedLayout.get_default_for(dummy_block, gl.float16)
    dummy_layout_i16 = gl.NVMMASharedLayout.get_default_for(dummy_block, gl.int16)
    
    a_pruned_desc = TensorDescriptor.from_tensor(A_pruned, dummy_block, dummy_layout_f16)
    a_comp_desc = TensorDescriptor.from_tensor(A_pruned, dummy_block, dummy_layout_f16)
    e_desc = TensorDescriptor.from_tensor(A_pruned.to(torch.int16), dummy_block, dummy_layout_i16)
    b_desc = TensorDescriptor.from_tensor(B, dummy_block, dummy_layout_f16)
    c_desc = TensorDescriptor.from_tensor(c, dummy_block, dummy_layout_f16)

    if tune:
        def grid(meta):
            num_sms = torch.cuda.get_device_properties("cuda").multi_processor_count
            num_pid = triton.cdiv(M, meta["BLOCK_SIZE_M"]) * triton.cdiv(N, meta["BLOCK_SIZE_N"])
            return (min(num_sms, num_pid), )
            
        sparse_ws_kernel_autotune[grid](a_pruned_desc, a_comp_desc, e_desc, b_desc, c_desc, GroupedPersistentTileScheduler(8), M, N, K)
    else:
        hook_kwargs = {
            "BLOCK_SIZE_M": manual_config["BM"], "BLOCK_SIZE_N": manual_config["BN"],
            "BLOCK_SIZE_K": manual_config["BK"], "SUBTILE_FACTOR": manual_config["SF"],
            "a_pruned_desc": a_pruned_desc, "a_comp_desc": a_comp_desc, "e_desc": e_desc,
            "b_desc": b_desc, "c_desc": c_desc
        }
        
        sparse_matmul_tma_set_block_size_hook(hook_kwargs)
        
        num_sms = torch.cuda.get_device_properties("cuda").multi_processor_count
        num_pid = triton.cdiv(M, manual_config["BM"]) * triton.cdiv(N, manual_config["BN"])
        grid = (min(num_sms, num_pid), )
        
        sparse_matmul_warp_specialized_kernel[grid](
            a_pruned_desc, a_comp_desc, e_desc, b_desc, c_desc, GroupedPersistentTileScheduler(8),
            M, N, K,
            BLOCK_SIZE_M=manual_config["BM"], BLOCK_SIZE_N=manual_config["BN"], BLOCK_SIZE_K=manual_config["BK"],
            num_buffers=manual_config["buffers"], SUBTILE_FACTOR=manual_config["SF"], num_warps=manual_config["warps"]
        )

    return c

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Fused-Compression Sparse Warp-Specialized Matmul")
    parser.add_argument("--tune", action="store_true", help="Enable Triton autotuning")
    
    parser.add_argument("--bm", type=int, default=128, help="BLOCK_SIZE_M")
    parser.add_argument("--bn", type=int, default=256, help="BLOCK_SIZE_N")
    parser.add_argument("--bk", type=int, default=64, help="BLOCK_SIZE_K")
    parser.add_argument("--warps", type=int, default=8, help="Number of warps")
    parser.add_argument("--buffers", type=int, default=3, help="Number of buffers")
    parser.add_argument("--sf", type=int, default=4, help="SUBTILE_FACTOR")
    
    args = parser.parse_args()

    manual_config = {
        "BM": args.bm, "BN": args.bn, "BK": args.bk,
        "warps": args.warps, "buffers": args.buffers, "SF": args.sf
    }

    os.environ["MLIR_ENABLE_DUMP"]="1"
    os.environ["MLIR_DUMP_PATH"] = "./MLIR_DUMP/7.6.3"
    os.environ["TRITON_ALWAYS_COMPILE"]="1"
    os.environ["TRITON_CACHE_DIR"]="./compiler_scratch/.triton_cache"

    for M, N, K in [(49152, 8192, 49152)]:
        if args.tune:
            print(f"Testing 2-Stage Topology WS (AUTOTUNE ON): M={M}, N={N}, K={K}...", end="\n", flush=True)
        else:
            print(f"Testing 2-Stage Topology WS config {manual_config}: M={M}, N={N}, K={K}...", end="\n", flush=True)

        A = torch.randn(M, K, device="cuda", dtype=torch.float16)
        B = torch.randn((K, N), device="cuda", dtype=torch.float16)

        A_pruned = prune_2_4(A)

        C = run_sparse_ws_matmul(A_pruned, B, tune=args.tune, manual_config=manual_config)
        C_ref = A_pruned @ B

        torch.testing.assert_close(C_ref, C, rtol=1e-3, atol=1e-1)
        print("PASSED")