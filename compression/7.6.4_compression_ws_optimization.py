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
)

from common import (
    WGMMA,
    GroupedPersistentTileScheduler
)

from prune import prune_2_4
from compress_2_4 import compress_dense_to_sparse

# ---------------------------------------------------------------------------
# COMPRESSION LOGIC (LUT-based metadata lookup optimization)
# ---------------------------------------------------------------------------

from typing import Union
from triton.experimental.gluon.language.nvidia.hopper import (
    warpgroup_mma,
    warpgroup_mma_wait,
    warpgroup_mma_accumulator,
)


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
    def initialize(
        dtype: gl.constexpr,
        BLOCK_M: gl.constexpr,
        BLOCK_N: gl.constexpr,
        num_warps: gl.constexpr,
    ):
        mma_layout: gl.constexpr = pick_sparse_wgmma_layout(
            dtype, BLOCK_M, BLOCK_N, num_warps
        )
        acc = gl.zeros((BLOCK_M, BLOCK_N), dtype=gl.float32, layout=mma_layout)
        return SparseWGMMA(acc, gl.to_tensor(False), mma_layout)
    
    @gluon.jit
    def generate_compressed_and_meta(self, a_pruned, BLOCK_M : gl.constexpr, BLOCK_K: gl.constexpr, a_compressed_layout: gl.constexpr):
        # 1. Reshape and extract 4 consecutive columns
        a_grouped = a_pruned.reshape(BLOCK_M, BLOCK_K // 4, 2, 2)
        a_even, a_odd = a_grouped.split()

        a0, a2 = a_even.split()  # col 4g+0, col 4g+2
        a1, a3 = a_odd.split()   # col 4g+1, col 4g+3

        # Non-zero flags
        b0_bool = a0 != 0
        b1_bool = a1 != 0
        b2_bool = a2 != 0

        # 2. Extract non-zero values (nz0, nz1)
        nz0 = gl.where(b0_bool, a0, gl.where(b1_bool, a1, a2))
        nz1 = gl.where(b0_bool & b1_bool, a1, gl.where(b2_bool & (b0_bool | b1_bool), a2, a3))

        a_compressed = gl.join(nz0, nz1).reshape(BLOCK_M, BLOCK_K // 2)

        meta_4 = gl.where(b0_bool,
             gl.where(b1_bool, 4, gl.where(b2_bool, 8, 12)),
             gl.where(b1_bool, gl.where(b2_bool, 9, 13), 14))

        # 4. Pack metadata (reshape & permute for DotOperandLayout)
        meta_4_reshaped = meta_4.reshape(BLOCK_M // 16, 2, 8, BLOCK_K // 64, 4, 2, 2)
        meta_4_permuted = meta_4_reshaped.permute(0, 3, 2, 4, 1, 5, 6)
        meta_4_ready = meta_4_permuted.reshape(BLOCK_M // 16, BLOCK_K, 2, 2)

        # 4. Pack metadata using inline PTX assembly instead of gl.reduce
        meta_even, meta_odd = meta_4_ready.split()
        mn0, mn2 = meta_even.split()
        mn1, mn3 = meta_odd.split()

        # Pack 4 local nibbles into 1 16-bit word per thread via PTX assembly
        meta_reordered = gl.inline_asm_elementwise(
            asm="""
            {
            .reg .b32 t1, t2, t3;
            shl.b32 t1, $2, 4;
            shl.b32 t2, $3, 8;
            shl.b32 t3, $4, 12;
            or.b32 $0, $1, t1;
            or.b32 $0, $0, t2;
            or.b32 $0, $0, t3;
            }
            """,
            constraints="=r,r,r,r,r",
            args=[mn0, mn1, mn2, mn3],
            dtype=gl.int16,
            is_pure=True,
            pack=1,
        )

        e_layout: gl.constexpr = gl.DotOperandLayout(
            operand_index=0,
            parent=self.layout,
            k_width=32 // gl.int16.primitive_bitwidth,
            meta=1,
        )

        a_compressed = gl.convert_layout(
            a_compressed, a_compressed_layout
        )
        e = gl.convert_layout(meta_reordered, e_layout)
        
        return a_compressed, e

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
# SHARED HELPERS & ARGS
# ---------------------------------------------------------------------------

@aggregate
class SparsePartitionArgs:
    a_pruned_desc: tma.tensor_descriptor
    b_desc: tma.tensor_descriptor
    c_desc: tma.tensor_descriptor
    a_pruned_bufs: gl.shared_memory_descriptor
    b_bufs: gl.shared_memory_descriptor
    load_empty_bars: gl.shared_memory_descriptor
    load_ready_bars: gl.shared_memory_descriptor
    acc_bufs: gl.shared_memory_descriptor
    acc_empty_bars: gl.shared_memory_descriptor
    acc_ready_bars: gl.shared_memory_descriptor
    SUBTILE_FACTOR: gl.constexpr
    num_warps: gl.constexpr

    @gluon.constexpr_function
    def __init__(self, a_pruned_desc, b_desc, c_desc, a_pruned_bufs, b_bufs,
                 load_empty_bars, load_ready_bars,
                 acc_bufs, acc_empty_bars, acc_ready_bars,
                 SUBTILE_FACTOR, num_warps):
        self.a_pruned_desc = a_pruned_desc
        self.b_desc = b_desc
        self.c_desc = c_desc
        self.a_pruned_bufs = a_pruned_bufs
        self.b_bufs = b_bufs
        self.load_empty_bars = load_empty_bars
        self.load_ready_bars = load_ready_bars
        self.acc_bufs = acc_bufs
        self.acc_empty_bars = acc_empty_bars
        self.acc_ready_bars = acc_ready_bars
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
def store_acc_to_smem_subtile(p, mma, acc_state):
    mma = mma.wait_num_outstanding(0)
    acc, mma = mma.take_result()
    accs = _split_n(acc, p.SUBTILE_FACTOR)

    for i in gl.static_range(p.SUBTILE_FACTOR):
        mbarrier.wait(p.acc_empty_bars.index(acc_state.index), acc_state.phase)
        c_buf = p.acc_bufs.index(acc_state.index)

        c_buf.store(accs[i].to(p.c_desc.dtype))
        fence_async_shared()
        mbarrier.arrive(p.acc_ready_bars.index(acc_state.index), count=1)
        acc_state = acc_state.next()

    return acc_state

# ---------------------------------------------------------------------------
# SPARSE PARTITIONS
# ---------------------------------------------------------------------------

@gluon.jit
def sparse_matmul_load_partition(p, SchedulerImpl: gl.constexpr):
    BLOCK_M: gl.constexpr = p.a_pruned_desc.block_type.shape[0]
    BLOCK_N: gl.constexpr = p.b_desc.block_type.shape[1]
    BLOCK_K: gl.constexpr = p.a_pruned_desc.block_type.shape[1]
    K = p.a_pruned_desc.shape[1]

    state = Counter.create(1, p.load_empty_bars.shape[0])
    scheduler = SchedulerImpl.initialize(p.c_desc.shape[0], p.c_desc.shape[1], BLOCK_M, BLOCK_N)

    for idx in range(scheduler.get_num_tiles()):
        pid_m, pid_n = scheduler.get_tile(idx)
        off_m = pid_m * BLOCK_M
        off_n = pid_n * BLOCK_N

        for k in range(0, K, BLOCK_K):
            bar = p.load_ready_bars.index(state.index)
            mbarrier.wait(p.load_empty_bars.index(state.index), state.phase)

            mbarrier.expect(bar, p.a_pruned_desc.block_type.nbytes + p.b_desc.block_type.nbytes)
            tma.async_copy_global_to_shared(p.a_pruned_desc, [off_m, k], bar, p.a_pruned_bufs.index(state.index))
            tma.async_copy_global_to_shared(p.b_desc, [k, off_n], bar, p.b_bufs.index(state.index))
            state = state.next()

@gluon.jit
def sparse_matmul_compute_iteration(
    p, load_state, mma, k_iter, outstanding_mmas: gl.constexpr,
    a_pruned_reg_layout: gl.constexpr, a_compressed_layout: gl.constexpr,
    BLOCK_M: gl.constexpr, BLOCK_K: gl.constexpr
):
    mbarrier.wait(p.load_ready_bars.index(load_state.index), load_state.phase)

    # 1. Load pruned tile from SMEM
    a_pruned = p.a_pruned_bufs.index(load_state.index).load(a_pruned_reg_layout)
    
    # 2. Compress and pack metadata using LUT table
    a_comp, e = mma.generate_compressed_and_meta(
        a_pruned, BLOCK_M, BLOCK_K, a_compressed_layout
    )

    # 3. Issue the math using the locally scoped registers
    mma = mma.issue_precompressed_async_mma(
        a_comp, e, p.b_bufs.index(load_state.index)
    )

    load_state = load_state.next()

    mma = mma.wait_num_outstanding(outstanding_mmas)

    mbarrier.arrive(
        p.load_empty_bars.index((k_iter - outstanding_mmas) % p.load_empty_bars.shape[0]),
        count=1, pred=k_iter >= outstanding_mmas
    )

    return load_state, mma, k_iter + 1

@gluon.jit
def sparse_matmul_compute_drain(
    p, load_state, mma, k_iter, limit, current_mma: gl.constexpr, num_mmas: gl.constexpr,
    a_pruned_reg_layout: gl.constexpr, a_compressed_layout: gl.constexpr,
    BLOCK_M: gl.constexpr, BLOCK_K: gl.constexpr
):
    if current_mma >= num_mmas: 
        return load_state, mma, k_iter
    else:
        if k_iter >= limit: 
            return load_state, mma, k_iter
        else:
            load_state, mma, k_iter = sparse_matmul_compute_iteration(
                p, load_state, mma, k_iter, num_mmas,
                a_pruned_reg_layout, a_compressed_layout, BLOCK_M, BLOCK_K
            )
            return sparse_matmul_compute_drain(
                p, load_state, mma, k_iter, limit, gl.constexpr(current_mma + 1), num_mmas,
                a_pruned_reg_layout, a_compressed_layout, BLOCK_M, BLOCK_K
            )

@gluon.jit
def sparse_matmul_compute_partition(p, SchedulerImpl: gl.constexpr):
    BLOCK_M: gl.constexpr = p.a_pruned_desc.block_type.shape[0]
    BLOCK_N: gl.constexpr = p.b_desc.block_type.shape[1]
    BLOCK_K: gl.constexpr = p.a_pruned_desc.block_type.shape[1]
    K = p.a_pruned_desc.shape[1]
    dtype: gl.constexpr = p.a_pruned_desc.dtype

    load_state = Counter.create(0, p.load_empty_bars.shape[0])
    acc_state = Counter.create(1, p.acc_empty_bars.shape[0])

    scheduler = SchedulerImpl.initialize(p.c_desc.shape[0], p.c_desc.shape[1], BLOCK_M, BLOCK_N)

    if p.num_warps == 4:
        a_warp_bases: gl.constexpr = [[16, 0], [32, 0]]
    elif p.num_warps == 8:
        a_warp_bases: gl.constexpr = [[16, 0], [32, 0], [64, 0]]
    elif p.num_warps == 16:
        a_warp_bases: gl.constexpr = [[16, 0], [32, 0], [64, 0], [128, 0]]
    
    a_pruned_reg_layout: gl.constexpr = gl.DistributedLinearLayout(
        reg_bases=[[0, 1], [0, 2], [8, 0], [0, 4], [0, 8]],
        lane_bases=[[0, 16], [0, 32], [1, 0], [2, 0], [4, 0]],
        warp_bases=a_warp_bases,
        block_bases=[],
        shape=[16 * p.num_warps, 64],
    )

    num_mmas: gl.constexpr = 2
    k_iter = 0

    for _ in range(scheduler.get_num_tiles()):
        mma = SparseWGMMA.initialize(dtype, BLOCK_M, BLOCK_N, p.num_warps)

        a_compressed_layout: gl.constexpr = gl.DotOperandLayout(
            operand_index=0,
            parent=mma.layout,
            k_width=32 // dtype.primitive_bitwidth,
            meta=0,
        )

        total_k_iters = (K + BLOCK_K - 1) // BLOCK_K

        # Statically Unrolled Pipeline
        for _ in range(total_k_iters // num_mmas):
            for _ in gl.static_range(num_mmas):
                load_state, mma, k_iter = sparse_matmul_compute_iteration(
                    p, load_state, mma, k_iter, num_mmas - 1,
                    a_pruned_reg_layout, a_compressed_layout, BLOCK_M, BLOCK_K
                )

        # Drain Epilogue
        load_state, mma, k_iter = sparse_matmul_compute_drain(
            p, load_state, mma, k_iter,
            (total_k_iters % num_mmas) + k_iter,
            0, num_mmas - 1,
            a_pruned_reg_layout, a_compressed_layout, BLOCK_M, BLOCK_K
        )

        acc_state = store_acc_to_smem_subtile(p, mma, acc_state)

@gluon.jit
def sparse_matmul_store_partition(p, SchedulerImpl: gl.constexpr):
    BLOCK_M: gl.constexpr = p.c_desc.block_type.shape[0]
    SPLIT_N: gl.constexpr = p.c_desc.block_type.shape[1]
    BLOCK_N: gl.constexpr = SPLIT_N * p.SUBTILE_FACTOR

    state = Counter.create(0, p.acc_empty_bars.shape[0])
    scheduler = SchedulerImpl.initialize(p.c_desc.shape[0], p.c_desc.shape[1], BLOCK_M, BLOCK_N)

    num_buffers: gl.constexpr = 2
    outstanding_stores: gl.constexpr = 1
    store_iter = 0

    for idx in range(scheduler.get_num_tiles()):
        pid_m, pid_n = scheduler.get_tile(idx)
        off_m, off_n = pid_m * BLOCK_M, pid_n * BLOCK_N

        for i in gl.static_range(p.SUBTILE_FACTOR):
            mbarrier.wait(p.acc_ready_bars.index(state.index), state.phase)
            c_buf = p.acc_bufs.index(state.index)

            tma.async_copy_shared_to_global(p.c_desc, [off_m, off_n + i * SPLIT_N], c_buf)

            if store_iter >= outstanding_stores:
                tma.store_wait(outstanding_stores)
                empty_idx = (store_iter - outstanding_stores) % num_buffers
                mbarrier.arrive(p.acc_empty_bars.index(empty_idx), count=1)

            state = state.next()
            store_iter += 1

    tma.store_wait(0)

# ---------------------------------------------------------------------------
# KERNEL LAUNCHER
# ---------------------------------------------------------------------------

@gluon.jit
def sparse_matmul_warp_specialized_kernel(a_pruned_desc, b_desc, c_desc, SchedulerImpl: gl.constexpr,
                                          M, N, K, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K,
                                          num_buffers: gl.constexpr, SUBTILE_FACTOR: gl.constexpr,
                                          num_warps: gl.constexpr):
    dtype: gl.constexpr = a_pruned_desc.dtype

    a_pruned_bufs = gl.allocate_shared_memory(dtype, [num_buffers] + a_pruned_desc.block_type.shape, a_pruned_desc.layout)
    b_bufs = gl.allocate_shared_memory(dtype, [num_buffers] + b_desc.block_type.shape, b_desc.layout)
    load_empty_bars = gl.allocate_shared_memory(gl.int64, [num_buffers, 1], mbarrier.MBarrierLayout())
    load_ready_bars = gl.allocate_shared_memory(gl.int64, [num_buffers, 1], mbarrier.MBarrierLayout())

    for i in gl.static_range(num_buffers):
        mbarrier.init(load_empty_bars.index(i), count=1)
        mbarrier.init(load_ready_bars.index(i), count=1)

    acc_bufs = gl.allocate_shared_memory(dtype, [2] + c_desc.block_type.shape, c_desc.layout)
    acc_empty_bars = gl.allocate_shared_memory(gl.int64, [2, 1], mbarrier.MBarrierLayout())
    acc_ready_bars = gl.allocate_shared_memory(gl.int64, [2, 1], mbarrier.MBarrierLayout())

    for i in gl.static_range(2):
        mbarrier.init(acc_empty_bars.index(i), count=1)
        mbarrier.init(acc_ready_bars.index(i), count=1)

    p = SparsePartitionArgs(a_pruned_desc, b_desc, c_desc, a_pruned_bufs, b_bufs,
                            load_empty_bars, load_ready_bars,
                            acc_bufs, acc_empty_bars, acc_ready_bars, SUBTILE_FACTOR, num_warps)

    gl.warp_specialize([
        (sparse_matmul_compute_partition, (p, SchedulerImpl)),
        (sparse_matmul_load_partition, (p, SchedulerImpl)),
        (sparse_matmul_store_partition, (p, SchedulerImpl)),
    ], [1, 1], [24, 24])


def sparse_matmul_get_configs(pre_hook=None, tune=True):
    def valid(BM, BN, BK, warps, buffers, SF):
        smem_bytes = 2 * (
                (buffers * BM * BK) +
                (buffers * BK * BN) +
                (2 * BM * (BN // SF))
        ) + (16 * buffers) + 32
        if smem_bytes > 232448: return False

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

        elements_per_thread = (BM * BN) / (warps * 32)
        required_regs = elements_per_thread + 48
        max_regs_per_thread = 65536 // (warps * 32)
        max_regs_per_thread = min(255, max_regs_per_thread)
        if required_regs > max_regs_per_thread: return False
        if elements_per_thread < 16: return False
        return True

    configs = [
        triton.Config(
            {
                "BLOCK_SIZE_M": BM,
                "BLOCK_SIZE_N": BN,
                "BLOCK_SIZE_K": BK,
                "num_buffers": buffers,
                "SUBTILE_FACTOR": SF,
            },
            num_warps=warps,
            pre_hook=pre_hook,
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
    nargs["b_desc"].block_shape = [block_k, block_n]
    nargs["c_desc"].block_shape = [block_m, split_n]

    nargs["a_pruned_desc"].layout = gl.NVMMASharedLayout.get_default_for(nargs["a_pruned_desc"].block_shape, gl.float16)
    nargs["b_desc"].layout = gl.NVMMASharedLayout.get_default_for(nargs["b_desc"].block_shape, gl.float16)
    nargs["c_desc"].layout = gl.NVMMASharedLayout.get_default_for(nargs["c_desc"].block_shape, gl.float16)

sparse_ws_kernel_autotune = triton.autotune(
    configs=sparse_matmul_get_configs(pre_hook=sparse_matmul_tma_set_block_size_hook, tune=True),
    key=["M", "N", "K"],
    do_bench = lambda kernel_call, quantiles: triton.testing.do_bench_cudagraph(
        kernel_call, rep=100, quantiles=quantiles),
)(sparse_matmul_warp_specialized_kernel)

sparse_ws_kernel_single = triton.autotune(
    configs=sparse_matmul_get_configs(pre_hook=sparse_matmul_tma_set_block_size_hook, tune=False),
    key=["M", "N", "K"],
)(sparse_matmul_warp_specialized_kernel)

def run_sparse_ws_matmul(A_pruned, B, tune=True, manual_config=None):
    M, K = A_pruned.shape[0], A_pruned.shape[1]
    N = B.shape[1]

    c = torch.empty((M, N), device=A_pruned.device, dtype=torch.float16)
    dummy_block = [1, 1]
    dummy_layout_f16 = gl.NVMMASharedLayout.get_default_for(dummy_block, gl.float16)
    
    a_pruned_desc = TensorDescriptor.from_tensor(A_pruned, dummy_block, dummy_layout_f16)
    b_desc = TensorDescriptor.from_tensor(B, dummy_block, dummy_layout_f16)
    c_desc = TensorDescriptor.from_tensor(c, dummy_block, dummy_layout_f16)

    if tune:
        def grid(meta):
            num_sms = torch.cuda.get_device_properties("cuda").multi_processor_count
            num_pid = triton.cdiv(M, meta["BLOCK_SIZE_M"]) * triton.cdiv(N, meta["BLOCK_SIZE_N"])
            return (min(num_sms, num_pid), )
            
        sparse_ws_kernel_autotune[grid](a_pruned_desc, b_desc, c_desc, GroupedPersistentTileScheduler(8), M, N, K)
    else:
        hook_kwargs = {
            "BLOCK_SIZE_M": manual_config["BM"],
            "BLOCK_SIZE_N": manual_config["BN"],
            "BLOCK_SIZE_K": manual_config["BK"],
            "SUBTILE_FACTOR": manual_config["SF"],
            "a_pruned_desc": a_pruned_desc, "b_desc": b_desc, "c_desc": c_desc
        }
        
        sparse_matmul_tma_set_block_size_hook(hook_kwargs)
        
        num_sms = torch.cuda.get_device_properties("cuda").multi_processor_count
        num_pid = triton.cdiv(M, manual_config["BM"]) * triton.cdiv(N, manual_config["BN"])
        grid = (min(num_sms, num_pid), )
        
        sparse_matmul_warp_specialized_kernel[grid](
            a_pruned_desc, b_desc, c_desc, GroupedPersistentTileScheduler(8),
            M, N, K,
            BLOCK_SIZE_M=manual_config["BM"], 
            BLOCK_SIZE_N=manual_config["BN"], 
            BLOCK_SIZE_K=manual_config["BK"],
            num_buffers=manual_config["buffers"], 
            SUBTILE_FACTOR=manual_config["SF"], 
            num_warps=manual_config["warps"]
        )

    return c

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Fused-Compression Sparse Warp-Specialized Matmul with LUT Optimization")
    parser.add_argument("--tune", action="store_true", help="Enable Triton autotuning")
    
    # Manual config arguments (ignored if --tune is passed)
    parser.add_argument("--bm", type=int, default=128, help="BLOCK_SIZE_M")
    parser.add_argument("--bn", type=int, default=256, help="BLOCK_SIZE_N")
    parser.add_argument("--bk", type=int, default=64, help="BLOCK_SIZE_K")
    parser.add_argument("--warps", type=int, default=8, help="Number of warps")
    parser.add_argument("--buffers", type=int, default=3, help="Number of buffers")
    parser.add_argument("--sf", type=int, default=4, help="SUBTILE_FACTOR")
    
    args = parser.parse_args()

    manual_config = {
        "BM": args.bm,
        "BN": args.bn,
        "BK": args.bk,
        "warps": args.warps,
        "buffers": args.buffers,
        "SF": args.sf
    }

    os.environ["MLIR_ENABLE_DUMP"]="1"
    os.environ["MLIR_DUMP_PATH"] = "./MLIR_DUMP/7.6.4"
    os.environ["TRITON_ALWAYS_COMPILE"]="1"
    os.environ["TRITON_CACHE_DIR"]="./compiler_scratch/.triton_cache"

    for M, N, K in [(49152, 8192, 49152)]:

        if args.tune:
            print(f"Testing 7.6.4_compression_ws_optimization (AUTOTUNE ON): M={M}, N={N}, K={K}...", end="\n", flush=True)
        else:
            print(f"Testing 7.6.4_compression_ws_optimization with config {manual_config}: M={M}, N={N}, K={K}...", end="\n", flush=True)

        A = torch.randn(M, K, device="cuda", dtype=torch.float16)
        B = torch.randn((K, N), device="cuda", dtype=torch.float16)

        A_pruned = prune_2_4(A)

        C = run_sparse_ws_matmul(A_pruned, B, tune=args.tune, manual_config=manual_config)
        C_ref = A_pruned @ B

        torch.testing.assert_close(C_ref, C, rtol=1e-3, atol=1e-1)
        print("PASSED")
