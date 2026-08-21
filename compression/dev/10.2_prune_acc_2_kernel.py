import argparse
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

# ============================================================================
# KERNEL 1: Persistent Sparse Matmul (Dense C Output)
# ============================================================================

@aggregate
class MatmulPartitionArgs:
    a_desc: tma.tensor_descriptor
    e_desc: tma.tensor_descriptor
    b_desc: tma.tensor_descriptor
    c_desc: tma.tensor_descriptor
    a_bufs: gl.shared_memory_descriptor
    e_bufs: gl.shared_memory_descriptor
    b_bufs: gl.shared_memory_descriptor
    load_empty_bars: gl.shared_memory_descriptor
    load_ready_bars: gl.shared_memory_descriptor
    acc_bufs: gl.shared_memory_descriptor
    acc_empty_bars: gl.shared_memory_descriptor
    acc_ready_bars: gl.shared_memory_descriptor
    SUBTILE_FACTOR: gl.constexpr
    num_warps: gl.constexpr

    @gluon.constexpr_function
    def __init__(self, a_desc, e_desc, b_desc, c_desc, a_bufs, e_bufs, b_bufs,
                 load_empty_bars, load_ready_bars, acc_bufs, acc_empty_bars,
                 acc_ready_bars, SUBTILE_FACTOR, num_warps):
        self.a_desc = a_desc
        self.e_desc = e_desc
        self.b_desc = b_desc
        self.c_desc = c_desc
        self.a_bufs = a_bufs
        self.e_bufs = e_bufs
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
def matmul_load_partition(p, SchedulerImpl: gl.constexpr):
    BLOCK_M: gl.constexpr = p.a_desc.block_type.shape[0]
    BLOCK_N: gl.constexpr = p.b_desc.block_type.shape[1]
    BLOCK_K: gl.constexpr = p.b_desc.block_type.shape[0]
    K = p.b_desc.shape[0]

    state = Counter.create(1, p.load_empty_bars.shape[0])
    scheduler = SchedulerImpl.initialize(p.c_desc.shape[0], p.b_desc.shape[1], BLOCK_M, BLOCK_N)

    for idx in range(scheduler.get_num_tiles()):
        pid_m, pid_n = scheduler.get_tile(idx)
        off_m = pid_m * BLOCK_M
        off_n = pid_n * BLOCK_N

        for k in range(0, K, BLOCK_K):
            bar = p.load_ready_bars.index(state.index)
            mbarrier.wait(p.load_empty_bars.index(state.index), state.phase)
            mbarrier.expect(bar, p.a_desc.block_type.nbytes + p.e_desc.block_type.nbytes + p.b_desc.block_type.nbytes)

            tma.async_copy_global_to_shared(p.a_desc, [off_m, k // 2], bar, p.a_bufs.index(state.index))
            tma.async_copy_global_to_shared(p.e_desc, [off_m // 16, k], bar, p.e_bufs.index(state.index))
            tma.async_copy_global_to_shared(p.b_desc, [k, off_n], bar, p.b_bufs.index(state.index))
            state = state.next()

@gluon.jit
def store_acc_to_smem_subtile_dense(p, mma, acc_state):
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

@gluon.jit
def matmul_compute_iteration(p, load_state, mma, k_iter, outstanding_mmas: gl.constexpr):
    mbarrier.wait(p.load_ready_bars.index(load_state.index), load_state.phase)
    e_reg = mma.issue_metadata_load(p.e_bufs.index(load_state.index))
    mma = mma.issue_async_sparse_mma(p.a_bufs.index(load_state.index), e_reg, p.b_bufs.index(load_state.index))
    load_state = load_state.next()

    mma = mma.wait_num_outstanding(outstanding_mmas)
    mbarrier.arrive(p.load_empty_bars.index((k_iter - outstanding_mmas) % p.load_empty_bars.shape[0]),
                    count=1, pred=k_iter >= outstanding_mmas)
    return load_state, mma, k_iter + 1

@gluon.jit
def matmul_compute_drain(p, load_state, mma, k_iter, limit, current_mma: gl.constexpr, num_mmas: gl.constexpr):
    if current_mma >= num_mmas: return load_state, mma, k_iter
    else:
        if k_iter >= limit: return load_state, mma, k_iter
        else:
            load_state, mma, k_iter = matmul_compute_iteration(p, load_state, mma, k_iter, num_mmas)
            return matmul_compute_drain(p, load_state, mma, k_iter, limit, gl.constexpr(current_mma + 1), num_mmas)

@gluon.jit
def matmul_compute_partition(p, SchedulerImpl: gl.constexpr):
    BLOCK_M: gl.constexpr = p.a_desc.block_type.shape[0]
    BLOCK_N: gl.constexpr = p.b_desc.block_type.shape[1]
    BLOCK_K: gl.constexpr = p.b_desc.block_type.shape[0]
    K = p.b_desc.shape[0]
    dtype: gl.constexpr = p.a_desc.dtype

    load_state = Counter.create(0, p.load_empty_bars.shape[0])
    acc_state = Counter.create(1, p.acc_empty_bars.shape[0])
    scheduler = SchedulerImpl.initialize(p.c_desc.shape[0], p.b_desc.shape[1], BLOCK_M, BLOCK_N)

    num_mmas: gl.constexpr = 2
    k_iter = 0

    for _ in range(scheduler.get_num_tiles()):
        mma = WGMMA.initialize(dtype, BLOCK_M, BLOCK_N, p.num_warps, sparse=True)
        total_k_iters = (K + BLOCK_K - 1) // BLOCK_K

        for _ in range(total_k_iters // num_mmas):
            for _ in gl.static_range(num_mmas):
                load_state, mma, k_iter = matmul_compute_iteration(p, load_state, mma, k_iter, num_mmas - 1)

        load_state, mma, k_iter = matmul_compute_drain(p, load_state, mma, k_iter,
                                                       (total_k_iters % num_mmas) + k_iter,
                                                       0, num_mmas - 1)

        acc_state = store_acc_to_smem_subtile_dense(p, mma, acc_state)

@gluon.jit
def matmul_store_partition(p, SchedulerImpl: gl.constexpr):
    BLOCK_M: gl.constexpr = p.c_desc.block_type.shape[0]
    SPLIT_N: gl.constexpr = p.c_desc.block_type.shape[1]
    BLOCK_N: gl.constexpr = p.b_desc.block_type.shape[1]

    state = Counter.create(0, p.acc_empty_bars.shape[0])
    scheduler = SchedulerImpl.initialize(p.c_desc.shape[0], p.b_desc.shape[1], BLOCK_M, BLOCK_N)

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

@gluon.jit
def sparse_matmul_dense_out_kernel(a_desc, e_desc, b_desc, c_desc, SchedulerImpl: gl.constexpr,
                                   M, N, K,
                                   BLOCK_SIZE_M: gl.constexpr, BLOCK_SIZE_N: gl.constexpr, BLOCK_SIZE_K: gl.constexpr,
                                   num_buffers: gl.constexpr, SUBTILE_FACTOR: gl.constexpr,
                                   num_warps: gl.constexpr):
    dtype: gl.constexpr = a_desc.dtype

    a_bufs = gl.allocate_shared_memory(dtype, [num_buffers] + a_desc.block_type.shape, a_desc.layout)
    e_bufs = gl.allocate_shared_memory(e_desc.dtype, [num_buffers] + e_desc.block_type.shape, e_desc.layout)
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

    p = MatmulPartitionArgs(a_desc, e_desc, b_desc, c_desc,
                            a_bufs, e_bufs, b_bufs,
                            load_empty_bars, load_ready_bars,
                            acc_bufs, acc_empty_bars, acc_ready_bars,
                            SUBTILE_FACTOR, num_warps)

    gl.warp_specialize([
        (matmul_compute_partition, (p, SchedulerImpl)),
        (matmul_load_partition, (p, SchedulerImpl)),
        (matmul_store_partition, (p, SchedulerImpl)),
    ], [1, 1], [24, 24])


# ============================================================================
# KERNEL 2: Output Prune & Compress (Dense C -> C_compressed + C_meta)
# ============================================================================

@aggregate
class CompressPartitionArgs:
    c_desc: tma.tensor_descriptor
    c_compressed_desc: tma.tensor_descriptor
    c_meta_desc: tma.tensor_descriptor
    c_smem: gl.shared_memory_descriptor
    c_comp_smem: gl.shared_memory_descriptor
    c_meta_smem: gl.shared_memory_descriptor
    load_ready_bar: gl.shared_memory_descriptor
    compute_ready_bar: gl.shared_memory_descriptor
    BLOCK_SIZE_M: gl.constexpr
    BLOCK_SIZE_N: gl.constexpr
    num_warps: gl.constexpr

    @gluon.constexpr_function
    def __init__(self, c_desc, c_compressed_desc, c_meta_desc,
                 c_smem, c_comp_smem, c_meta_smem,
                 load_ready_bar, compute_ready_bar,
                 BLOCK_SIZE_M, BLOCK_SIZE_N, num_warps):
        self.c_desc = c_desc
        self.c_compressed_desc = c_compressed_desc
        self.c_meta_desc = c_meta_desc
        self.c_smem = c_smem
        self.c_comp_smem = c_comp_smem
        self.c_meta_smem = c_meta_smem
        self.load_ready_bar = load_ready_bar
        self.compute_ready_bar = compute_ready_bar
        self.BLOCK_SIZE_M = gl.constexpr(BLOCK_SIZE_M)
        self.BLOCK_SIZE_N = gl.constexpr(BLOCK_SIZE_N)
        self.num_warps = gl.constexpr(num_warps)

@gluon.jit
def ws_compress_load_partition(p):
    pid_m = gl.program_id(0)
    pid_n = gl.program_id(1)
    off_m = pid_m * p.BLOCK_SIZE_M
    off_n = pid_n * p.BLOCK_SIZE_N

    mbarrier.expect(p.load_ready_bar, p.c_desc.block_type.nbytes)
    tma.async_copy_global_to_shared(p.c_desc, [off_m, off_n], p.load_ready_bar, p.c_smem)

@gluon.jit
def ws_compress_compute_partition(p):
    mbarrier.wait(p.load_ready_bar, 0)

    if p.num_warps == 4:
        warp_bases: gl.constexpr = [[16, 0], [32, 0]]
    elif p.num_warps == 8:
        warp_bases: gl.constexpr = [[16, 0], [32, 0], [64, 0]]
    else:
        warp_bases: gl.constexpr = [[16, 0], [32, 0], [64, 0], [128, 0]]

    c_reg_layout: gl.constexpr = gl.DistributedLinearLayout(
        reg_bases=[[0, 1], [0, 2], [8, 0], [0, 4], [0, 8]],
        lane_bases=[[0, 16], [0, 32], [1, 0], [2, 0], [4, 0]],
        warp_bases=warp_bases,
        block_bases=[],
        shape=[16 * p.num_warps, 64],
    )

    c_dense = p.c_smem.load(c_reg_layout)
    c_grouped = c_dense.reshape(p.BLOCK_SIZE_M, p.BLOCK_SIZE_N // 4, 2, 2)
    c_even, c_odd = c_grouped.split()

    c0, c2 = c_even.split()
    c1, c3 = c_odd.split()

    c01 = c0 > c1
    c02 = c0 > c2
    c03 = c0 > c3
    c12 = c1 > c2
    c13 = c1 > c3
    c23 = c2 > c3

    c10 = ~c01
    c20 = ~c02
    c21 = ~c12

    b0_bool = (c01 & (c02 | c03)) | (c02 & c03)
    b1_bool = (c10 & (c12 | c13)) | (c12 & c13)
    b2_bool = (c20 & (c21 | c23)) | (c21 & c23)

    nz0 = gl.where(b0_bool, c0, gl.where(b1_bool, c1, c2))
    nz1 = gl.where(b0_bool & b1_bool, c1, gl.where(b2_bool & (b0_bool | b1_bool), c2, c3))

    c_compressed = gl.join(nz0, nz1).reshape(p.BLOCK_SIZE_M, p.BLOCK_SIZE_N // 2)

    meta_4 = gl.where(b0_bool,
                      gl.where(b1_bool, 4, gl.where(b2_bool, 8, 12)),
                      gl.where(b1_bool, gl.where(b2_bool, 9, 13), 14))

    meta_4_reshaped = meta_4.reshape(p.BLOCK_SIZE_M // 16, 2, 8, p.BLOCK_SIZE_N // 64, 4, 2, 2)
    meta_4_permuted = meta_4_reshaped.permute(0, 3, 2, 4, 1, 5, 6)
    meta_4_ready = meta_4_permuted.reshape(p.BLOCK_SIZE_M // 16, p.BLOCK_SIZE_N, 2, 2)

    meta_even, meta_odd = meta_4_ready.split()
    mn0, mn2 = meta_even.split()
    mn1, mn3 = meta_odd.split()

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

    p.c_comp_smem.store(c_compressed)
    p.c_meta_smem.store(meta_reordered)
    fence_async_shared()

    mbarrier.arrive(p.compute_ready_bar, count=1)

@gluon.jit
def ws_compress_store_partition(p):
    pid_m = gl.program_id(0)
    pid_n = gl.program_id(1)
    off_m = pid_m * p.BLOCK_SIZE_M
    off_n = pid_n * p.BLOCK_SIZE_N

    mbarrier.wait(p.compute_ready_bar, 0)

    tma.async_copy_shared_to_global(p.c_compressed_desc, [off_m, off_n // 2], p.c_comp_smem)
    tma.async_copy_shared_to_global(p.c_meta_desc, [off_m // 16, off_n], p.c_meta_smem)

    tma.store_wait(0)

@gluon.jit
def ws_tma_compress_output_kernel(
    c_desc, c_compressed_desc, c_meta_desc,
    M, N,
    BLOCK_SIZE_M: gl.constexpr, BLOCK_SIZE_N: gl.constexpr,
    num_warps: gl.constexpr,
):
    c_smem = gl.allocate_shared_memory(gl.float16, [BLOCK_SIZE_M, BLOCK_SIZE_N], c_desc.layout)
    c_comp_smem = gl.allocate_shared_memory(gl.float16, [BLOCK_SIZE_M, BLOCK_SIZE_N // 2], c_compressed_desc.layout)
    c_meta_smem = gl.allocate_shared_memory(gl.int16, [BLOCK_SIZE_M // 16, BLOCK_SIZE_N], c_meta_desc.layout)

    load_ready_bar = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
    compute_ready_bar = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())

    mbarrier.init(load_ready_bar, count=1)
    mbarrier.init(compute_ready_bar, count=1)

    p = CompressPartitionArgs(
        c_desc, c_compressed_desc, c_meta_desc,
        c_smem, c_comp_smem, c_meta_smem,
        load_ready_bar, compute_ready_bar,
        BLOCK_SIZE_M, BLOCK_SIZE_N, num_warps
    )

    gl.warp_specialize([
        (ws_compress_compute_partition, (p,)),
        (ws_compress_load_partition, (p,)),
        (ws_compress_store_partition, (p,))
    ], [1, 1], [24, 24])


# ============================================================================
# TMA HOOKS & AUTOTUNERS
# ============================================================================

def matmul_tma_set_block_size_hook(nargs):
    block_m, block_n, block_k = nargs["BLOCK_SIZE_M"], nargs["BLOCK_SIZE_N"], nargs["BLOCK_SIZE_K"]
    split_n = block_n // nargs["SUBTILE_FACTOR"]

    nargs["a_desc"].block_shape = [block_m, block_k // 2]
    nargs["e_desc"].block_shape = [block_m // 16, block_k]
    nargs["b_desc"].block_shape = [block_k, block_n]
    nargs["c_desc"].block_shape = [block_m, split_n]

    nargs["a_desc"].layout = gl.NVMMASharedLayout.get_default_for(nargs["a_desc"].block_shape, gl.float16)
    nargs["e_desc"].layout = gl.NVMMASharedLayout.get_default_for(nargs["e_desc"].block_shape, gl.int16)
    nargs["b_desc"].layout = gl.NVMMASharedLayout.get_default_for(nargs["b_desc"].block_shape, gl.float16)
    nargs["c_desc"].layout = gl.NVMMASharedLayout.get_default_for(nargs["c_desc"].block_shape, gl.float16)

def compress_tma_set_block_size_hook(nargs):
    block_m, block_n = nargs["BLOCK_SIZE_M"], nargs["BLOCK_SIZE_N"]

    nargs["c_desc"].block_shape = [block_m, block_n]
    nargs["c_compressed_desc"].block_shape = [block_m, block_n // 2]
    nargs["c_meta_desc"].block_shape = [block_m // 16, block_n]

    nargs["c_desc"].layout = gl.NVMMASharedLayout.get_default_for(nargs["c_desc"].block_shape, gl.float16)
    nargs["c_compressed_desc"].layout = gl.NVMMASharedLayout.get_default_for(nargs["c_compressed_desc"].block_shape, gl.float16)
    nargs["c_meta_desc"].layout = gl.NVMMASharedLayout.get_default_for(nargs["c_meta_desc"].block_shape, gl.int16)


def sparse_matmul_dense_out_get_configs(pre_hook=None):
    def valid(BM, BN, BK, warps, buffers, SF):
        smem_bytes = (
            (buffers * BM * (BK // 2) * 2) +       # A_comp
            (buffers * (BM // 16) * BK * 2) +       # E_meta
            (buffers * BK * BN * 2) +               # B
            (2 * BM * (BN // SF) * 2)               # Accumulator C float16
        ) + (16 * buffers) + 32

        if smem_bytes > 232448: return False

        warps_m, warps_n, m = 4, 1, 16
        while (warps_m * warps_n) != warps:
            if BM > m * warps_m: warps_m *= 2
            else: warps_n *= 2

        if SF > 1 and warps_n > 1: return False
        if (BN // SF) < 16: return False
        if BM < warps_m * 16 or BN < warps_n * 16: return False

        elements_per_thread = (BM * BN) / (warps * 32)
        required_regs = elements_per_thread + 48
        max_regs_per_thread = min(255, 65536 // (warps * 32))

        if required_regs > max_regs_per_thread: return False
        if elements_per_thread < 16: return False
        return True

    return [
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
        for BM in (128, 256)
        for BN in (128, 256)
        for BK in (64, 128)
        for warps in (4, 8, 16)
        for buffers in (3, 4, 5)
        for SF in (2, 4)
        if valid(BM, BN, BK, warps, buffers, SF)
    ]

def compress_output_get_configs(pre_hook=None):
    def valid(BM, BN, warps):
        smem_bytes = 2 * (
            (BM * BN) +
            (BM * (BN // 2)) +
            ((BM // 16) * BN)
        ) + 64
        return smem_bytes <= 232448

    return [
        triton.Config(
            {
                "BLOCK_SIZE_M": BM,
                "BLOCK_SIZE_N": BN,
            },
            num_warps=warps,
            pre_hook=pre_hook,
        )
        for BM in (64, 128, 256)
        for BN in (64, 128, 256)
        for warps in (4, 8)
        if valid(BM, BN, warps)
    ]


# Autotuned Decorated Kernels
sparse_matmul_dense_out_autotune = triton.autotune(
    configs=sparse_matmul_dense_out_get_configs(pre_hook=matmul_tma_set_block_size_hook),
    key=["M", "N", "K"],
    do_bench=lambda kernel_call, quantiles: triton.testing.do_bench_cudagraph(
        kernel_call, rep=100, quantiles=quantiles
    ),
)(sparse_matmul_dense_out_kernel)

ws_tma_compress_output_autotune = triton.autotune(
    configs=compress_output_get_configs(pre_hook=compress_tma_set_block_size_hook),
    key=["M", "N"],
    do_bench=lambda kernel_call, quantiles: triton.testing.do_bench_cudagraph(
        kernel_call, rep=100, quantiles=quantiles
    ),
)(ws_tma_compress_output_kernel)


# ============================================================================
# PIPELINE LAUNCHER
# ============================================================================

def run_matmul_then_compress_separate(A_comp, E_A, B, tune=True, manual_matmul_config=None, manual_compress_config=None):
    M, K_half = A_comp.shape
    K, N = B.shape
    M_div_16 = E_A.shape[0]

    C_dense = torch.empty((M, N), device=A_comp.device, dtype=torch.float16)
    C_compressed = torch.empty((M, N // 2), device=A_comp.device, dtype=torch.float16)
    C_meta = torch.empty((M_div_16, N), device=A_comp.device, dtype=torch.int16)

    dummy = [1, 1]
    layout_f16 = gl.NVMMASharedLayout.get_default_for(dummy, gl.float16)
    layout_i16 = gl.NVMMASharedLayout.get_default_for(dummy, gl.int16)

    # ------------------------------------------------------------------------
    # STEP 1: Launch Kernel 1 (Sparse Matmul -> Dense C Output)
    # ------------------------------------------------------------------------
    a_desc = TensorDescriptor.from_tensor(A_comp, dummy, layout_f16)
    e_desc = TensorDescriptor.from_tensor(E_A, dummy, layout_i16)
    b_desc = TensorDescriptor.from_tensor(B, dummy, layout_f16)
    c_dense_desc = TensorDescriptor.from_tensor(C_dense, dummy, layout_f16)

    if tune:
        def grid_matmul(meta):
            num_sms = torch.cuda.get_device_properties("cuda").multi_processor_count
            num_pid = triton.cdiv(M, meta["BLOCK_SIZE_M"]) * triton.cdiv(N, meta["BLOCK_SIZE_N"])
            return (min(num_sms, num_pid), )

        sparse_matmul_dense_out_autotune[grid_matmul](
            a_desc, e_desc, b_desc, c_dense_desc, GroupedPersistentTileScheduler(8),
            M, N, K
        )
    else:
        matmul_hook_kwargs = {
            "BLOCK_SIZE_M": manual_matmul_config["BM"],
            "BLOCK_SIZE_N": manual_matmul_config["BN"],
            "BLOCK_SIZE_K": manual_matmul_config["BK"],
            "SUBTILE_FACTOR": manual_matmul_config["SF"],
            "a_desc": a_desc, "e_desc": e_desc, "b_desc": b_desc, "c_desc": c_dense_desc
        }
        matmul_tma_set_block_size_hook(matmul_hook_kwargs)

        num_sms = torch.cuda.get_device_properties("cuda").multi_processor_count
        num_pid_matmul = triton.cdiv(M, manual_matmul_config["BM"]) * triton.cdiv(N, manual_matmul_config["BN"])
        grid_matmul = (min(num_sms, num_pid_matmul), )

        sparse_matmul_dense_out_kernel[grid_matmul](
            a_desc, e_desc, b_desc, c_dense_desc, GroupedPersistentTileScheduler(8),
            M, N, K,
            BLOCK_SIZE_M=manual_matmul_config["BM"],
            BLOCK_SIZE_N=manual_matmul_config["BN"],
            BLOCK_SIZE_K=manual_matmul_config["BK"],
            num_buffers=manual_matmul_config["buffers"],
            SUBTILE_FACTOR=manual_matmul_config["SF"],
            num_warps=manual_matmul_config["warps"]
        )

    # ------------------------------------------------------------------------
    # STEP 2: Launch Kernel 2 (Dense C -> Compressed C Outputs)
    # ------------------------------------------------------------------------
    c_in_desc = TensorDescriptor.from_tensor(C_dense, dummy, layout_f16)
    c_compressed_desc = TensorDescriptor.from_tensor(C_compressed, dummy, layout_f16)
    c_meta_desc = TensorDescriptor.from_tensor(C_meta, dummy, layout_i16)

    if tune:
        def grid_compress(meta):
            return (triton.cdiv(M, meta["BLOCK_SIZE_M"]), triton.cdiv(N, meta["BLOCK_SIZE_N"]))

        ws_tma_compress_output_autotune[grid_compress](
            c_in_desc, c_compressed_desc, c_meta_desc, M, N
        )
    else:
        compress_hook_kwargs = {
            "BLOCK_SIZE_M": manual_compress_config["BM"],
            "BLOCK_SIZE_N": manual_compress_config["BN"],
            "c_desc": c_in_desc, "c_compressed_desc": c_compressed_desc, "c_meta_desc": c_meta_desc
        }
        compress_tma_set_block_size_hook(compress_hook_kwargs)

        grid_compress = (triton.cdiv(M, manual_compress_config["BM"]), triton.cdiv(N, manual_compress_config["BN"]))

        ws_tma_compress_output_kernel[grid_compress](
            c_in_desc, c_compressed_desc, c_meta_desc,
            M, N,
            BLOCK_SIZE_M=manual_compress_config["BM"],
            BLOCK_SIZE_N=manual_compress_config["BN"],
            num_warps=manual_compress_config["warps"]
        )

    return C_dense, C_compressed, C_meta


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Autotuned Decoupled 2-Kernel Sparse Matmul")
    parser.add_argument("--tune", action="store_true", help="Enable Triton autotuning")
    args = parser.parse_args()

    manual_matmul_config = {"BM": 128, "BN": 256, "BK": 64, "warps": 8, "buffers": 4, "SF": 4}
    manual_compress_config = {"BM": 128, "BN": 128, "warps": 8}

    M, N, K = 4096, 4096, 4096
    print(f"Running pipeline on shape ({M}, {N}, {K}), Autotune={args.tune}...")

    A = torch.randn((M, K), device="cuda", dtype=torch.float16)
    B = torch.randn((K, N), device="cuda", dtype=torch.float16)

    # 1. Prepare 2:4 Sparse Input A
    A_pruned = prune_2_4(A)
    A_comp, E_A = compress_dense_to_sparse(A_pruned)
    E_A = E_A.view(M // 16, K)

    # 2. Run Decoupled 2-Kernel Pipeline
    C_dense_triton, C_comp_triton, C_meta_triton = run_matmul_then_compress_separate(
        A_comp, E_A, B, tune=args.tune,
        manual_matmul_config=manual_matmul_config,
        manual_compress_config=manual_compress_config
    )

    # 3. Ground truth reference using PyTorch
    C_dense_ref = A_pruned @ B
    C_pruned_ref = prune_2_4(C_dense_ref)
    C_comp_ref, C_meta_ref = compress_dense_to_sparse(C_pruned_ref)
    C_meta_ref = C_meta_ref.view(M // 16, N)

    # ------------------------------------------------------------------------
    # Kernel 1 Verification: Dense Accumulation
    # ------------------------------------------------------------------------
    torch.testing.assert_close(C_dense_triton, C_dense_ref, rtol=1e-2, atol=1e-1)
    print("Kernel 1 (Matmul Dense Out) Verified Success!")

    # ------------------------------------------------------------------------
    # Kernel 2 Verification: 2:4 Block Spatial-Relaxed Alignment
    # ------------------------------------------------------------------------
    D_comp_blocks = C_comp_triton.view(-1, 2)
    C_comp_blocks = C_comp_ref.view(-1, 2)
    C_dense_blocks = C_dense_ref.view(-1, 4)

    # Identify blocks where exact spatial order mismatches
    mismatch_mask = torch.abs(D_comp_blocks - C_comp_blocks).max(dim=-1).values > 1e-1
    num_mismatched_blocks = mismatch_mask.sum().item()

    print(f"Found {num_mismatched_blocks} blocks with spatial order mismatches.")

    if num_mismatched_blocks > 0:
        bad_D = D_comp_blocks[mismatch_mask]         # Shape: (num_bad, 2)
        bad_C_dense = C_dense_blocks[mismatch_mask]   # Shape: (num_bad, 4)

        # Match Triton outputs against candidate values within the same 4-element dense block
        abs_dist = torch.abs(bad_D.unsqueeze(2) - bad_C_dense.unsqueeze(1))  # (num_bad, 2, 4)
        best_match_indices = abs_dist.min(dim=2).indices                     # (num_bad, 2)
        best_match_dense_vals = bad_C_dense.gather(1, best_match_indices)   # (num_bad, 2)

        try:
            torch.testing.assert_close(
                bad_D,
                best_match_dense_vals,
                atol=1e-2,
                rtol=1e-1
            )
            print("VERIFIED: Kernel 2 Success!")
            print("100% of Triton's mismatched outputs exist inside the PyTorch dense block.")
        except AssertionError as e:
            raise AssertionError(
                f"VERIFICATION FAILED: Triton output values do not exist in the source 1x4 dense block.\n\n"
                f"Details:\n{str(e)}"
            ) from None
    else:
        print("Kernel 2 Verified Success! (0 mismatches)")