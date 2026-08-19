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

# Warp-Specialization

@aggregate
class PartitionArgs:
    a_desc: tma.tensor_descriptor
    b_desc: tma.tensor_descriptor
    c_desc: tma.tensor_descriptor
    a_bufs: gl.shared_memory_descriptor
    b_bufs: gl.shared_memory_descriptor
    load_empty_bars: gl.shared_memory_descriptor
    load_ready_bars: gl.shared_memory_descriptor
    acc_bufs: gl.shared_memory_descriptor
    acc_empty_bars: gl.shared_memory_descriptor
    acc_ready_bars: gl.shared_memory_descriptor
    SUBTILE_FACTOR: gl.constexpr
    num_warps: gl.constexpr

    @gluon.constexpr_function
    def __init__(self, a_desc, b_desc, c_desc, a_bufs, b_bufs, load_empty_bars, load_ready_bars, acc_bufs,
                 acc_empty_bars, acc_ready_bars, SUBTILE_FACTOR, num_warps):
        self.a_desc = a_desc
        self.b_desc = b_desc
        self.c_desc = c_desc
        self.a_bufs = a_bufs
        self.b_bufs = b_bufs
        self.load_empty_bars = load_empty_bars
        self.load_ready_bars = load_ready_bars
        self.acc_bufs = acc_bufs
        self.acc_empty_bars = acc_empty_bars
        self.acc_ready_bars = acc_ready_bars
        self.SUBTILE_FACTOR = gl.constexpr(SUBTILE_FACTOR)
        self.num_warps = gl.constexpr(num_warps)

@aggregate
class SparsePartitionArgs:
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
    def __init__(self, a_desc, e_desc, b_desc, c_desc, a_bufs, e_bufs, b_bufs, load_empty_bars, load_ready_bars, acc_bufs,
                 acc_empty_bars, acc_ready_bars, SUBTILE_FACTOR, num_warps):
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


# ---------------------------------------------------------------------------
# HELPER: Slices the accumulator registers along the N dimension
# ---------------------------------------------------------------------------
@gluon.jit
def _split_n(x, SUBTILE_FACTOR: gl.constexpr):
    split_count: gl.constexpr = SUBTILE_FACTOR.bit_length() - 1  # log2
    xs = (x, )
    for _ in gl.static_range(split_count):
        next_xs = ()
        for j in gl.static_range(len(xs)):
            x = xs[j]
            # Reshape to (M, 2, N//2) then permute so that tensor elements
            # remain contiguous along N.
            next_xs += x.reshape(x.shape[0], 2, x.shape[1] // 2).permute(0, 2, 1).split()
        xs = next_xs
    return xs

# ---------------------------------------------------------------------------
# PARTITIONS
# ---------------------------------------------------------------------------
@gluon.jit
def matmul_load_partition(p, SchedulerImpl: gl.constexpr):
    BLOCK_M: gl.constexpr = p.a_desc.block_type.shape[0]
    BLOCK_N: gl.constexpr = p.b_desc.block_type.shape[1]
    BLOCK_K: gl.constexpr = p.a_desc.block_type.shape[1]
    K = p.a_desc.shape[1]

    state = Counter.create(1, p.load_empty_bars.shape[0])
    scheduler = SchedulerImpl.initialize(p.c_desc.shape[0], p.c_desc.shape[1], BLOCK_M, BLOCK_N)

    for idx in range(scheduler.get_num_tiles()):
        pid_m, pid_n = scheduler.get_tile(idx)
        off_m = pid_m * BLOCK_M
        off_n = pid_n * BLOCK_N

        for k in range(0, K, BLOCK_K):
            bar = p.load_ready_bars.index(state.index)
            mbarrier.wait(p.load_empty_bars.index(state.index), state.phase)

            mbarrier.expect(bar, p.a_desc.block_type.nbytes + p.b_desc.block_type.nbytes)
            tma.async_copy_global_to_shared(p.a_desc, [off_m, k], bar, p.a_bufs.index(state.index))
            tma.async_copy_global_to_shared(p.b_desc, [k, off_n], bar, p.b_bufs.index(state.index))
            state = state.next()

@gluon.jit
def sparse_matmul_load_partition(p, SchedulerImpl: gl.constexpr):
    BLOCK_M: gl.constexpr = p.a_desc.block_type.shape[0]
    BLOCK_N: gl.constexpr = p.b_desc.block_type.shape[1]
    BLOCK_K: gl.constexpr = p.b_desc.block_type.shape[0]
    K = p.b_desc.shape[0]

    state = Counter.create(1, p.load_empty_bars.shape[0])
    scheduler = SchedulerImpl.initialize(p.c_desc.shape[0], p.c_desc.shape[1], BLOCK_M, BLOCK_N)

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

@gluon.jit
def matmul_compute_partition(p, SchedulerImpl: gl.constexpr):
    BLOCK_M: gl.constexpr = p.a_desc.block_type.shape[0]
    BLOCK_N: gl.constexpr = p.b_desc.block_type.shape[1]
    BLOCK_K: gl.constexpr = p.a_desc.block_type.shape[1]
    K = p.a_desc.shape[1]
    dtype: gl.constexpr = p.a_desc.dtype

    load_state = Counter.create(0, p.load_empty_bars.shape[0])
    acc_state = Counter.create(1, p.acc_empty_bars.shape[0])

    scheduler = SchedulerImpl.initialize(p.c_desc.shape[0], p.c_desc.shape[1], BLOCK_M, BLOCK_N)

    outstanding_mmas: gl.constexpr = 1
    k_iter = 0

    for _ in range(scheduler.get_num_tiles()):
        mma = WGMMA.initialize(dtype, BLOCK_M, BLOCK_N, p.num_warps)

        for _ in range(0, K, BLOCK_K):
            mbarrier.wait(p.load_ready_bars.index(load_state.index), load_state.phase)

            # Keep a shallow async pipeline instead of fully serializing each k-step.
            # a_reg_layout: gl.constexpr = gl.DotOperandLayout(
            #     operand_index=0,
            #     parent=mma.layout,
            #     k_width=2,
            #     meta=0,
            # )
            # a_reg = p.a_bufs.index(load_state.index).load(a_reg_layout)
            #
            # mma = mma.issue_async_mma(a_reg, p.b_bufs.index(load_state.index))

            mma = mma.issue_async_mma(p.a_bufs.index(load_state.index), p.b_bufs.index(load_state.index))

            load_state = load_state.next()
            mma = mma.wait_num_outstanding(outstanding_mmas)

            # If we've passed the outstanding limit, the WGMMA instruction from
            # (outstanding_mmas + 1) iterations ago is guaranteed to have finished.
            mbarrier.arrive(p.load_empty_bars.index((k_iter - outstanding_mmas) % p.load_empty_bars.shape[0]),
                            count=1, pred=k_iter>=outstanding_mmas)

            k_iter += 1

        acc_state = store_acc_to_smem_subtile(p, mma, acc_state)

@gluon.jit
def sparse_matmul_compute_iteration(p, load_state, mma, k_iter, outstanding_mmas: gl.constexpr):
    mbarrier.wait(p.load_ready_bars.index(load_state.index), load_state.phase)

    e_reg = mma.issue_metadata_load(p.e_bufs.index(load_state.index))
    mma = mma.issue_async_sparse_mma(p.a_bufs.index(load_state.index), e_reg, p.b_bufs.index(load_state.index))

    load_state = load_state.next()

    mma = mma.wait_num_outstanding(outstanding_mmas)

    mbarrier.arrive(p.load_empty_bars.index((k_iter - outstanding_mmas) % p.load_empty_bars.shape[0]),
                    count=1, pred=k_iter>=outstanding_mmas)

    return load_state, mma, k_iter + 1

@gluon.jit
def sparse_matmul_compute_drain(p, load_state, mma, k_iter, limit, current_mma: gl.constexpr, num_mmas: gl.constexpr):
    if current_mma >= num_mmas: return load_state, mma, k_iter
    else:
        if k_iter >= limit: return load_state, mma, k_iter
        else:
            load_state, mma, k_iter = sparse_matmul_compute_iteration(p, load_state, mma, k_iter, num_mmas)
            return sparse_matmul_compute_drain(p, load_state, mma, k_iter, limit, gl.constexpr(current_mma + 1), num_mmas)


@gluon.jit
def sparse_matmul_compute_partition(p, SchedulerImpl: gl.constexpr):
    BLOCK_M: gl.constexpr = p.a_desc.block_type.shape[0]
    BLOCK_N: gl.constexpr = p.b_desc.block_type.shape[1]
    BLOCK_K: gl.constexpr = p.b_desc.block_type.shape[0]
    K = p.b_desc.shape[0]
    dtype: gl.constexpr = p.a_desc.dtype

    load_state = Counter.create(0, p.load_empty_bars.shape[0])
    acc_state = Counter.create(1, p.acc_empty_bars.shape[0])

    scheduler = SchedulerImpl.initialize(p.c_desc.shape[0], p.c_desc.shape[1], BLOCK_M, BLOCK_N)

    num_mmas: gl.constexpr = 2
    k_iter = 0

    for _ in range(scheduler.get_num_tiles()):
        mma = WGMMA.initialize(dtype, BLOCK_M, BLOCK_N, p.num_warps, sparse=True)

        total_k_iters = (K + BLOCK_K - 1) // BLOCK_K

        # Statically Unrolled
        for _ in range(total_k_iters // num_mmas):
            for _ in gl.static_range(num_mmas):
                load_state, mma, k_iter = sparse_matmul_compute_iteration(p, load_state, mma, k_iter, num_mmas - 1)

        # Drain
        load_state, mma, k_iter = sparse_matmul_compute_drain(
            p, load_state, mma, k_iter,
            (total_k_iters % num_mmas) + k_iter,
            0, num_mmas - 1
        )

        acc_state = store_acc_to_smem_subtile(p, mma, acc_state)

@gluon.jit
def matmul_store_partition(p, SchedulerImpl: gl.constexpr):
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

            # Do not stall on the warmup iterations; release buffers once stores age out.
            if store_iter >= outstanding_stores:
                tma.store_wait(outstanding_stores)
                empty_idx = (store_iter - outstanding_stores) % num_buffers
                mbarrier.arrive(p.acc_empty_bars.index(empty_idx), count=1)

            state = state.next()
            store_iter += 1

    tma.store_wait(0)

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

            # Do not stall on the warmup iterations; release buffers once stores age out.
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
def matmul_warp_specialized_kernel(a_desc, b_desc, c_desc, SchedulerImpl: gl.constexpr,
                                   M, N, K, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K,
                                   num_buffers: gl.constexpr, SUBTILE_FACTOR: gl.constexpr,
                                   num_warps: gl.constexpr):
    dtype: gl.constexpr = a_desc.dtype

    a_bufs = gl.allocate_shared_memory(dtype, [num_buffers] + a_desc.block_type.shape, a_desc.layout)
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

    # Pass num_warps straight into the arguments so the layout generator gets it
    p = PartitionArgs(a_desc, b_desc, c_desc, a_bufs, b_bufs,
                      load_empty_bars, load_ready_bars,
                      acc_bufs, acc_empty_bars, acc_ready_bars, SUBTILE_FACTOR, num_warps)

    gl.warp_specialize([
        (matmul_compute_partition, (p, SchedulerImpl)),
        (matmul_load_partition, (p, SchedulerImpl)),
        (matmul_store_partition, (p, SchedulerImpl)),
    ], [1, 1], [24, 24])

@gluon.jit
def sparse_matmul_warp_specialized_kernel(a_desc, e_desc, b_desc, c_desc, SchedulerImpl: gl.constexpr,
                                          M, N, K, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K,
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

    # Pass num_warps straight into the arguments so the layout generator gets it
    p = SparsePartitionArgs(a_desc, e_desc, b_desc, c_desc, a_bufs, e_bufs, b_bufs,
                            load_empty_bars, load_ready_bars,
                            acc_bufs, acc_empty_bars, acc_ready_bars, SUBTILE_FACTOR, num_warps)

    gl.warp_specialize([
        (sparse_matmul_compute_partition, (p, SchedulerImpl)),
        (sparse_matmul_load_partition, (p, SchedulerImpl)),
        (sparse_matmul_store_partition, (p, SchedulerImpl)),
    ], [1, 1], [24, 24])

def matmul_get_configs(pre_hook=None):
    def valid(BM, BN, BK, warps, buffers, SF):
        # Shared Memory
        smem_bytes = 2 * (
                (buffers * BM * BK) +
                (buffers * BK * BN) +
                (2 * BM * (BN // SF))
        ) + (16 * buffers) + 32

        if smem_bytes > 232448: return False

        # if (BN // SF) < 32:
        #     return False

        # 1. Simulate get_warps_per_cta to find the physical N-axis distribution
        warps_m = 4
        warps_n = 1
        m = 16
        while (warps_m * warps_n) != warps:
            if BM > m * warps_m:
                warps_m *= 2
            else:
                warps_n *= 2

        # Prevent SPMD splitting across physical Warp Group boundaries.
        # If warps_n > 1, WG0 owns the left half and WG1 owns the right half.
        # Triton's split() cannot divide a tensor where the halves belong to different warps.
        if SF > 1 and warps_n > 1:
            return False

        # Ensure the subtile is physically large enough to split
        # Hopper WGMMA registers require at least 16 columns to give 2 elements per thread
        if (BN // SF) < 16:
            return False

        # STEALB
        # if SB and 2 * BN * BK < BM * BN: return False
        # if SB and BM > BK: return False
        #
        # if (BM * BN) >= 65536 and warps < 12:  # 256x256 blocks require at least 3 warp groups
        #     return False
        # if (BM * BN) <= 4096 and warps > 8:    # Tiny blocks will starve 12 or 16 warps
        #     return False
        #
        # elements_per_thread = (BM * BN) / (warps * 32)
        # if elements_per_thread > 256:
        #     return False

        if BM < warps_m * 16 or BN < warps_n * 16:
            return False

        # REGISTER EXHAUSTION:
        # elements_per_thread = (BM * BN) / (warps * 32)
        # if elements_per_thread > 256:
        #     return False

        elements_per_thread = (BM * BN) / (warps * 32)

        # Add a safe buffer (~48) for TMA pointers, loop state, and layout logic
        required_regs = elements_per_thread + 48

        # H100 absolute physical limits:
        # 65,536 registers per SM, divided evenly among block threads
        max_regs_per_thread = 65536 // (warps * 32)

        # Hardware caps any single thread to 255 registers
        max_regs_per_thread = min(255, max_regs_per_thread)

        if required_regs > max_regs_per_thread:
            return False

        # WARP STARVATION:
        if elements_per_thread < 16:
            return False

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
        for BM in (64, 128, 256,)
        for BN in (64, 128, 256,)#(64, 128, 256)
        for BK in (64, 128, 256,)#(64, 128, 256)
        for warps in (4, 8, 16)#(4, 8, 16)
        for buffers in (3, 4, 5, 6, 7)#(3, 4, 5, 6)
        for SF in (1, 2, 4, 8)#(2, 4, 8)
        if valid(BM, BN, BK, warps, buffers, SF)
    ]

def sparse_matmul_get_configs(pre_hook=None): # TODO: Fix the guards
    def valid(BM, BN, BK, warps, buffers, SF):
        # Shared Memory
        smem_bytes = (
                             (buffers * BM * BK) +               # Compressed A
                             (buffers * BM * BK // 8) +          # Metadata E
                             (buffers * BK * BN * 2) +           # Dense B
                             (4 * BM * (BN // SF))               # Accumulator C (2 buffers * 2 bytes)
                     ) + (16 * buffers) + 32                     # MBarriers

        if smem_bytes > 232448: return False

        # if (BN // SF) < 32:
        #     return False

        # Simulate get_warps_per_cta to find the physical N-axis distribution
        warps_m = 4
        warps_n = 1
        m = 16
        while (warps_m * warps_n) != warps:
            if BM > m * warps_m:
                warps_m *= 2
            else:
                warps_n *= 2

        # Prevent SPMD splitting across physical Warp Group boundaries.
        # If warps_n > 1, WG0 owns the left half and WG1 owns the right half.
        # Triton's split() cannot divide a tensor where the halves belong to different warps.
        if SF > 1 and warps_n > 1:
            return False

        # Ensure the subtile is physically large enough to split
        # Hopper WGMMA registers require at least 16 columns to give 2 elements per thread
        if (BN // SF) < 16:
            return False

        # STEALB
        # if SB and 2 * BN * BK < BM * BN: return False
        # if SB and BM > BK: return False
        #
        # if (BM * BN) >= 65536 and warps < 12:  # 256x256 blocks require at least 3 warp groups
        #     return False
        # if (BM * BN) <= 4096 and warps > 8:    # Tiny blocks will starve 12 or 16 warps
        #     return False
        #
        # elements_per_thread = (BM * BN) / (warps * 32)
        # if elements_per_thread > 256:
        #     return False

        if BM < warps_m * 16 or BN < warps_n * 16:
            return False

        elements_per_thread = (BM * BN) / (warps * 32)

        # Add a safe buffer (~48) for TMA pointers, loop state, and layout logic
        required_regs = elements_per_thread + 48

        # H100 absolute physical limits:
        # 65,536 registers per SM, divided evenly among block threads
        max_regs_per_thread = 65536 // (warps * 32)

        # Hardware caps any single thread to 255 registers
        max_regs_per_thread = min(255, max_regs_per_thread)

        if required_regs > max_regs_per_thread:
            return False

        # WARP STARVATION:
        if elements_per_thread < 16:
            return False

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
        for BM in (64, 128, 256,)
        for BN in (64, 128, 256,)#(64, 128, 256)
        for BK in (64, 128, 256,)#(64, 128, 256)
        for warps in (4, 8, 16)#(4, 8, 16)
        for buffers in (3, 4, 5, 6, 7)#(3, 4, 5, 6)
        for SF in (1, 2, 4, 8)#(2, 4, 8)
        if valid(BM, BN, BK, warps, buffers, SF)
    ]

def matmul_tma_set_block_size_hook(nargs):
    block_m = nargs["BLOCK_SIZE_M"]
    block_n = nargs["BLOCK_SIZE_N"]
    block_k = nargs["BLOCK_SIZE_K"]
    split_n = nargs["BLOCK_SIZE_N"] // nargs["SUBTILE_FACTOR"]

    nargs["a_desc"].block_shape = [block_m, block_k]
    nargs["b_desc"].block_shape = [block_k, block_n]
    nargs["c_desc"].block_shape = [block_m, split_n]

    nargs["a_desc"].layout = gl.NVMMASharedLayout.get_default_for(nargs["a_desc"].block_shape, gl.float16)
    nargs["b_desc"].layout = gl.NVMMASharedLayout.get_default_for(nargs["b_desc"].block_shape, gl.float16)
    nargs["c_desc"].layout = gl.NVMMASharedLayout.get_default_for(nargs["c_desc"].block_shape, gl.float16)

def sparse_matmul_tma_set_block_size_hook(nargs):
    block_m = nargs["BLOCK_SIZE_M"]
    block_n = nargs["BLOCK_SIZE_N"]
    block_k = nargs["BLOCK_SIZE_K"]
    split_n = nargs["BLOCK_SIZE_N"] // nargs["SUBTILE_FACTOR"]

    nargs["a_desc"].block_shape = [block_m, block_k // 2]
    nargs["e_desc"].block_shape = [block_m // 16, block_k]
    nargs["b_desc"].block_shape = [block_k, block_n]
    nargs["c_desc"].block_shape = [block_m, split_n]

    nargs["a_desc"].layout = gl.NVMMASharedLayout.get_default_for(nargs["a_desc"].block_shape, gl.float16)
    nargs["e_desc"].layout = gl.NVMMASharedLayout.get_default_for(nargs["e_desc"].block_shape, gl.int16)
    nargs["b_desc"].layout = gl.NVMMASharedLayout.get_default_for(nargs["b_desc"].block_shape, gl.float16)
    nargs["c_desc"].layout = gl.NVMMASharedLayout.get_default_for(nargs["c_desc"].block_shape, gl.float16)

ws_kernel = triton.autotune(
    configs=matmul_get_configs(pre_hook=matmul_tma_set_block_size_hook),
    key=["M", "N", "K"],
    do_bench = lambda kernel_call, quantiles: triton.testing.do_bench_cudagraph(
        kernel_call, quantiles=quantiles),
)(matmul_warp_specialized_kernel)

sparse_ws_kernel = triton.autotune(
    configs=sparse_matmul_get_configs(pre_hook=sparse_matmul_tma_set_block_size_hook),
    key=["M", "N", "K"],
    do_bench = lambda kernel_call, quantiles: triton.testing.do_bench_cudagraph(
        kernel_call, quantiles=quantiles),
)(sparse_matmul_warp_specialized_kernel)

def run_ws_matmul(A, B):
    M, N, K = A.shape[0], B.shape[1], B.shape[0]

    c = torch.empty((M, N), device=A.device, dtype=torch.float16)
    dummy_block = [1, 1]
    dummy_layout_f16 = gl.NVMMASharedLayout.get_default_for(dummy_block, gl.float16)
    a_desc = TensorDescriptor.from_tensor(A, dummy_block, dummy_layout_f16)
    b_desc = TensorDescriptor.from_tensor(B, dummy_block, dummy_layout_f16)
    c_desc = TensorDescriptor.from_tensor(c, dummy_block, dummy_layout_f16)

    def grid(meta):
        num_sms = torch.cuda.get_device_properties("cuda").multi_processor_count
        num_pid = triton.cdiv(M, meta["BLOCK_SIZE_M"]) * triton.cdiv(N, meta["BLOCK_SIZE_N"])
        return (min(num_sms, num_pid), )
    ws_kernel[grid](a_desc, b_desc, c_desc, GroupedPersistentTileScheduler(8), M, N, K)

    return c

def run_sparse_ws_matmul(A, E, B):
    M, N, K = A.shape[0], B.shape[1], B.shape[0]

    c = torch.empty((M, N), device=A.device, dtype=torch.float16)
    dummy_block = [1, 1]
    dummy_layout_f16 = gl.NVMMASharedLayout.get_default_for(dummy_block, gl.float16)
    dummy_layout_i16 = gl.NVMMASharedLayout.get_default_for(dummy_block, gl.int16)
    a_desc = TensorDescriptor.from_tensor(A, dummy_block, dummy_layout_f16)
    e_desc = TensorDescriptor.from_tensor(E, dummy_block, dummy_layout_i16)
    b_desc = TensorDescriptor.from_tensor(B, dummy_block, dummy_layout_f16)
    c_desc = TensorDescriptor.from_tensor(c, dummy_block, dummy_layout_f16)

    def grid(meta):
        num_sms = torch.cuda.get_device_properties("cuda").multi_processor_count
        num_pid = triton.cdiv(M, meta["BLOCK_SIZE_M"]) * triton.cdiv(N, meta["BLOCK_SIZE_N"])
        return (min(num_sms, num_pid), )
    sparse_ws_kernel[grid](a_desc, e_desc, b_desc, c_desc, GroupedPersistentTileScheduler(8), M, N, K)

    return c

if __name__ == "__main__":
    sizes = [
        (768, 768, 768),
        (768, 768, 896),
        (2048, 1024, 2048)
    ]


    from compress_2_4 import *
    from prune import *

    for M, N, K in sizes:
        A = torch.randn(M, K, device="cuda", dtype=torch.float16)
        B = torch.randn((K, N), device="cuda", dtype=torch.float16)
        C = torch.empty(M, N, device="cuda", dtype=torch.float16)
        D = run_ws_matmul(A,B)
        torch.testing.assert_close(A@ B, D, rtol=1e-3, atol=1e-1)
    print("Done dense.")

    for M, N, K in sizes:
        A = torch.randn(M, K, device="cuda", dtype=torch.float16)
        B = torch.randn((K, N), device="cuda", dtype=torch.float16)
        C = torch.empty(M, N, device="cuda", dtype=torch.float16)
        A_pruned = prune_2_4(A)
        A, E = compress_dense_to_sparse(A_pruned)
        E = E.view(M // 16, K)
        D = run_sparse_ws_matmul(A, E, B)
        torch.testing.assert_close(A_pruned @ B, D, rtol=1e-3, atol=1e-1)
    print("Done sparse.")
