import torch
import triton
from typing import Union
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.language.core import _aggregate as aggregate
import os

from prune import prune_2_4
from compress_2_4 import compress_dense_to_sparse

from triton.experimental.gluon.nvidia.hopper import TensorDescriptor
from triton.experimental.gluon.language.nvidia.hopper import (
    tma,
    mbarrier,
    fence_async_shared,
    warpgroup_mma,
    warpgroup_mma_wait,
    warpgroup_mma_accumulator,
)


@gluon.constexpr_function
def get_warps_per_cta(BLOCK_M, BLOCK_N, num_warps):
    warps_per_cta = [4, 1]
    m = 16
    # Tile the atom until we have enough warps.
    while warps_per_cta[0] * warps_per_cta[1] != num_warps:
        # Tile along M only if it would not cause broadcasting.
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
def pick_wgmma_layout(dtype, BLOCK_M, BLOCK_N, num_warps):
    m = 16
    k = 256 // dtype.primitive_bitwidth
    n = get_instr_shape_n(BLOCK_M, BLOCK_N, num_warps)
    warps_per_cta = get_warps_per_cta(BLOCK_M, BLOCK_N, num_warps)
    return gl.NVMMADistributedLayout(
        version=[3, 0],
        warps_per_cta=warps_per_cta,
        instr_shape=[m, n, k],
    )


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
        # --- Extract groups of 4 consecutive columns using reshape + split ---
        a_grouped = a_pruned.reshape(BLOCK_M, BLOCK_K // 4, 2, 2)
        a_even, a_odd = a_grouped.split()

        # split again to separate the pairs
        a0, a2 = a_even.split()  # a0 = col 4g+0, a2 = col 4g+2
        a1, a3 = a_odd.split()   # a1 = col 4g+1, a3 = col 4g+3

        # OPTIMIZATION 1: Cache the non-zero checks.
        # This stops the compiler from redundantly executing `!= 0` over and over.
        # We don't need b3 because in a strict 2:4 matrix, if it's not the first 3, it must be b3.
        b0 = a0 != 0
        b1 = a1 != 0
        b2 = a2 != 0
        # OPTIMIZATION 2: Streamlined value extraction.
        # We bypass idx0/idx1 entirely and route the values directly based on the booleans.
        nz0 = gl.where(b0, a0, gl.where(b1, a1, a2))

        # a1 is the second value only if b0 is also true.
        # a2 is the second value if b2 is true AND either b0 or b1 was the first non-zero.
        nz1 = gl.where(b0 & b1, a1, gl.where(b2 & (b0 | b1), a2, a3))

        a_compressed = gl.join(nz0, nz1)
        a_compressed = a_compressed.reshape(BLOCK_M, BLOCK_K // 2)

        # OPTIMIZATION 3: Direct metadata generation.
        # Instead of bitwise index math (idx0 | idx1 << 2), we use a minimal decision tree 
        # to directly yield the exact 2:4 metadata hex values.
        meta_4 = gl.where(b0,
             gl.where(b1, 4, gl.where(b2, 8, 12)),
             gl.where(b1, gl.where(b2, 9, 13), 14))

        # To lower register usage, we do the reshape and permute BEFORE the reduction!
        # This prevents permuting a replicated layout, which causes massive register spills.
        meta_4_reshaped = meta_4.reshape(BLOCK_M // 16, 2, 8, BLOCK_K // 64, 4, 2, 2)
        meta_4_permuted = meta_4_reshaped.permute(0, 3, 2, 4, 1, 5, 6)
        meta_4_ready = meta_4_permuted.reshape(BLOCK_M // 16, BLOCK_K, 2, 2)
        
        # gl.static_print(gl.to_linear_layout(meta_4_ready.type.layout, (BLOCK_M // 16, BLOCK_K, 2, 2)))
        
        meta_reordered = gl.reduce(
            gl.reduce(meta_4_ready, 3, create_metadata), 2, create_metadata_8
        ).to(gl.int16)
        #######################################################################################

        e_layout: gl.constexpr = gl.DotOperandLayout(
            operand_index=0,
            parent=self.layout,
            k_width=32 // gl.int16.primitive_bitwidth,
            meta=1,
        )

        # gl.static_print(gl.to_linear_layout(a_compressed.type.layout, (BLOCK_M, BLOCK_K // 2)))
        a_compressed = gl.convert_layout(
            a_compressed, a_compressed_layout, assert_trivial=False
        )
        # gl.static_print(gl.to_linear_layout(a_compressed.type.layout, (BLOCK_M, BLOCK_K // 2)))
        e = gl.convert_layout(meta_reordered, e_layout)
        
        return a_compressed, e

    @gluon.jit
    def issue_async_mma(
        self,
        a,
        b,
        a_pruned_reg_layout: gl.constexpr,
        a_compressed_layout: gl.constexpr,
        BLOCK_M: gl.constexpr,
        BLOCK_K: gl.constexpr,
    ):
        # 1. Compress A tile in shared memory & Generate and Pack Metadata
        a_pruned = a.load(a_pruned_reg_layout)
        
        # test bank conflicts
        # a_dis_type = gl.distributed_type(gl.float16, [BLOCK_M, BLOCK_K], a_pruned_reg_layout)
        # gl.static_print(gl.bank_conflicts(a_dis_type, a.type))
        
        # a_pruned = gl.convert_layout(a_pruned, a_pruned_reg_layout)

        a_compressed, e = self.generate_compressed_and_meta(a_pruned, BLOCK_M, BLOCK_K, a_compressed_layout)

        acc = warpgroup_mma(
            a_compressed,
            b,
            self.acc,
            e=e,
            is_async=True,
            use_acc=self.use_acc,
        )
        # Note that aggregates don't support in-place mutation, so we need to
        # return a new instance and re-assign it at the callsite.
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


@aggregate
class PersistentTileScheduler:
    pid_start: gl.tensor
    pid_end: gl.tensor
    num_pid_m: gl.tensor

    @gluon.constexpr_function
    def __init__(self, pid_start, pid_end, num_pid_m):
        self.pid_start = pid_start
        self.pid_end = pid_end
        self.num_pid_m = num_pid_m

    @gluon.jit
    def initialize(M, N, BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr):
        kernel_id = gl.program_id(axis=0)
        num_kernels = gl.num_programs(axis=0)
        num_pid_m = gl.cdiv(M, BLOCK_M)
        num_pid_n = gl.cdiv(N, BLOCK_N)
        num_pid = num_pid_m * num_pid_n
        pid_per_kernel = gl.cdiv(num_pid, num_kernels)
        pid_start = kernel_id * pid_per_kernel
        pid_end = min(pid_start + pid_per_kernel, num_pid)
        return PersistentTileScheduler(pid_start, pid_end, num_pid_m)

    @gluon.jit
    def get_num_tiles(self):
        return self.pid_end - self.pid_start

    @gluon.jit
    def get_tile(self, idx):
        # Delinearize the tile ID along M.
        pid = self.pid_start + idx
        pid_m = pid % self.num_pid_m
        pid_n = pid // self.num_pid_m
        return pid_m, pid_n


@gluon.jit
def issue_sparse_loads_stealb(
    producer,
    a_pruned_desc,
    b_desc,
    off_m,
    off_n,
    k,
    bars,
    a_pruned_bufs,
    b_bufs,
    stealb: gl.constexpr,
    num_buffers: gl.constexpr,
    pred=True,
):
    index = producer % num_buffers
    b_index = producer % (num_buffers + stealb)
    producer += 1
    bar = bars.index(index)
    mbarrier.expect(
        bar,
        a_pruned_desc.block_type.nbytes + b_desc.block_type.nbytes,
        pred,
    )
    tma.async_copy_global_to_shared(
        a_pruned_desc, [off_m, k], bar, a_pruned_bufs.index(index), pred
    )
    tma.async_copy_global_to_shared(
        b_desc, [k, off_n], bar, b_bufs.index(b_index), pred
    )
    return producer


@gluon.jit
def issue_sparse_mma_stealb(
    consumer,
    mma,
    bars,
    a_pruned_bufs,
    b_bufs,
    a_pruned_reg_layout: gl.constexpr,
    stealb: gl.constexpr,
    num_buffers: gl.constexpr,
    a_compressed_layout: gl.constexpr,
    BLOCK_M: gl.constexpr,
    BLOCK_K: gl.constexpr,
):
    index = consumer % num_buffers
    b_index = consumer % (num_buffers + stealb)
    phase = consumer // num_buffers & 1
    consumer += 1
    mbarrier.wait(bars.index(index), phase)
    mma = mma.wait_num_outstanding(0)
    mma = mma.issue_async_mma(
        a_pruned_bufs.index(index),
        b_bufs.index(b_index),
        a_pruned_reg_layout,
        a_compressed_layout,
        BLOCK_M,
        BLOCK_K,
    )
    return consumer, mma


@gluon.jit
def sparse_persistent_matmul_pipelined_kernel(
    a_pruned_desc,
    b_desc,
    c_desc,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    BLOCK_K: gl.constexpr,
    MMAImpl: gl.constexpr,
    SchedulerImpl: gl.constexpr,
    num_buffers: gl.constexpr,
    STEALB: gl.constexpr,
    num_warps: gl.constexpr,
):
    dtype: gl.constexpr = a_pruned_desc.dtype
    K = a_pruned_desc.shape[1]

    # All buffers share the same liverange.
    gl.static_assert(num_buffers >= 3, "expected at least 3 buffers")
    a_pruned_bufs = gl.allocate_shared_memory(
        dtype, [num_buffers] + a_pruned_desc.block_type.shape, a_pruned_desc.layout
    )
    # Add an extra B buffer when stealing.
    b_bufs = gl.allocate_shared_memory(
        dtype, [num_buffers + STEALB] + b_desc.block_type.shape, b_desc.layout
    )
    if not STEALB:
        c_smem = gl.allocate_shared_memory(
            dtype, c_desc.block_type.shape, c_desc.layout
        )
    else:
        gl.static_assert(
            2 * BLOCK_N * BLOCK_K >= BLOCK_M * BLOCK_N,
            "B tile not large enough to steal",
        )
    bars = gl.allocate_shared_memory(
        gl.int64, [num_buffers, 1], mbarrier.MBarrierLayout()
    )
    for i in gl.static_range(num_buffers):
        mbarrier.init(bars.index(i), count=1)
    producer = 0
    consumer = 0

    mma = MMAImpl.initialize(dtype, BLOCK_M, BLOCK_N, num_warps)
    scheduler = SchedulerImpl.initialize(
        c_desc.shape[0], c_desc.shape[1], BLOCK_M, BLOCK_N
    )
    num_tiles = scheduler.get_num_tiles()

    ############################################
    # Initializing layouts and index for wgmma #
    ############################################

    if num_warps == 4:
        a_warp_bases: gl.constexpr = [[16, 0], [32, 0]]
    elif num_warps == 8:
        a_warp_bases: gl.constexpr = [[16, 0], [32, 0], [64, 0]]
    elif num_warps == 16:
        a_warp_bases: gl.constexpr = [[16, 0], [32, 0], [64, 0], [128, 0]]
    
    a_pruned_reg_layout: gl.constexpr = gl.DistributedLinearLayout(
        reg_bases=[[0, 1], [0, 2], [8, 0], [0, 8], [0, 16]],
        lane_bases=[[0, 4], [0, 32], [1, 0], [2, 0], [4, 0]],
        warp_bases=a_warp_bases,
        block_bases=[],
        shape=[16 * num_warps, 64],
    )

    # trivially convert a_compressed layout to DotOpreandLayout
    a_compressed_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=0,
        parent=mma.layout,
        k_width=32 // a_pruned_desc.dtype.primitive_bitwidth,
        meta=0,
    )
    ##############################################################################

    # Peeled inner loop prologue.
    idx = 0
    pid_m, pid_n = scheduler.get_tile(idx)
    off_m = pid_m * BLOCK_M
    off_n = pid_n * BLOCK_N
    for ki in gl.static_range(0, BLOCK_K * (num_buffers - 2), BLOCK_K):
        producer = issue_sparse_loads_stealb(
            producer,
            a_pruned_desc,
            b_desc,
            off_m,
            off_n,
            ki,
            bars,
            a_pruned_bufs,
            b_bufs,
            STEALB,
            num_buffers,
        )
    k = BLOCK_K * (num_buffers - 2)
    producer = issue_sparse_loads_stealb(
        producer,
        a_pruned_desc,
        b_desc,
        off_m,
        off_n,
        k,
        bars,
        a_pruned_bufs,
        b_bufs,
        STEALB,
        num_buffers,
    )
    for _ in range(num_tiles):
        consumer, mma = issue_sparse_mma_stealb(
            consumer,
            mma,
            bars,
            a_pruned_bufs,
            b_bufs,
            a_pruned_reg_layout,
            STEALB,
            num_buffers,
            a_compressed_layout,
            BLOCK_M,
            BLOCK_K,
        )
        if STEALB:
            # Wait for the epilogue before the first TMA load.
            tma.store_wait(pendings=0)
        for k in range(BLOCK_K * (num_buffers - 1), K, BLOCK_K):
            producer = issue_sparse_loads_stealb(
                producer,
                a_pruned_desc,
                b_desc,
                off_m,
                off_n,
                k,
                bars,
                a_pruned_bufs,
                b_bufs,
                STEALB,
                num_buffers,
            )
            consumer, mma = issue_sparse_mma_stealb(
                consumer,
                mma,
                bars,
                a_pruned_bufs,
                b_bufs,
                a_pruned_reg_layout,
                STEALB,
                num_buffers,
                a_compressed_layout,
                BLOCK_M,
                BLOCK_K,
            )

        epilogue_off_m = off_m
        epilogue_off_n = off_n

        # Peel the next prologue and fuse it with the pipeline drain loop.
        idx += 1
        pid_m, pid_n = scheduler.get_tile(idx)
        off_m = pid_m * BLOCK_M
        off_n = pid_n * BLOCK_N
        # Predicate the peeled prologue instead of using a conditional.
        pred = idx < num_tiles
        for ki in gl.static_range(0, BLOCK_K * (num_buffers - 2), BLOCK_K):
            producer = issue_sparse_loads_stealb(
                producer,
                a_pruned_desc,
                b_desc,
                off_m,
                off_n,
                ki,
                bars,
                a_pruned_bufs,
                b_bufs,
                STEALB,
                num_buffers,
                pred,
            )
            consumer, mma = issue_sparse_mma_stealb(
                consumer,
                mma,
                bars,
                a_pruned_bufs,
                b_bufs,
                a_pruned_reg_layout,
                STEALB,
                num_buffers,
                a_compressed_layout,
                BLOCK_M,
                BLOCK_K,
            )
        k = BLOCK_K * (num_buffers - 2)
        producer = issue_sparse_loads_stealb(
            producer,
            a_pruned_desc,
            b_desc,
            off_m,
            off_n,
            k,
            bars,
            a_pruned_bufs,
            b_bufs,
            STEALB,
            num_buffers,
        )

        mma = mma.wait_num_outstanding(0)
        c, mma = mma.take_result()
        c = c.to(dtype)
        if not STEALB:
            c_buf = c_smem
            tma.store_wait(pendings=0)
        else:
            # Steal the next 2 B buffers for the epilogue.
            c_buf = b_bufs.index(producer % (num_buffers + STEALB))._reinterpret(
                dtype, c_desc.block_type.shape, c_desc.layout
            )
        c_buf.store(c)
        fence_async_shared()
        tma.async_copy_shared_to_global(c_desc, [epilogue_off_m, epilogue_off_n], c_buf)
    tma.store_wait(pendings=0)


def sparse_persistent_matmul_pipelined(
    A_pruned, B, C, BLOCK_M, BLOCK_N, BLOCK_K, num_buffers, num_warps, SchedulerImpl
):
    M, N = C.shape

    a_pruned_layout = gl.NVMMASharedLayout.get_default_for(
        [BLOCK_M, BLOCK_K], gl.float16
    )
    b_layout = gl.NVMMASharedLayout.get_default_for([BLOCK_K, BLOCK_N], gl.float16)
    c_layout = gl.NVMMASharedLayout.get_default_for([BLOCK_M, BLOCK_N], gl.float16)

    a_pruned_desc = TensorDescriptor.from_tensor(
        A_pruned, [BLOCK_M, BLOCK_K], a_pruned_layout
    )
    b_desc = TensorDescriptor.from_tensor(B, [BLOCK_K, BLOCK_N], b_layout)
    c_desc = TensorDescriptor.from_tensor(C, [BLOCK_M, BLOCK_N], c_layout)

    num_sms = torch.cuda.get_device_properties("cuda").multi_processor_count
    num_pid = triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N)
    grid = (min(num_sms, num_pid),)
    sparse_persistent_matmul_pipelined_kernel[grid](
        a_pruned_desc,
        b_desc,
        c_desc,
        BLOCK_M,
        BLOCK_N,
        BLOCK_K,
        SparseWGMMA,
        SchedulerImpl,
        num_buffers,
        STEALB=num_buffers == 4,
        num_warps=num_warps,
    )


if __name__ == "__main__":
    os.environ["MLIR_ENABLE_DUMP"] = "1"
    os.environ["MLIR_DUMP_PATH"] = "./MLIR_DUMP/7.3"
    os.environ["TRITON_ALWAYS_COMPILE"] = "1"
    for M, N, K in [(49152, 16, 49152)]:
        for BLOCK_M, BLOCK_N, BLOCK_K in [(64, 64, 128)]:
            for num_warps in [4]:
                # print(f"Testing dense persistent: M={M}, N={N}, K={K}, BLOCK_M={BLOCK_M}, BLOCK_N={BLOCK_N}, BLOCK_K={BLOCK_K}, num_warps={num_warps}...", end=" ", flush=True)

                A = torch.randn(M, K, device="cuda", dtype=torch.float16)
                B = torch.randn((K, N), device="cuda", dtype=torch.float16)
                C = torch.empty(M, N, device="cuda", dtype=torch.float16)

                A_pruned = prune_2_4(A)

                # sparse_persistent_matmul(A, E, B, C, BLOCK_M, BLOCK_N, BLOCK_K, 3, num_warps, PersistentTileScheduler)
                sparse_persistent_matmul_pipelined(
                    A_pruned,
                    B,
                    C,
                    BLOCK_M,
                    BLOCK_N,
                    BLOCK_K,
                    5,
                    num_warps,
                    PersistentTileScheduler,
                )
                C_ref = A_pruned @ B
                torch.testing.assert_close(C_ref, C, rtol=1e-3, atol=1e-1)
                print("PASSED")
