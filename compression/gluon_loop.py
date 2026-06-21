# import pytest
import torch
import triton
import itertools
import os
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

from prune import prune_2_4
from compress_2_4 import compress_dense_to_sparse

from triton.experimental.gluon.nvidia.hopper import TensorDescriptor
from triton.experimental.gluon.language.nvidia.hopper import (
    tma,
    mbarrier,
    fence_async_shared,
    warpgroup_mma_init,
    warpgroup_mma,
    warpgroup_mma_wait,
)

os.environ["MLIR_ENABLE_DUMP"] = "1"
os.environ["MLIR_DUMP_PATH"] = "./MLIR_DUMP/dense_loop_diff_layout"
os.environ["TRITON_ALWAYS_COMPILE"] = "1"


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
def blocked_matmul_kernel(
    a_desc, b_desc, c_desc, TRANSPOSE_B: gl.constexpr, num_warps: gl.constexpr  #
):
    BLOCK_M: gl.constexpr = c_desc.block_type.shape[0]
    BLOCK_N: gl.constexpr = c_desc.block_type.shape[1]
    BLOCK_K: gl.constexpr = a_desc.block_type.shape[1]
    dtype: gl.constexpr = a_desc.dtype
    K = a_desc.shape[1]

    a_smem = gl.allocate_shared_memory(dtype, a_desc.block_type.shape, a_desc.layout)
    b_smem = gl.allocate_shared_memory(dtype, b_desc.block_type.shape, b_desc.layout)

    # The block of C this program is processing is (pid_m, pid_n).
    pid_m = gl.program_id(axis=0)
    pid_n = gl.program_id(axis=1)
    off_m = pid_m * BLOCK_M
    off_n = pid_n * BLOCK_N

    # Determine the WGMMA layout.
    mma_layout: gl.constexpr = pick_wgmma_layout(dtype, BLOCK_M, BLOCK_N, num_warps)
    acc = gl.zeros((BLOCK_M, BLOCK_N), dtype=gl.float32, layout=mma_layout)

    bar = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
    mbarrier.init(bar, count=1)
    phase = 0

    for k in range(0, K, BLOCK_K):
        # Load tiles of A and B.
        mbarrier.expect(bar, a_desc.block_type.nbytes + b_desc.block_type.nbytes)
        tma.async_copy_global_to_shared(a_desc, [off_m, k], bar, a_smem)
        if TRANSPOSE_B:
            tma.async_copy_global_to_shared(b_desc, [off_n, k], bar, b_smem)
        else:
            tma.async_copy_global_to_shared(b_desc, [k, off_n], bar, b_smem)
        mbarrier.wait(bar, phase=phase)
        phase ^= 1  # toggle the parity phase between 0 and 1

        # We can transpose B by creating a transposed view over tile of B in
        # shared memory. This forwards the transposition to WGMMA, which handles
        # it for us.
        if TRANSPOSE_B:
            b = b_smem.permute((1, 0))
        else:
            b = b_smem

        acc = warpgroup_mma(a_smem, b, acc, is_async=True)
        acc = warpgroup_mma_wait(num_outstanding=0, deps=(acc,))

    mbarrier.invalidate(bar)

    # Downcast accumulator and store tile of C.
    c_smem = gl.allocate_shared_memory(dtype, c_desc.block_type.shape, c_desc.layout)
    c_smem.store(acc.to(dtype))
    fence_async_shared()
    tma.async_copy_shared_to_global(c_desc, [off_m, off_n], c_smem)
    tma.store_wait(pendings=0)


def blocked_matmul(A, B, C, BLOCK_M, BLOCK_N, BLOCK_K, TRANSPOSE_B, num_warps):
    M, N = C.shape

    a_layout = gl.NVMMASharedLayout.get_default_for([BLOCK_M, BLOCK_K], gl.float16)
    a_desc = TensorDescriptor.from_tensor(A, [BLOCK_M, BLOCK_K], a_layout)

    B_BLOCK_SHAPE = [BLOCK_N, BLOCK_K] if TRANSPOSE_B else [BLOCK_K, BLOCK_N]
    b_layout = gl.NVMMASharedLayout.get_default_for(B_BLOCK_SHAPE, gl.float16)
    b_desc = TensorDescriptor.from_tensor(B, B_BLOCK_SHAPE, b_layout)

    c_layout = gl.NVMMASharedLayout.get_default_for([BLOCK_M, BLOCK_N], gl.float16)
    c_desc = TensorDescriptor.from_tensor(C, [BLOCK_M, BLOCK_N], c_layout)

    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    blocked_matmul_kernel[grid](
        a_desc, b_desc, c_desc, TRANSPOSE_B, num_warps=num_warps
    )


@gluon.jit
def sparse_blocked_matmul_kernel(
    a_desc,
    e_desc,
    b_desc,
    c_desc,  #
    TRANSPOSE_B: gl.constexpr,
    num_warps: gl.constexpr,
):
    BLOCK_M: gl.constexpr = c_desc.block_type.shape[0]
    BLOCK_N: gl.constexpr = c_desc.block_type.shape[1]
    BLOCK_K: gl.constexpr = a_desc.block_type.shape[1] * 2
    dtype: gl.constexpr = a_desc.dtype
    K = a_desc.shape[1] * 2

    a_smem = gl.allocate_shared_memory(dtype, a_desc.block_type.shape, a_desc.layout)
    e_smem = gl.allocate_shared_memory(
        e_desc.dtype, e_desc.block_type.shape, e_desc.layout
    )
    b_smem = gl.allocate_shared_memory(dtype, b_desc.block_type.shape, b_desc.layout)

    # The block of C this program is processing is (pid_m, pid_n).
    pid_m = gl.program_id(axis=0)
    pid_n = gl.program_id(axis=1)
    off_m = pid_m * BLOCK_M
    off_n = pid_n * BLOCK_N

    # Determine the WGMMA layout.
    mma_layout: gl.constexpr = pick_sparse_wgmma_layout(
        dtype, BLOCK_M, BLOCK_N, num_warps
    )
    acc = gl.zeros((BLOCK_M, BLOCK_N), dtype=gl.float32, layout=mma_layout)

    bar = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
    mbarrier.init(bar, count=1)
    phase = 0

    for k in range(0, K, BLOCK_K):
        # Load tiles of A and B.
        mbarrier.expect(
            bar,
            a_desc.block_type.nbytes
            + b_desc.block_type.nbytes
            + e_desc.block_type.nbytes,
        )
        tma.async_copy_global_to_shared(a_desc, [off_m, k // 2], bar, a_smem)
        tma.async_copy_global_to_shared(e_desc, [off_m // 16, k], bar, e_smem)
        if TRANSPOSE_B:
            tma.async_copy_global_to_shared(b_desc, [off_n, k], bar, b_smem)
        else:
            tma.async_copy_global_to_shared(b_desc, [k, off_n], bar, b_smem)
        mbarrier.wait(bar, phase=phase)
        phase ^= 1  # toggle the parity phase between 0 and 1

        # We can transpose B by creating a transposed view over tile of B in
        # shared memory. This forwards the transposition to WGMMA, which handles
        # it for us.
        if TRANSPOSE_B:
            b = b_smem.permute((1, 0))
        else:
            b = b_smem

        a_reg_layout: gl.constexpr = gl.BlockedLayout(
            size_per_thread=[1, 1],
            threads_per_warp=[1, 32],
            warps_per_cta=[1, 4],
            order=[1, 0],
        )

        e_reg_layout: gl.constexpr = gl.DotOperandLayout(
            operand_index=0,
            parent=mma_layout,
            k_width=32 // e_desc.dtype.primitive_bitwidth,
            meta=1,
        )

        a_wgmma_layout: gl.constexpr = gl.DotOperandLayout(
            operand_index=0,
            parent=mma_layout,
            k_width=32 // a_desc.dtype.primitive_bitwidth,
            meta=0,
        )

        # a = a_smem.load(a_wgmma_layout)
        a = a_smem.load(a_reg_layout)
        e = e_smem.load(e_reg_layout)


        a = gl.convert_layout(a, a_wgmma_layout)

        acc = warpgroup_mma(a, b, acc, e=e, is_async=True)
        acc = warpgroup_mma_wait(num_outstanding=0, deps=(acc,))

    # if pid_n == 0 and pid_m == 0:
    #     gl.static_print(gl.to_linear_layout(a_reg_layout, (BLOCK_M, BLOCK_K // 2)))
    mbarrier.invalidate(bar)

    # Downcast accumulator and store tile of C.
    c_smem = gl.allocate_shared_memory(dtype, c_desc.block_type.shape, c_desc.layout)
    c_smem.store(acc.to(dtype))
    fence_async_shared()
    tma.async_copy_shared_to_global(c_desc, [off_m, off_n], c_smem)
    tma.store_wait(pendings=0)


def sparse_blocked_matmul(
    A, E, B, C, BLOCK_M, BLOCK_N, BLOCK_K, TRANSPOSE_B, num_warps
):
    M, N = C.shape

    a_layout = gl.NVMMASharedLayout.get_default_for([BLOCK_M, BLOCK_K // 2], gl.float16)
    a_desc = TensorDescriptor.from_tensor(A, [BLOCK_M, BLOCK_K // 2], a_layout)

    e_layout = gl.NVMMASharedLayout.get_default_for([BLOCK_M // 16, BLOCK_K], gl.int16)
    e_desc = TensorDescriptor.from_tensor(E, [BLOCK_M // 16, BLOCK_K], e_layout)

    B_BLOCK_SHAPE = [BLOCK_N, BLOCK_K] if TRANSPOSE_B else [BLOCK_K, BLOCK_N]
    b_layout = gl.NVMMASharedLayout.get_default_for(B_BLOCK_SHAPE, gl.float16)
    b_desc = TensorDescriptor.from_tensor(B, B_BLOCK_SHAPE, b_layout)

    c_layout = gl.NVMMASharedLayout.get_default_for([BLOCK_M, BLOCK_N], gl.float16)
    c_desc = TensorDescriptor.from_tensor(C, [BLOCK_M, BLOCK_N], c_layout)

    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    sparse_blocked_matmul_kernel[grid](
        a_desc, e_desc, b_desc, c_desc, TRANSPOSE_B, num_warps=num_warps
    )


# @pytest.mark.parametrize("M, N, K", [(208, 416, 304), (2000, 1000, 2000)])
# @pytest.mark.parametrize("BLOCK_M, BLOCK_N, BLOCK_K", [(64, 64, 64), (128, 128, 128)])
# @pytest.mark.parametrize("TRANSPOSE_B", [False, True])
# @pytest.mark.parametrize("num_warps", [4, 8])
# def test_blocked_matmul(M, N, K, BLOCK_M, BLOCK_N, BLOCK_K, TRANSPOSE_B, num_warps):
#     A = torch.randn(M, K, device="cuda", dtype=torch.float16)
#     B = torch.randn((N, K) if TRANSPOSE_B else (K, N), device="cuda", dtype=torch.float16)
#     C = torch.empty(M, N, device="cuda", dtype=torch.float16)
#
#     blocked_matmul(A, B, C, BLOCK_M, BLOCK_N, BLOCK_K, TRANSPOSE_B, num_warps)
#
#     C_ref = A @ (B.T if TRANSPOSE_B else B)
#     torch.testing.assert_close(C_ref, C, rtol=1e-3, atol=1e-1)

if __name__ == "__main__":
    # for M, N, K in [(208, 416, 304), (2000, 1000, 2000)]:
    #     for BLOCK_M, BLOCK_N, BLOCK_K in [(64, 64, 64), (128, 128, 128)]:
    #         for TRANSPOSE_B in [False, True]:
    #             for num_warps in [4, 8]:
    #                 A = torch.randn(M, K, device="cuda", dtype=torch.float16)
    #                 B = torch.randn((N, K) if TRANSPOSE_B else (K, N), device="cuda", dtype=torch.float16)
    #                 C = torch.empty(M, N, device="cuda", dtype=torch.float16)
    #
    #                 blocked_matmul(A, B, C, BLOCK_M, BLOCK_N, BLOCK_K, TRANSPOSE_B, num_warps)
    #
    #                 C_ref = A @ (B.T if TRANSPOSE_B else B)
    #                 torch.testing.assert_close(C_ref, C, rtol=1e-3, atol=1e-1)
    test_configs = [
        # (M, N, K, BLOCK_M, BLOCK_N, BLOCK_K, TRANSPOSE_B, num_warps)
        (768, 768, 768, 64, 64, 128, False, 4),
        # (768, 768, 768, 64, 64, 128, True, 4),
        # (768, 768, 768, 64, 64, 128, False, 8),
        # # (768, 768, 768, 64, 64, 128, True, 8),
        # (768, 768, 768, 128, 128, 128, False, 4),
        # # (768, 768, 768, 128, 128, 128, True, 4),
        # (768, 768, 768, 128, 128, 128, False, 8),
        # # (768, 768, 768, 128, 128, 128, True, 8),
        # (2048, 1024, 2048, 64, 64, 128, False, 4),
        # # (2048, 1024, 2048, 64, 64, 128, True, 4),
        # (2048, 1024, 2048, 64, 64, 128, False, 8),
        # # (2048, 1024, 2048, 64, 64, 128, True, 8),
        # (2048, 1024, 2048, 128, 128, 128, False, 4),
        # # (2048, 1024, 2048, 128, 128, 128, True, 4),
        # (2048, 1024, 2048, 128, 128, 128, False, 8),
        # (2048, 1024, 2048, 128, 128, 128, True, 8),
    ]

    # for _ in range(10):
    for config in test_configs:
        M, N, K, BLOCK_M, BLOCK_N, BLOCK_K, TRANSPOSE_B, num_warps = config
        print(
            f"Config: M={M}, N={N}, K={K}, BLOCK_M={BLOCK_M}, BLOCK_N={BLOCK_N}, BLOCK_K={BLOCK_K}, TRANSPOSE_B={TRANSPOSE_B}, num_warps={num_warps}..."
        )

        A = torch.randn(M, K, device="cuda", dtype=torch.float16)
        A_pruned = prune_2_4(A)
        A, E = compress_dense_to_sparse(A_pruned)
        E = E.view(M // 16, K)

        B = torch.randn(
            (N, K) if TRANSPOSE_B else (K, N), device="cuda", dtype=torch.float16
        )
        C = torch.empty(M, N, device="cuda", dtype=torch.float16)

        sparse_blocked_matmul(
            A, E, B, C, BLOCK_M, BLOCK_N, BLOCK_K, TRANSPOSE_B, num_warps
        )

        C_ref = A_pruned @ (B.T if TRANSPOSE_B else B)
        torch.testing.assert_close(C_ref, C, rtol=1e-3, atol=1e-1)
