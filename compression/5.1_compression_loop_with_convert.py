# import pytest
import torch
import time
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
def sparse_compress_blocked_matmul_kernel(
    a_pruned_desc,
    b_desc,
    c_desc,
    a_compressed_desc,
    e_desc,
    TRANSPOSE_B: gl.constexpr,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    BLOCK_K: gl.constexpr,
    num_warps: gl.constexpr,
):
    dtype: gl.constexpr = a_pruned_desc.dtype
    K = a_pruned_desc.shape[1]

    # Allocate shared memory for input tiles
    a_pruned_smem = gl.allocate_shared_memory(dtype, a_pruned_desc.block_type.shape, a_pruned_desc.layout)
    b_smem = gl.allocate_shared_memory(dtype, b_desc.block_type.shape, b_desc.layout)

    # smem allocated for debugging
    a_comp_out_smem = gl.allocate_shared_memory(dtype, a_compressed_desc.block_type.shape, a_compressed_desc.layout)
    e_out_smem = gl.allocate_shared_memory(gl.int16, e_desc.block_type.shape, e_desc.layout)

    # Determine program id and offsets
    pid_m = gl.program_id(axis=0)
    pid_n = gl.program_id(axis=1)
    off_m = pid_m * BLOCK_M
    off_n = pid_n * BLOCK_N

    # Determine the sparse WGMMA layout and initialize accumulator
    mma_layout: gl.constexpr = pick_sparse_wgmma_layout(dtype, BLOCK_M, BLOCK_N, num_warps)
    acc = gl.zeros((BLOCK_M, BLOCK_N), dtype=gl.float32, layout=mma_layout)

    # Allocate and initialize barrier
    bar = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
    mbarrier.init(bar, count=1)
    phase = 0

    # localize data transfers in registers
    idx_layout: gl.constexpr = gl.BlockedLayout(
        [1, 4],
        [4, 8],
        [num_warps, 1],
        [1, 0]
    )
    col_idx = gl.arange(0, BLOCK_K//4, gl.SliceLayout(0, idx_layout))[None, :]*4
    row_idx = gl.arange(0, BLOCK_M, gl.SliceLayout(1, idx_layout))[:, None]*0
    slice_idx = row_idx + col_idx

    # Layout for loading A pruned tile

    # trivially convert a_compressed layout to DotOpreandLayout
    a_compressed_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=0,
        parent=mma_layout,
        k_width=32 // a_pruned_desc.dtype.primitive_bitwidth,
        meta=0,
    )

    # Layouts for metadata index and slice (fully dynamic)
    parent_n: gl.constexpr = BLOCK_K // 16
    parent_m: gl.constexpr = 32 // parent_n
    size_m: gl.constexpr = BLOCK_M // (parent_m * num_warps)

    meta_idx_layout: gl.constexpr = gl.BlockedLayout(
        [1, 4], [4, 8], [num_warps, 1], [1, 0]
    )
    meta_col = gl.arange(0, BLOCK_K // 16, gl.SliceLayout(0, meta_idx_layout))[None, :] * 4
    meta_row = gl.arange(0, BLOCK_M, gl.SliceLayout(1, meta_idx_layout))[:, None] * 0
    meta_slice = meta_row + meta_col

    # DistributedLinearLayout for metadata reordering
    if num_warps == 4:
        e_reg_layout: gl.constexpr = gl.DistributedLinearLayout(
            reg_bases=[[0, 1], [0, 64]],
            lane_bases=[[0, 2], [0, 4], [0, 8], [0, 16], [0, 32]],
            warp_bases=[[1, 0], [2, 0]],
            block_bases=[],
            shape=[4, 128],
        )
    elif num_warps == 8:
        e_reg_layout: gl.constexpr = gl.DistributedLinearLayout(
            reg_bases=[[0, 1], [0, 64]],
            lane_bases=[[0, 2], [0, 4], [0, 8], [0, 16], [0, 32]],
            warp_bases=[[1, 0], [2, 0], [4, 0]],  # Shift third warp bit by 4 rows along internal axis
            block_bases=[],
            shape=[8, 128],  # Safely expand track capacity to accommodate the shift bounds
        )
    out_idx_layout: gl.constexpr = e_reg_layout

    # Dynamic generation of source_col and row_idx for metadata gather
    c = gl.arange(0, BLOCK_K, gl.SliceLayout(0, out_idx_layout))[None, :]
    c0 = c & 1
    c1 = (c >> 1) & 1
    c2 = (c >> 2) & 1
    c345 = c & 0x38
    c6 = (c >> 6) & 1
    source_col = (c0 << 6) | c345 | (c6 << 2) | (c2 << 1) | c1

    meta_row_idx = gl.arange(0, BLOCK_M // 16, gl.SliceLayout(1, out_idx_layout))[:, None] * 0
    meta_slice_idx = meta_row_idx + source_col
    for k in range(0, K, BLOCK_K):
        # Load tiles of A_pruned and B
        mbarrier.expect(bar, a_pruned_desc.block_type.nbytes + b_desc.block_type.nbytes)
        tma.async_copy_global_to_shared(a_pruned_desc, [off_m, k], bar, a_pruned_smem)
        if TRANSPOSE_B:
            tma.async_copy_global_to_shared(b_desc, [off_n, k], bar, b_smem)
        else:
            tma.async_copy_global_to_shared(b_desc, [k, off_n], bar, b_smem)
        mbarrier.wait(bar, phase=phase)
        phase ^= 1

        # Transpose B if necessary
        if TRANSPOSE_B:
            b = b_smem.permute((1, 0))
        else:
            b = b_smem

        # 1. Compress A tile in shared memory & Generate and Pack Metadata
        a_pruned = a_pruned_smem.load(a_compressed_layout)
        a0 = a_pruned.gather(slice_idx, 1)
        a1 = a_pruned.gather(slice_idx + 1, 1)
        a2 = a_pruned.gather(slice_idx + 2, 1)
        a3 = a_pruned.gather(slice_idx + 3, 1)

        m0 = a0 != 0
        m1 = a1 != 0
        m3 = a3 != 0

        bit0 = ~m0 & m1
        bit1 = ~m0 & ~m1
        bit2 = (m0 & m1) | (~m0 & ~m1) | m3
        bit3 = (~m0 & m1) | ~m1

        idx0 = bit0 | (bit1.to(gl.int16) << 1)
        idx1 = bit2 | (bit3.to(gl.int16) << 1)

        nz0 = gl.where(idx0 == 0, a0, gl.where(idx0 == 1, a1, gl.where(idx0 == 2, a2, a3)))
        nz1 = gl.where(idx1 == 0, a0, gl.where(idx1 == 1, a1, gl.where(idx1 == 2, a2, a3)))

        a_compressed = gl.join(nz0, nz1)
        a_compressed = a_compressed.reshape(BLOCK_M, BLOCK_K // 2)

        # write back to global for debugging
        # if pid_n == 0:
        #     a_comp_out_smem.store(a_compressed)
        #     fence_async_shared()
        #     tma.async_copy_shared_to_global(a_compressed_desc, [off_m, k // 2], a_comp_out_smem)
        #     tma.store_wait(pendings=0)

        # Since the current layout of a_compressed is equivalent of a_compressed_layout
        # we can set assert_trivial=True so there there's no overhead
        a_compressed = gl.convert_layout(a_compressed, a_compressed_layout, assert_trivial = False)

        meta_4 = idx0 | (idx1 << 2)

        mn0 = meta_4.gather(meta_slice, 1).to(gl.int16)
        mn1 = meta_4.gather(meta_slice + 1, 1).to(gl.int16)
        mn2 = meta_4.gather(meta_slice + 2, 1).to(gl.int16)
        mn3 = meta_4.gather(meta_slice + 3, 1).to(gl.int16)

        meta = mn0 | (mn1 << 4) | (mn2 << 8) | (mn3 << 12)
        meta_reshaped = meta.reshape((BLOCK_M // 16, BLOCK_K))

        # Reorder metadata for Hopper sparse MMA instruction format
        meta_reordered = meta_reshaped.gather(meta_slice_idx, 1)
        # print(meta_reordered.type)

        # write back to global for debugging
        # if pid_n == 0:
        #     e_out_smem.store(meta_reordered)
        #     fence_async_shared()
        #     tma.async_copy_shared_to_global(e_desc, [off_m // 16, k], e_out_smem)
        #     tma.store_wait(pendings=0)

        fence_async_shared()

        # 3. Call warpgroup_mma with compressed A and metadata reordered
        acc = warpgroup_mma(a_compressed, b, acc, e=meta_reordered, is_async=True)
        acc = warpgroup_mma_wait(num_outstanding=0, deps=(acc, ))

        gl.barrier()
        
    mbarrier.invalidate(bar)
    # acc = warpgroup_mma_wait(num_outstanding=0, deps = (acc, ))

    # Downcast accumulator and store tile of C.
    c_smem = gl.allocate_shared_memory(dtype, c_desc.block_type.shape, c_desc.layout)
    c_smem.store(acc.to(dtype))
    fence_async_shared()
    tma.async_copy_shared_to_global(c_desc, [off_m, off_n], c_smem)
    tma.store_wait(pendings=0)


def sparse_compress_blocked_matmul(A_pruned, B, C, A_compressed, E, BLOCK_M, BLOCK_N, BLOCK_K, TRANSPOSE_B, num_warps):
    M, N = C.shape

    a_pruned_layout = gl.NVMMASharedLayout.get_default_for([BLOCK_M, BLOCK_K], gl.float16)
    a_pruned_desc = TensorDescriptor.from_tensor(A_pruned, [BLOCK_M, BLOCK_K], a_pruned_layout)

    # check after performing wgmma
    e_layout = gl.NVMMASharedLayout.get_default_for([BLOCK_M // 16, BLOCK_K], gl.int16)
    e_desc = TensorDescriptor.from_tensor(E, [BLOCK_M // 16, BLOCK_K], e_layout)
    a_compressed_layout = gl.NVMMASharedLayout.get_default_for([BLOCK_M, BLOCK_K // 2], gl.float16)
    a_compressed_desc = TensorDescriptor.from_tensor(A_compressed, [BLOCK_M, BLOCK_K // 2], a_compressed_layout)


    B_BLOCK_SHAPE = [BLOCK_N, BLOCK_K] if TRANSPOSE_B else [BLOCK_K, BLOCK_N]
    b_layout = gl.NVMMASharedLayout.get_default_for(B_BLOCK_SHAPE, gl.float16)
    b_desc = TensorDescriptor.from_tensor(B, B_BLOCK_SHAPE, b_layout)

    c_layout = gl.NVMMASharedLayout.get_default_for([BLOCK_M, BLOCK_N], gl.float16)
    c_desc = TensorDescriptor.from_tensor(C, [BLOCK_M, BLOCK_N], c_layout)

    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    sparse_compress_blocked_matmul_kernel[grid](
        a_pruned_desc,
        b_desc,
        c_desc,
        a_compressed_desc,
        e_desc,
        TRANSPOSE_B,
        BLOCK_M=BLOCK_M, 
        BLOCK_N=BLOCK_N,  
        BLOCK_K=BLOCK_K, 
        num_warps=num_warps,
    )


if __name__ == "__main__":
    os.environ["MLIR_ENABLE_DUMP"]="1"
    os.environ["MLIR_DUMP_PATH"] = "./MLIR_DUMP/5.1"
    os.environ["TRITON_ALWAYS_COMPILE"]="1"
    print("Testing 2:4 Sparse Blocked Matmul with Runtime Compression & Metadata Generation")
    print("==========================================================================")
    torch.set_printoptions(threshold=10_000, linewidth=200, edgeitems=16)
    
    test_configs = [
        # (M, N, K, BLOCK_M, BLOCK_N, BLOCK_K, TRANSPOSE_B, num_warps)
        (768, 768, 768, 64, 64, 128, False, 4),
        (768, 768, 768, 64, 64, 128, True, 4),
        (768, 768, 768, 64, 64, 128, False, 8),
        (768, 768, 768, 64, 64, 128, True, 8),
        (768, 768, 768, 128, 128, 128, False, 4),
        (768, 768, 768, 128, 128, 128, True, 4),
        (768, 768, 768, 128, 128, 128, False, 8),
        (768, 768, 768, 128, 128, 128, True, 8),
        (4096, 4096, 4096, 64, 64, 128, False, 4),
        (4096, 4096, 4096, 64, 64, 128, True, 4),
        (4096, 4096, 4096, 64, 64, 128, False, 8),
        (4096, 4096, 4096, 64, 64, 128, True, 8),
        (4096, 4096, 4096, 128, 128, 128, False, 4),
        (4096, 4096, 4096, 128, 128, 128, True, 4),
        (4096, 4096, 4096, 128, 128, 128, False, 8),
        (4096, 4096, 4096, 128, 128, 128, True, 8),
    ]

    for config in test_configs:
        M, N, K, BLOCK_M, BLOCK_N, BLOCK_K, TRANSPOSE_B, num_warps = config
        # if num_warps != 8 or K != 2048:
        #     continue
        print(f"Config: M={M}, N={N}, K={K}, BLOCK_M={BLOCK_M}, BLOCK_N={BLOCK_N}, BLOCK_K={BLOCK_K}, TRANSPOSE_B={TRANSPOSE_B}, num_warps={num_warps}...", end=" ", flush=True)
        
        A = torch.randn(M, K, device="cuda", dtype=torch.float16)
        A_pruned = prune_2_4(A)
        A_ref, E_ref = compress_dense_to_sparse(A_pruned)
        E_ref = E_ref.view(M // 16, K)
        
        B = torch.randn((N, K) if TRANSPOSE_B else (K, N), device="cuda", dtype=torch.float16)
        C = torch.empty(M, N, device="cuda", dtype=torch.float16)

        A_compressed = torch.empty((M, K // 2), device="cuda", dtype=torch.float16)
        E = torch.empty((M // 16, K), device="cuda", dtype=torch.int16)

        sparse_compress_blocked_matmul(A_pruned, B, C, A_compressed, E, BLOCK_M, BLOCK_N, BLOCK_K, TRANSPOSE_B, num_warps)

        C_ref = A_pruned @ (B.T if TRANSPOSE_B else B)
        # print("C:")
        # print(C[0:1, 0:50])
        # print(C[64:65, 0:50])
        # print("C_ref:")
        # print(C_ref[0:1, 0:50])
        # for i in range(0, 768):
        #     try:
        #         torch.testing.assert_close(C_ref[i, :], C[i, :], rtol=1e-3, atol=1e-1)
        #     except AssertionError:
        #         print(f"{i} row is incorrect")
        # torch.testing.assert_close(A_ref, A_compressed, rtol=0, atol=0)
        # torch.testing.assert_close(E_ref, E, rtol=0, atol=0)
        torch.testing.assert_close(C_ref, C, rtol=1e-3, atol=1e-1)
        print("PASSED")
        # time.sleep(1)

    print("\nAll tests passed successfully!")
