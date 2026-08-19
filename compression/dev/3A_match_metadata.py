import torch
import triton
import itertools
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


@gluon.jit
def small_mma_kernel(
    a_pruned_desc,
    e_desc,
    a_compressed_layout: gl.constexpr,
    BLOCK_M: gl.constexpr,
    BLOCK_K: gl.constexpr,
    INSTR_SHAPE_N: gl.constexpr,
    num_warps: gl.constexpr,
):

    bar = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
    mbarrier.init(bar, count=1)

    # Load A tiles.
    a_pruned_smem = gl.allocate_shared_memory(
        a_pruned_desc.dtype, a_pruned_desc.block_type.shape, a_pruned_desc.layout
    )
    e_smem = gl.allocate_shared_memory(
        e_desc.dtype, e_desc.block_type.shape, e_desc.layout
    )

    mbarrier.expect(bar, a_pruned_desc.block_type.nbytes)
    tma.async_copy_global_to_shared(a_pruned_desc, [0, 0], bar, a_pruned_smem)
    mbarrier.wait(bar, phase=0)
    mbarrier.invalidate(bar)

    ############################
    # new code for compression #
    ############################
    compress_shape: gl.constexpr = (
        a_pruned_desc.block_type.shape[0],
        a_pruned_desc.block_type.shape[1] // 2,
    )
    a_compressed_smem = gl.allocate_shared_memory(
        a_pruned_desc.dtype, compress_shape, a_compressed_layout
    )

    idx_layout: gl.constexpr = gl.BlockedLayout(
        [BLOCK_M // num_warps, 1], [1, 32], [num_warps, 1], [1, 0]
    )
    col_idx = gl.arange(0, BLOCK_K // 4, gl.SliceLayout(0, idx_layout))[None, :] * 4
    row_idx = gl.arange(0, BLOCK_M, gl.SliceLayout(1, idx_layout))[:, None] * 0
    slice_idx = row_idx + col_idx

    a_pruned_reg_layout: gl.constexpr = gl.BlockedLayout(
        [64 // num_warps, 1], [1, 32], [num_warps, 1], [1, 0]
    )
    a_pruned = a_pruned_smem.load(a_pruned_reg_layout)

    a0 = a_pruned.gather(slice_idx, 1)
    a1 = a_pruned.gather(slice_idx + 1, 1)
    a2 = a_pruned.gather(slice_idx + 2, 1)
    a3 = a_pruned.gather(slice_idx + 3, 1)

    nz0 = gl.where(a0 != 0, a0, gl.where(a1 != 0, a1, a2))
    nz1 = gl.where(a3 != 0, a3, gl.where(a2 != 0, a2, a1))

    a_compressed = gl.join(nz0, nz1)
    a_compressed = a_compressed.reshape(BLOCK_M, BLOCK_K // 2)

    # store a_compressed to smem for now
    a_compressed_smem.store(a_compressed)
    ######################################################################

    #########################
    # new code for metadata #
    #########################
    # copy and consolidate from compress_2_4.py to save register
    m0 = a0 != 0
    m1 = a1 != 0
    # m2 = a2 != 0      # m2 was never used
    m3 = a3 != 0

    # expr0 = m0 & m1
    # expr1 = ~m0 & m1
    # expr2 = ~m0 & ~m1

    bit0 = ~m0 & m1
    bit1 = ~m0 & ~m1
    bit2 = (m0 & m1) | (~m0 & ~m1) | m3
    bit3 = (~m0 & m1) | ~m1

    idx0 = bit0 | (bit1.to(gl.int16) << 1)
    idx1 = bit2 | (bit3.to(gl.int16) << 1)

    # 4-bit nibbles per group-of-4, shape (BLOCK_M, BLOCK_K//4)
    meta_4 = idx0 | (idx1 << 2)

    # Pack 4 consecutive nibbles into int16, same gather pattern as compression
    meta_idx_layout: gl.constexpr = gl.BlockedLayout(
        [4, 1], [4, 8], [num_warps, 1], [1, 0]
    )
    meta_col = (
        gl.arange(0, BLOCK_K // 16, gl.SliceLayout(0, meta_idx_layout))[None, :] * 4
    )
    meta_row = gl.arange(0, BLOCK_M, gl.SliceLayout(1, meta_idx_layout))[:, None] * 0
    meta_slice = meta_row + meta_col

    mn0 = meta_4.gather(meta_slice, 1).to(gl.int16)
    mn1 = meta_4.gather(meta_slice + 1, 1).to(gl.int16)
    mn2 = meta_4.gather(meta_slice + 2, 1).to(gl.int16)
    mn3 = meta_4.gather(meta_slice + 3, 1).to(gl.int16)

    # Pack nibbles: matches compress_2_4.py's meta_n[:,:,0] | (meta_n[:,:,1]<<4) | ...
    meta = mn0 | (mn1 << 4) | (mn2 << 8) | (mn3 << 12)
    # meta shape: (BLOCK_M, BLOCK_K//16) = (64, 8)

    # Reorder metadata to match Hopper's sparse TMA format
    # This involves reshaping to (4, 128) and swapping bits 2 and 6 of the column index
    meta_reshaped = meta.reshape((BLOCK_M // 16, BLOCK_K))

    out_idx_layout: gl.constexpr = gl.BlockedLayout(
        [1, 4], [4, 8], [1, num_warps], [1, 0]
    )

    c = gl.arange(0, BLOCK_K, gl.SliceLayout(0, out_idx_layout))[None, :]
    if BLOCK_K == 256:
        c0 = c & 1
        source_col = (c0 << 7) | ((c & 0x38) << 1) | ((c & 0xc0) >> 4) | ((c & 0x6) >> 1)
    elif BLOCK_K == 128:
        c0 = c & 1
        c1 = (c >> 1) & 1
        c2 = (c >> 2) & 1
        c345 = c & 0x38
        c6 = (c >> 6) & 1
        source_col = (c0 << 6) | c345 | (c6 << 2) | (c2 << 1) | c1
    elif BLOCK_K == 64:
        c0 = c & 1
        source_col = (c0 << 5) | ((c & 0x38) >> 1) | ((c & 0x6) >> 1)
    else:
        raise ValueError(f"Unsupported BLOCK_K: {BLOCK_K}")

    row_idx = (
        gl.arange(0, BLOCK_M // 16, gl.SliceLayout(1, out_idx_layout))[:, None] * 0
    )
    slice_idx = row_idx + source_col

    meta_reordered = meta_reshaped.gather(slice_idx, 1)

    e_smem.store(meta_reordered)
    tma.async_copy_shared_to_global(e_desc, (0, 0), e_smem)
    tma.store_wait(pendings=0)


def small_mma(A_pruned, E, INSTR_SHAPE_N, num_warps=4):
    a_pruned_layout = gl.NVMMASharedLayout.get_default_for(A_pruned.shape, gl.float16)
    e_layout = gl.NVMMASharedLayout.get_default_for(E.shape, gl.int16)
    a_compressed_layout = gl.NVMMASharedLayout.get_default_for(
        (A_pruned.shape[0], A_pruned.shape[1] // 2), gl.float16
    )

    a_pruned_desc = TensorDescriptor.from_tensor(
        A_pruned, A_pruned.shape, a_pruned_layout
    )
    e_desc = TensorDescriptor.from_tensor(E, E.shape, e_layout)

    small_mma_kernel[(1,)](
        a_pruned_desc,
        e_desc,
        a_compressed_layout,
        A_pruned.shape[0],
        A_pruned.shape[1],
        INSTR_SHAPE_N,
        num_warps=num_warps,
    )


if __name__ == "__main__":
    print("Benchmarking WGMMA")
    print("==================")
    torch.set_printoptions(threshold=10_000, linewidth=200, edgeitems=16)

    M, N, K = 64, 16, 256
    num_warps = 4
    A = torch.randn(M, K, device="cuda", dtype=torch.float16)
    B = torch.randn(K, N, device="cuda", dtype=torch.float16)
    C = torch.zeros((M, N), device="cuda", dtype=torch.float32)

    A_pruned = prune_2_4(A)
    A, E = compress_dense_to_sparse(A_pruned)
    E = E.view(M // 16, K)

    E_gpu = torch.empty_like(E, dtype=torch.int16)
    # print(E)
    print("INSTR_SHAPE_N time (us)")
    for INSTR_SHAPE_N in [16]:
        small_mma(A_pruned, E_gpu, INSTR_SHAPE_N, num_warps)
        D_ref = torch.matmul(A_pruned, B)

        # print(E_gpu)

        torch.testing.assert_close(E, E_gpu, rtol=1e-3, atol=1e-1)

        # fn = lambda: small_mma(A, B, C, D, INSTR_SHAPE_N, num_warps)
        # ms = triton.testing.do_bench(fn)
        # print(f"{INSTR_SHAPE_N:>13} {ms*1000:>9.2f}")
    print()
