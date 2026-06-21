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


@gluon.jit
def small_mma_kernel(
    a_pruned_desc,
    b_desc,
    c_desc,
    d_desc,  #
    BLOCK_M: gl.constexpr,
    BLOCK_K: gl.constexpr,
    INSTR_SHAPE_N: gl.constexpr,
    num_warps: gl.constexpr,
):

    bar = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
    mbarrier.init(bar, count=1)

    # Load A, B, C tiles.
    a_pruned_smem = gl.allocate_shared_memory(
        a_pruned_desc.dtype, a_pruned_desc.block_type.shape, a_pruned_desc.layout
    )
    b_smem = gl.allocate_shared_memory(
        b_desc.dtype, b_desc.block_type.shape, b_desc.layout
    )
    c_smem = gl.allocate_shared_memory(
        c_desc.dtype, c_desc.block_type.shape, c_desc.layout
    )
    mbarrier.expect(
        bar,
        a_pruned_desc.block_type.nbytes
        + b_desc.block_type.nbytes
        + c_desc.block_type.nbytes,
    )

    tma.async_copy_global_to_shared(a_pruned_desc, [0, 0], bar, a_pruned_smem)
    tma.async_copy_global_to_shared(b_desc, [0, 0], bar, b_smem)
    tma.async_copy_global_to_shared(c_desc, [0, 0], bar, c_smem)
    mbarrier.wait(bar, phase=0)
    mbarrier.invalidate(bar)

    #########################################
    # new code for compression and metadata #
    #########################################
    compress_shape: gl.constexpr = (
        a_pruned_desc.block_type.shape[0],
        a_pruned_desc.block_type.shape[1] // 2,
    )

    # pre shuffle Linear Layout so that after slicing and gathering the layout is equivalent to DotOperandLayout
    a_pruned_reg_layout: gl.constexpr = gl.BlockedLayout(
        [1, 4],
        [4, 8],
        [num_warps, 1],
        [1, 0]
    )

    col_idx = gl.arange(0, BLOCK_K//4, gl.SliceLayout(0, a_pruned_reg_layout))[None, :]*4
    row_idx = gl.arange(0, BLOCK_M, gl.SliceLayout(1, a_pruned_reg_layout))[:, None]*0
    slice_idx = row_idx + col_idx

    a_pruned = a_pruned_smem.load(a_pruned_reg_layout)

    a0 = a_pruned.gather(slice_idx, 1)
    a1 = a_pruned.gather(slice_idx + 1, 1)
    a2 = a_pruned.gather(slice_idx + 2, 1)
    a3 = a_pruned.gather(slice_idx + 3, 1)

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

    nz0 = gl.where(idx0 == 0, a0, gl.where(idx0 == 1, a1, gl.where(idx0 == 2, a2, a3)))
    nz1 = gl.where(idx1 == 0, a0, gl.where(idx1 == 1, a1, gl.where(idx1 == 2, a2, a3)))

    a_compressed = gl.join(nz0, nz1)
    a_compressed = a_compressed.reshape(BLOCK_M, BLOCK_K // 2)

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

    out_idx_layout: gl.constexpr = gl.DistributedLinearLayout(
        reg_bases=[[0, 1], [0, 64]],
        lane_bases=[[0, 2], [0, 4], [0, 8], [0, 16], [0, 32]],
        warp_bases=[[1, 0], [2, 0]],
        block_bases=[],
        shape=[4, 128],
    )

    c = gl.arange(0, BLOCK_K, gl.SliceLayout(0, out_idx_layout))[None, :]
    c0 = c & 1
    c1 = (c >> 1) & 1
    c2 = (c >> 2) & 1
    c345 = c & 0x38
    c6 = (c >> 6) & 1
    source_col = (c0 << 6) | c345 | (c6 << 2) | (c2 << 1) | c1

    row_idx = (
        gl.arange(0, BLOCK_M // 16, gl.SliceLayout(1, out_idx_layout))[:, None] * 0
    )

    slice_idx = row_idx + source_col
    meta_reordered = meta_reshaped.gather(slice_idx, 1)
    #######################################################################################

    m: gl.constexpr = 16
    k: gl.constexpr = 32
    n: gl.constexpr = INSTR_SHAPE_N

    warps_per_cta: gl.constexpr = [num_warps, 1]

    c_layout: gl.constexpr = gl.NVMMADistributedLayout(
        version=[3, 0],
        warps_per_cta=warps_per_cta,
        instr_shape=[m, n, k],
    )

    # convert a_compressed to DotOpreandLayout
    a_compressed_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=0,
        parent=c_layout,
        k_width=32 // a_pruned_desc.dtype.primitive_bitwidth,
        meta=0,
    )

    # Since the current layout of a_compressed is equivalent of a_compressed_layout
    # we can set assert_trivial=True so there there's no overhead
    a_compressed = gl.convert_layout(a_compressed, a_compressed_layout, assert_trivial = False)
    c = c_smem.load(c_layout)

    d = warpgroup_mma(
        a_compressed, b_smem, c, e=meta_reordered, is_async=True, use_acc=True
    )
    d = warpgroup_mma_wait(num_outstanding=0, deps=(d,))

    d_smem = gl.allocate_shared_memory(
        d_desc.dtype, d_desc.block_type.shape, d_desc.layout
    )
    d_smem.store(d.to(gl.float16))

    fence_async_shared()

    tma.async_copy_shared_to_global(d_desc, [0, 0], d_smem)
    tma.store_wait(pendings=0)


def small_mma(A_pruned, B, C, D, INSTR_SHAPE_N, num_warps=4):
    a_pruned_layout = gl.NVMMASharedLayout.get_default_for(A_pruned.shape, gl.float16)
    b_layout = gl.NVMMASharedLayout.get_default_for(B.shape, gl.float16)
    c_layout = gl.NVMMASharedLayout.get_default_for(C.shape, gl.float32)
    d_layout = gl.NVMMASharedLayout.get_default_for(C.shape, gl.float16)

    a_pruned_desc = TensorDescriptor.from_tensor(
        A_pruned, A_pruned.shape, a_pruned_layout
    )
    b_desc = TensorDescriptor.from_tensor(B, B.shape, b_layout)
    c_desc = TensorDescriptor.from_tensor(C, C.shape, c_layout)
    d_desc = TensorDescriptor.from_tensor(D, D.shape, d_layout)

    small_mma_kernel[(1,)](
        a_pruned_desc,
        b_desc,
        c_desc,
        d_desc,  #
        64,
        128,
        INSTR_SHAPE_N,
        num_warps=num_warps,
    )


if __name__ == "__main__":
    os.environ["MLIR_ENABLE_DUMP"]="1"
    os.environ["MLIR_DUMP_PATH"] = "./MLIR_DUMP/3C"
    os.environ["TRITON_ALWAYS_COMPILE"]="1"
    print("Benchmarking WGMMA")
    print("==================")

    M, N, K = 64, 16, 128
    num_warps = 4
    A = torch.randn(M, K, device="cuda", dtype=torch.float16)
    B = torch.randn(K, N, device="cuda", dtype=torch.float16)
    C = torch.zeros((M, N), device="cuda", dtype=torch.float32)

    A_pruned = prune_2_4(A)
    A, E = compress_dense_to_sparse(A_pruned)
    E = E.view(M // 16, K)

    D = torch.empty_like(C, dtype=torch.float16)

    print("INSTR_SHAPE_N time (us)")
    for INSTR_SHAPE_N in [16]:
        small_mma(A_pruned, B, C, D, INSTR_SHAPE_N, num_warps)
        D_ref = torch.matmul(A_pruned, B)

        # print(D_ref)
        # print(D)

        torch.testing.assert_close(D_ref, D, rtol=1e-3, atol=1e-1)

        # fn = lambda: small_mma(A, B, C, D, INSTR_SHAPE_N, num_warps)
        # ms = triton.testing.do_bench(fn)
        # print(f"{INSTR_SHAPE_N:>13} {ms*1000:>9.2f}")
    print()
