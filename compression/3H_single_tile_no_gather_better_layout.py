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

    m: gl.constexpr = 16
    k: gl.constexpr = 32
    n: gl.constexpr = INSTR_SHAPE_N

    warps_per_cta: gl.constexpr = [num_warps, 1]

    c_layout: gl.constexpr = gl.NVMMADistributedLayout(
        version=[3, 0],
        warps_per_cta=warps_per_cta,
        instr_shape=[m, n, k],
    )

    #########################################
    # new code for compression and metadata #
    # --- NO gl.gather() version ---        #
    #########################################

    a_compressed_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=0,
        parent=c_layout,
        k_width=32 // a_pruned_desc.dtype.primitive_bitwidth,
        meta=0,
    )
    # gl.static_print(a_compressed_layout.format_tensor_view([BLOCK_M, BLOCK_K]))
    # gl.static_print(gl.to_linear_layout(a_compressed_layout, [BLOCK_M, BLOCK_K // 4]))

    e_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index = 0,
        parent=c_layout,
        k_width=32 // gl.int16.primitive_bitwidth,
        meta = 1
    )
    # gl.static_print(e_layout.format_tensor_view([BLOCK_M//16, BLOCK_K]))

    # a_pruned_reg_layout: gl.constexpr = gl.DistributedLinearLayout(
    #     reg_bases=[(0, 1), (0, 2), (0, 8), (0, 16), (0, 32), (0, 64)], 
    #     lane_bases=[(8, 0), (0, 4), (1, 0), (2, 0), (4, 0)], 
    #     warp_bases=[(16, 0), (32, 0)], 
    #     block_bases=[], 
    #     shape=[64, 128]
    # )
    a_warp_bases: gl.constexpr = [[16, 0], [32, 0]] if num_warps == 4 else ([[16, 0], [32, 0], [0, 0]] if num_warps == 8 else [[16, 0], [32, 0], [0, 0], [0, 0]])
    a_shape: gl.constexpr = [64, 64]
    a_pruned_reg_layout: gl.constexpr = gl.DistributedLinearLayout(
        reg_bases=[[0, 1], [0, 2], [0, 4], [0, 8], [8, 0]], 
        lane_bases=[[0, 16], [0, 32], [1, 0], [2, 0], [4, 0]], 
        warp_bases=a_warp_bases, 
        block_bases=[], 
        shape=a_shape
    )
    gl.static_print(gl.to_linear_layout(a_pruned_reg_layout, [BLOCK_M, BLOCK_K]))
    a_pruned = a_pruned_smem.load(a_compressed_layout)
    a_pruned = gl.convert_layout(a_pruned, a_pruned_reg_layout)

    # --- Extract groups of 4 consecutive columns using reshape + split ---
    # a_pruned shape: (BLOCK_M, BLOCK_K) = (64, 128)
    # Reshape to (BLOCK_M, BLOCK_K//4, 2, 2) to decompose groups of 4
    # Element [m, g, j, i] = original column 4*g + 2*j + i
    a_grouped = a_pruned.reshape(BLOCK_M, BLOCK_K // 4, 2, 2)
    # gl.static_print(gl.to_linear_layout(a_grouped.type.layout, [BLOCK_M, BLOCK_K // 4, 2, 2]))

    # split last dim (size 2): lo[m,g,j] = [m,g,j,0], hi[m,g,j] = [m,g,j,1]
    # lo contains columns 4g+0 (j=0) and 4g+2 (j=1) -> a0, a2
    # hi contains columns 4g+1 (j=0) and 4g+3 (j=1) -> a1, a3
    a_even, a_odd = a_grouped.split()

    # split again to separate the pairs
    a0, a2 = a_even.split()  # a0 = col 4g+0, a2 = col 4g+2
    a1, a3 = a_odd.split()   # a1 = col 4g+1, a3 = col 4g+3

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

    # nz0 = gl.convert_layout(nz0, a_compressed_layout)
    # nz1 = gl.convert_layout(nz1, a_compressed_layout)

    a_compressed = gl.join(nz0, nz1)
    a_compressed = a_compressed.reshape(BLOCK_M, BLOCK_K // 2)
    gl.static_print("Before convert layout:")
    gl.static_print(gl.to_linear_layout(a_compressed.type.layout, [BLOCK_M, BLOCK_K // 2]))
    gl.static_print(a_compressed.type.layout.format_tensor_view([BLOCK_M, BLOCK_K // 2]))

    # gl.static_print(nz0.type)

    # 4-bit nibbles per group-of-4, shape (BLOCK_M, BLOCK_K//4)
    meta_4 = idx0 | (idx1 << 2)

    # --- Pack 4 consecutive nibbles using reshape + split (no gather) ---
    # meta_4 shape: (BLOCK_M, BLOCK_K//4) = (64, 32)
    # Reshape to (BLOCK_M, BLOCK_K//16, 2, 2) to group 4 consecutive nibbles
    # Element [m, g, j, i] = meta_4[m, 4*g + 2*j + i]
    meta_grouped = meta_4.reshape(BLOCK_M, BLOCK_K // 16, 2, 2)

    # split last dim: even = [m,g,j,0], odd = [m,g,j,1]
    meta_even, meta_odd = meta_grouped.split()

    mn0, mn2 = meta_even.split()  # mn0 = nibble 4g+0, mn2 = nibble 4g+2
    mn1, mn3 = meta_odd.split()   # mn1 = nibble 4g+1, mn3 = nibble 4g+3

    mn0 = mn0.to(gl.int16)
    mn1 = mn1.to(gl.int16)
    mn2 = mn2.to(gl.int16)
    mn3 = mn3.to(gl.int16)

    # Pack nibbles: matches compress_2_4.py's meta_n[:,:,0] | (meta_n[:,:,1]<<4) | ...
    meta = mn0 | (mn1 << 4) | (mn2 << 8) | (mn3 << 12)
    # meta shape: (BLOCK_M, BLOCK_K//16) = (64, 8)

    # --- Reorder metadata using reshape + permute (no gather) ---
    # The original code reshapes to (BLOCK_M//16, BLOCK_K) = (4, 128) then gathers
    # with a bit-swapped index pattern that permutes column bit positions:
    #   source[c6,c5,c4,c3,c2,c1,c0] -> dest[c0,c5,c4,c3,c6,c2,c1]
    #
    # We achieve this by decomposing 128 into individual bit dimensions,
    # permuting, and reshaping back.

    # meta_reshaped = meta.reshape((BLOCK_M // 16, BLOCK_K))

    # if BLOCK_K == 128:
    #     # Decompose 128 cols into 7 binary dimensions (MSB to LSB):
    #     # (4, 128) -> (4, 2_c6, 2_c5, 2_c4, 2_c3, 2_c2, 2_c1, 2_c0)
    #     meta_bits = meta_reshaped.reshape(BLOCK_M // 16, 2, 2, 2, 2, 2, 2, 2)

    #     # Permute to target bit ordering:
    #     # Source dims: (row=0, c6=1, c5=2, c4=3, c3=4, c2=5, c1=6, c0=7)
    #     # Target dims: (row=0, c0=7, c5=2, c4=3, c3=4, c6=1, c2=5, c1=6)
    #     meta_perm = meta_bits.permute(0, 5, 2, 3, 4, 6, 7, 1)
    # elif BLOCK_K == 64:
    #     # for block_k = 64
    #     meta_bits = meta_reshaped.reshape(BLOCK_M // 16, 2, 2, 2, 2, 2, 2)
    #     meta_perm = meta_bits.permute(0, 2, 3, 4, 5, 6, 1)
    # elif BLOCK_K == 256:
    #     # for block_k = 256
    #     meta_bits = meta_reshaped.reshape(BLOCK_M // 16, 2, 2, 2, 2, 2, 2, 2, 2)
    #     meta_perm = meta_bits.permute(0, 5, 6, 2, 3, 4, 7, 8, 1)

    # # Reshape back to (4, 128)
    # meta_reordered = meta_perm.reshape(BLOCK_M // 16, BLOCK_K)
    meta_reshaped = meta.reshape(BLOCK_M // 16, 2, 8, BLOCK_K // 64, 4)
    meta_reordered = meta_reshaped.permute(0, 3, 2, 4, 1).reshape(BLOCK_M // 16, BLOCK_K)
    #######################################################################################

    # convert a_compressed to DotOperandLayout

    a_compressed = gl.convert_layout(a_compressed, a_compressed_layout)
    gl.static_print("After convert layout:")
    gl.static_print(gl.to_linear_layout(a_compressed.type.layout, [BLOCK_M, BLOCK_K // 2]))
    gl.static_print(a_compressed.type.layout.format_tensor_view([BLOCK_M, BLOCK_K // 2]))
    e = gl.convert_layout(meta_reordered, e_layout, assert_trivial = False)
    c = c_smem.load(c_layout)

    d = warpgroup_mma(
        a_compressed, b_smem, c, e=e, is_async=True, use_acc=True
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
        A_pruned.shape[0],
        A_pruned.shape[1],
        INSTR_SHAPE_N,
        num_warps=num_warps,
    )


if __name__ == "__main__":
    os.environ["MLIR_ENABLE_DUMP"]="1"
    os.environ["MLIR_DUMP_PATH"] = "./MLIR_DUMP/3H"
    os.environ["TRITON_ALWAYS_COMPILE"]="1"
    print("Benchmarking WGMMA (no gather)")
    print("===============================")

    M, N, K = 64, 16, 64
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
