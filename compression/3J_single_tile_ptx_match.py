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
)

LAYOUT_TRANSPOSE_PTX = """
{
    .reg .pred %p1, %p2;
    .reg .b32 %lane, %t1, %t2;
    mov.u32 %lane, %laneid;

    // Calculate predicates for exchange
    and.b32 %t1, %lane, 1;
    setp.ne.b32 %p1, %t1, 0;

    and.b32 %t2, %lane, 2;
    setp.ne.b32 %p2, %t2, 0;

    .reg .b32 %r0_0, %r1_0, %r2_0, %r3_0;
    .reg .b32 %r0_8, %r1_8, %r2_8, %r3_8;

    // ================= ROW 0 =================
    // Step 1: Exchange distance 1 (XOR by 1)
    .reg .b32 %s1_0, %s3_0, %e1_0, %e3_0;
    selp.b32 %s1_0, $8, $9, %p1;
    selp.b32 %s3_0, $10, $11, %p1;
    shfl.sync.bfly.b32 %e1_0, %s1_0, 1, 0x1f, 0xffffffff;
    shfl.sync.bfly.b32 %e3_0, %s3_0, 1, 0x1f, 0xffffffff;

    @%p1  mov.b32 %r0_0, %e1_0;
    @!%p1 mov.b32 %r1_0, %e1_0;
    @%p1  mov.b32 %r2_0, %e3_0;
    @!%p1 mov.b32 %r3_0, %e3_0;

    @!%p1 mov.b32 %r0_0, $8;
    @%p1  mov.b32 %r1_0, $9;
    @!%p1 mov.b32 %r2_0, $10;
    @%p1  mov.b32 %r3_0, $11;

    // Step 2: Exchange distance 2 (XOR by 2)
    .reg .b32 %s2_0, %s3_new_0, %e2_0, %e3_new_0;
    selp.b32 %s2_0, %r0_0, %r2_0, %p2;
    selp.b32 %s3_new_0, %r1_0, %r3_0, %p2;
    shfl.sync.bfly.b32 %e2_0, %s2_0, 2, 0x1f, 0xffffffff;
    shfl.sync.bfly.b32 %e3_new_0, %s3_new_0, 2, 0x1f, 0xffffffff;

    @%p2  mov.b32 %r0_0, %e2_0;
    @!%p2 mov.b32 %r2_0, %e2_0;
    @%p2  mov.b32 %r1_0, %e3_new_0;
    @!%p2 mov.b32 %r3_0, %e3_new_0;

    // ================= ROW 8 =================
    // Step 1: Exchange distance 1 (XOR by 1)
    .reg .b32 %s1_8, %s3_8, %e1_8, %e3_8;
    selp.b32 %s1_8, $12, $13, %p1;
    selp.b32 %s3_8, $14, $15, %p1;
    shfl.sync.bfly.b32 %e1_8, %s1_8, 1, 0x1f, 0xffffffff;
    shfl.sync.bfly.b32 %e3_8, %s3_8, 1, 0x1f, 0xffffffff;

    @%p1  mov.b32 %r0_8, %e1_8;
    @!%p1 mov.b32 %r1_8, %e1_8;
    @%p1  mov.b32 %r2_8, %e3_8;
    @!%p1 mov.b32 %r3_8, %e3_8;

    @!%p1 mov.b32 %r0_8, $12;
    @%p1  mov.b32 %r1_8, $13;
    @!%p1 mov.b32 %r2_8, $14;
    @%p1  mov.b32 %r3_8, $15;

    // Step 2: Exchange distance 2 (XOR by 2)
    .reg .b32 %s2_8, %s3_new_8, %e2_8, %e3_new_8;
    selp.b32 %s2_8, %r0_8, %r2_8, %p2;
    selp.b32 %s3_new_8, %r1_8, %r3_8, %p2;
    shfl.sync.bfly.b32 %e2_8, %s2_8, 2, 0x1f, 0xffffffff;
    shfl.sync.bfly.b32 %e3_new_8, %s3_new_8, 2, 0x1f, 0xffffffff;

    @%p2  mov.b32 %r0_8, %e2_8;
    @!%p2 mov.b32 %r2_8, %e2_8;
    @%p2  mov.b32 %r1_8, %e3_new_8;
    @!%p2 mov.b32 %r3_8, %e3_new_8;

    // ================= REASSEMBLE WGMMA LAYOUT =================
    mov.b32 $0, %r0_0;
    mov.b32 $1, %r0_8;
    mov.b32 $2, %r1_0;
    mov.b32 $3, %r1_8;
    mov.b32 $4, %r2_0;
    mov.b32 $5, %r2_8;
    mov.b32 $6, %r3_0;
    mov.b32 $7, %r3_8;
}
"""


@gluon.jit
def pt_transpose_verify_kernel(
    a_pruned_desc,
    a_comp_out_desc,  # We will write the compressed matrix here
    BLOCK_M: gl.constexpr,
    BLOCK_K: gl.constexpr,
    INSTR_SHAPE_N: gl.constexpr,
    num_warps: gl.constexpr,
    TRANSPOSE_PTX: gl.constexpr,
):

    bar = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
    mbarrier.init(bar, count=1)

    # Load A tile
    a_pruned_smem = gl.allocate_shared_memory(
        a_pruned_desc.dtype, a_pruned_desc.block_type.shape, a_pruned_desc.layout
    )

    mbarrier.expect(bar, a_pruned_desc.block_type.nbytes)
    tma.async_copy_global_to_shared(a_pruned_desc, [0, 0], bar, a_pruned_smem)

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

    a_compressed_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=0,
        parent=c_layout,
        k_width=32 // a_pruned_desc.dtype.primitive_bitwidth,
        meta=0,
    )

    # gl.static_print("WGMMA:")
    # gl.static_print(a_compressed_layout.format_tensor_view([BLOCK_M, BLOCK_K // 2]))

    a_warp_bases: gl.constexpr = (
        [[16, 0], [32, 0]]
        if num_warps == 4
        else (
            [[16, 0], [32, 0], [0, 0]]
            if num_warps == 8
            else [[16, 0], [32, 0], [0, 0], [0, 0]]
        )
    )
    a_shape: gl.constexpr = [64, 64]
    a_pruned_reg_layout: gl.constexpr = gl.DistributedLinearLayout(
        reg_bases=[[0, 1], [0, 2], [0, 4], [0, 8], [8, 0]],
        lane_bases=[[0, 16], [0, 32], [1, 0], [2, 0], [4, 0]],
        warp_bases=a_warp_bases,
        block_bases=[],
        shape=a_shape,
    )

    a_pruned = a_pruned_smem.load(a_compressed_layout)
    a_pruned = gl.convert_layout(a_pruned, a_pruned_reg_layout)

    # --- Extract groups of 4 consecutive columns using reshape + split ---
    a_grouped = a_pruned.reshape(BLOCK_M, BLOCK_K // 4, 2, 2)
    a_even, a_odd = a_grouped.split()
    a0, a2 = a_even.split()
    a1, a3 = a_odd.split()

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

    # convert layout with ptx
    # =====================================================================
    # 1. Safely pack pairs of float16 into single int32 registers for PTX
    a_int16 = a_compressed.to(gl.int16, bitcast=True)
    a_pairs = a_int16.reshape(BLOCK_M, BLOCK_K // 4, 2)
    p0, p1 = a_pairs.split()

    # Masking prevents sign-extension when upcasting negative floats
    p0_32 = p0.to(gl.int32) & 0xFFFF
    p1_32 = p1.to(gl.int32) & 0xFFFF
    a_int32 = p0_32 | (p1_32 << 16)

    # 2. EXECUTE PTX
    (y_int32_reassembled,) = gl.inline_asm_elementwise(
        TRANSPOSE_PTX,
        "=r,=r,=r,=r,=r,=r,=r,=r,r,r,r,r,r,r,r,r",
        [a_int32],
        dtype=(gl.int32,),
        is_pure=True,
        pack=8,
    )

    # 5. Unpack back to int16 pairs and bitcast to float16
    y_p0 = (y_int32_reassembled & 0xFFFF).to(gl.int16)
    y_p1 = ((y_int32_reassembled >> 16) & 0xFFFF).to(gl.int16)
    y_pairs = gl.join(y_p0, y_p1)
    y_reshaped = y_pairs.reshape(BLOCK_M, BLOCK_K // 2)

    a_compressed_swapped = y_reshaped.to(gl.float16, bitcast=True)

    gl.static_print("PTX:")
    gl.static_print(
        a_compressed_swapped.type.layout.format_tensor_view([BLOCK_M, BLOCK_K // 2])
    )
    # =====================================================================

    # Write the transposed data back to Global Memory to verify
    a_comp_out_smem = gl.allocate_shared_memory(
        a_comp_out_desc.dtype, a_comp_out_desc.block_type.shape, a_comp_out_desc.layout
    )

    # Triton will reverse-shuffle the DotOperandLayout automatically
    # when storing it back into standard Shared Memory layout!
    a_comp_out_smem.store(a_compressed_swapped)

    fence_async_shared()
    tma.async_copy_shared_to_global(a_comp_out_desc, [0, 0], a_comp_out_smem)
    tma.store_wait(pendings=0)


def small_mma_test(A_pruned, A_comp_out, INSTR_SHAPE_N, num_warps=4):
    a_pruned_layout = gl.NVMMASharedLayout.get_default_for(A_pruned.shape, gl.float16)
    a_comp_out_layout = gl.NVMMASharedLayout.get_default_for(
        A_comp_out.shape, gl.float16
    )

    a_pruned_desc = TensorDescriptor.from_tensor(
        A_pruned, A_pruned.shape, a_pruned_layout
    )
    a_comp_out_desc = TensorDescriptor.from_tensor(
        A_comp_out, A_comp_out.shape, a_comp_out_layout
    )

    pt_transpose_verify_kernel[(1,)](
        a_pruned_desc,
        a_comp_out_desc,
        A_pruned.shape[0],
        A_pruned.shape[1],
        INSTR_SHAPE_N,
        num_warps=num_warps,
        TRANSPOSE_PTX=LAYOUT_TRANSPOSE_PTX,
    )


if __name__ == "__main__":
    os.environ["MLIR_ENABLE_DUMP"] = "1"
    os.environ["MLIR_DUMP_PATH"] = "./MLIR_DUMP/3J"
    os.environ["TRITON_ALWAYS_COMPILE"] = "1"

    torch.set_printoptions(threshold=10_000, linewidth=20000, edgeitems=16)

    print("Testing PTX Transpose correctness")
    print("=================================")

    M, N, K = 64, 16, 64
    num_warps = 8
    A = torch.randn(M, K, device="cuda", dtype=torch.float16)

    # Generate pruned input
    A_pruned = prune_2_4(A)

    # Get PyTorch reference compressed matrix
    A_comp_ref, _ = compress_dense_to_sparse(A_pruned)

    # Target tensor for Triton Kernel
    A_comp_out = torch.empty((M, K // 2), device="cuda", dtype=torch.float16)

    for INSTR_SHAPE_N in [16]:
        small_mma_test(A_pruned, A_comp_out, INSTR_SHAPE_N, num_warps)

        print("Kernel Output Sample:")
        print(A_comp_out)

        m: gl.constexpr = 16
        k: gl.constexpr = 32
        n: gl.constexpr = INSTR_SHAPE_N
        warps_per_cta: gl.constexpr = [num_warps, 1]

        c_layout: gl.constexpr = gl.NVMMADistributedLayout(
            version=[3, 0],
            warps_per_cta=warps_per_cta,
            instr_shape=[m, n, k],
        )

        a_compressed_layout: gl.constexpr = gl.DotOperandLayout(
            operand_index=0,
            parent=c_layout,
            k_width=32 // gl.float16.primitive_bitwidth,
            meta=0,
        )

        print("WGMMA:")
        print(a_compressed_layout.format_tensor_view([M, K // 2]))

        print("\nPyTorch Reference Sample:")
        print(A_comp_ref)

        try:
            torch.testing.assert_close(A_comp_ref, A_comp_out, rtol=1e-3, atol=1e-3)
            print("\n✅ SUCCESS: PTX Transpose matches PyTorch Reference exactly!")
        except Exception as e:
            print(f"\n❌ FAILED: Values do not match.\n{e}")
