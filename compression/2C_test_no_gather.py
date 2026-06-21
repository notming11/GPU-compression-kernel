import torch
import triton
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

@gluon.jit
def compression_no_gather_kernel(
    a_pruned_desc, a_compressed_desc, meta_reordered_desc,
    BLOCK_M: gl.constexpr, BLOCK_K: gl.constexpr, 
    num_warps: gl.constexpr
):
    bar = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
    mbarrier.init(bar, count=1)

    a_pruned_smem = gl.allocate_shared_memory(a_pruned_desc.dtype, a_pruned_desc.block_type.shape, a_pruned_desc.layout)
    mbarrier.expect(bar, a_pruned_desc.block_type.nbytes)
    tma.async_copy_global_to_shared(a_pruned_desc, [0, 0], bar, a_pruned_smem)
    mbarrier.wait(bar, phase=0)
    mbarrier.invalidate(bar)

    m: gl.constexpr = 16
    k: gl.constexpr = 32
    n: gl.constexpr = 16

    warps_per_cta: gl.constexpr = [num_warps, 1]

    c_layout: gl.constexpr = gl.NVMMADistributedLayout(
        version=[3, 0],
        warps_per_cta=warps_per_cta,
        instr_shape=[m, n, k],
    )

    a_dot_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=0,
        parent=c_layout,
        k_width=32 // a_pruned_desc.dtype.primitive_bitwidth,
        meta=0,
    )

    a_pruned_val = a_pruned_smem.load(a_dot_layout)

    a_pruned_reg_layout: gl.constexpr = gl.BlockedLayout(
        [1, 16],
        [4, 8],
        [num_warps, 1],
        [1, 0]
    )
    a_pruned = gl.convert_layout(a_pruned_val, a_pruned_reg_layout)

    # --- Extract groups of 4 consecutive columns using reshape + split ---
    a_grouped = a_pruned.reshape(BLOCK_M, BLOCK_K // 4, 2, 2)

    a_even, a_odd = a_grouped.split()
    a_even = a_even.reshape(BLOCK_M, BLOCK_K // 4, 2)
    a_odd = a_odd.reshape(BLOCK_M, BLOCK_K // 4, 2)

    a0, a2 = a_even.split()  # a0 = col 4g+0, a2 = col 4g+2
    a1, a3 = a_odd.split()   # a1 = col 4g+1, a3 = col 4g+3

    a0 = a0.reshape(BLOCK_M, BLOCK_K // 4)
    a1 = a1.reshape(BLOCK_M, BLOCK_K // 4)
    a2 = a2.reshape(BLOCK_M, BLOCK_K // 4)
    a3 = a3.reshape(BLOCK_M, BLOCK_K // 4)

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

    a_compressed_smem = gl.allocate_shared_memory(a_compressed_desc.dtype, a_compressed_desc.block_type.shape, a_compressed_desc.layout)
    a_compressed_smem.store(a_compressed)
    fence_async_shared()
    tma.async_copy_shared_to_global(a_compressed_desc, [0, 0], a_compressed_smem)
    tma.store_wait(pendings=0)

    # 4-bit nibbles per group-of-4, shape (BLOCK_M, BLOCK_K//4)
    meta_4 = idx0 | (idx1 << 2)

    # Pack 4 consecutive nibbles using reshape + split (no gather)
    meta_grouped = meta_4.reshape(BLOCK_M, BLOCK_K // 16, 2, 2)

    meta_even, meta_odd = meta_grouped.split()
    meta_even = meta_even.reshape(BLOCK_M, BLOCK_K // 16, 2)
    meta_odd = meta_odd.reshape(BLOCK_M, BLOCK_K // 16, 2)

    mn0, mn2 = meta_even.split()
    mn1, mn3 = meta_odd.split()
    mn0 = mn0.reshape(BLOCK_M, BLOCK_K // 16)
    mn1 = mn1.reshape(BLOCK_M, BLOCK_K // 16)
    mn2 = mn2.reshape(BLOCK_M, BLOCK_K // 16)
    mn3 = mn3.reshape(BLOCK_M, BLOCK_K // 16)

    mn0 = mn0.to(gl.int16)
    mn1 = mn1.to(gl.int16)
    mn2 = mn2.to(gl.int16)
    mn3 = mn3.to(gl.int16)

    meta = mn0 | (mn1 << 4) | (mn2 << 8) | (mn3 << 12)

    meta_reshaped = meta.reshape((BLOCK_M // 16, BLOCK_K))
    meta_bits = meta_reshaped.reshape(BLOCK_M // 16, 2, 2, 2, 2, 2, 2, 2)
    meta_perm = meta_bits.permute(0, 5, 2, 3, 4, 6, 7, 1)
    meta_reordered = meta_perm.reshape(BLOCK_M // 16, BLOCK_K)

    meta_reordered_smem = gl.allocate_shared_memory(meta_reordered_desc.dtype, meta_reordered_desc.block_type.shape, meta_reordered_desc.layout)
    meta_reordered_smem.store(meta_reordered)
    fence_async_shared()
    tma.async_copy_shared_to_global(meta_reordered_desc, [0, 0], meta_reordered_smem)
    tma.store_wait(pendings=0)


def test_compression(A_pruned, A_compressed_out, E_out, num_warps=4):
    a_pruned_layout = gl.NVMMASharedLayout.get_default_for(A_pruned.shape, gl.float16)
    a_compressed_layout = gl.NVMMASharedLayout.get_default_for(A_compressed_out.shape, gl.float16)
    meta_reordered_layout = gl.NVMMASharedLayout.get_default_for(E_out.shape, gl.int16)

    a_pruned_desc = TensorDescriptor.from_tensor(A_pruned, A_pruned.shape, a_pruned_layout)
    a_compressed_desc = TensorDescriptor.from_tensor(A_compressed_out, A_compressed_out.shape, a_compressed_layout)
    meta_reordered_desc = TensorDescriptor.from_tensor(E_out, E_out.shape, meta_reordered_layout)

    compression_no_gather_kernel[(1, )](
        a_pruned_desc, a_compressed_desc, meta_reordered_desc,
        64, 128,
        num_warps=num_warps
    )


if __name__ == "__main__":
    os.environ["TRITON_ALWAYS_COMPILE"] = "1"
    print("Testing compression logic (no gather)")
    print("=====================================")

    M, K = 64, 128
    num_warps = 4
    A_dense = torch.randn(M, K, device="cuda", dtype=torch.float16)
    A_pruned = prune_2_4(A_dense)
    A_compressed_ref, E_ref = compress_dense_to_sparse(A_pruned)
    E_ref = E_ref.view(M // 16, K)

    A_compressed_out = torch.empty_like(A_compressed_ref)
    E_out = torch.empty_like(E_ref)

    test_compression(A_pruned, A_compressed_out, E_out, num_warps)

    print("Comparing compressed A...")
    try:
        torch.testing.assert_close(A_compressed_ref, A_compressed_out, rtol=1e-3, atol=1e-3)
        print("A_compressed matches!")
    except Exception as e:
        print("A_compressed mismatch:")
        print(e)
        mismatch_mask = (A_compressed_ref != A_compressed_out)
        mismatch_indices = mismatch_mask.nonzero()
        print(f"Total mismatched elements: {mismatch_mask.sum().item()} / {mismatch_mask.numel()}")
        print("First 10 mismatch indices and values:")
        for idx in mismatch_indices[:10]:
            r, c = idx[0].item(), idx[1].item()
            print(f"  At ({r}, {c}): Ref={A_compressed_ref[r, c].item()}, Out={A_compressed_out[r, c].item()}")

    print("Comparing reordered metadata...")
    try:
        torch.testing.assert_close(E_ref, E_out, rtol=0, atol=0)
        print("Metadata matches!")
    except Exception as e:
        print("Metadata mismatch:")
        print(e)
        mismatch_mask = (E_ref != E_out)
        mismatch_indices = mismatch_mask.nonzero()
        print(f"Total mismatched elements: {mismatch_mask.sum().item()} / {mismatch_mask.numel()}")
        print("First 10 mismatch indices and values:")
        for idx in mismatch_indices[:10]:
            r, c = idx[0].item(), idx[1].item()
            print(f"  At ({r}, {c}): Ref={E_ref[r, c].item()}, Out={E_out[r, c].item()}")

