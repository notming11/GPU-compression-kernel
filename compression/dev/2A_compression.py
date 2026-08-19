# Given a pruned but uncompressed A. compress A during the runtime of the kernel
# Metadata is given, the kernel will not produce the metadata at this step, wgmma will not be run, i.e. we ignore the layout
# A is compressed by:
#   1. lowering A to smem
#   2. lower the splitted A into register
#   3. Split A into 4 with gl.split()
#   4. Find nonzero element
#   5. Stack nonzero element back to a tensor
#   6. load it back to smem and global memory

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
def compression_ignore_wgmma_kernel(
    a_pruned_desc, a_compressed_desc,
    BLOCK_M: gl.constexpr, BLOCK_K: gl.constexpr, 
    a_pruned_reg_layout: gl.constexpr, a_sliced_layout: gl.constexpr,
    INSTR_SHAPE_N: gl.constexpr, num_warps: gl.constexpr
):
    # 1. lower A to smem
    bar = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
    mbarrier.init(bar, count=1)

    a_pruned_smem = gl.allocate_shared_memory(a_pruned_desc.dtype, a_pruned_desc.block_type.shape, a_pruned_desc.layout)

    mbarrier.expect(bar, a_pruned_desc.block_type.nbytes)
    tma.async_copy_global_to_shared(a_pruned_desc, [0, 0], bar, a_pruned_smem)
    mbarrier.wait(bar, phase = 0)
    mbarrier.invalidate(bar)

    a_compressed_smem = gl.allocate_shared_memory(a_compressed_desc.dtype, a_compressed_desc.block_type.shape, a_compressed_desc.layout)

    # 2. split A into 4 and load them to register
    idx_layout: gl.constexpr = gl.BlockedLayout(
        [BLOCK_M // num_warps, 1],
        [1, 32],
        [num_warps, 1],
        [1, 0]
    )
    col_idx = gl.arange(0, BLOCK_K//4, gl.SliceLayout(0, idx_layout))[None, :]*4
    row_idx = gl.arange(0, BLOCK_M, gl.SliceLayout(1, idx_layout))[:, None]*0
    slice_idx = row_idx + col_idx

    a_pruned = a_pruned_smem.load(a_pruned_reg_layout)

    a_0 = a_pruned.gather(slice_idx, 1)
    a_1 = a_pruned.gather(slice_idx+1, 1)
    a_2 = a_pruned.gather(slice_idx+2, 1)
    a_3 = a_pruned.gather(slice_idx+3, 1)

    nz0 = gl.where(a_0 != 0, a_0, gl.where(a_1 != 0, a_1, a_2))
    nz1 = gl.where(a_3 != 0, a_3, gl.where(a_2 != 0, a_2, a_1))

    a_compressed = gl.join(nz0, nz1)
    a_compressed = a_compressed.reshape(BLOCK_M, BLOCK_K // 2)
    a_compressed_smem.store(a_compressed)
    tma.async_copy_shared_to_global(a_compressed_desc, [0, 0], a_compressed_smem)
    tma.store_wait(pendings = 0)


def compression_ignore_wgmma(A_pruned, A_compressed, A_0, A_1, A_2, A_3, INSTR_SHAPE_N, num_warps=4):

    a_pruned_layout = gl.NVMMASharedLayout.get_default_for(A_pruned.shape, gl.float16)
    a_compressed_layout = gl.NVMMASharedLayout.get_default_for(A_compressed.shape, gl.float16)
    a_sliced_layout = gl.NVMMASharedLayout.get_default_for(A_0.shape, gl.float16)

    a_pruned_reg_layout = gl.BlockedLayout(
        [64 // num_warps, 1],
        [1, 32],
        [num_warps, 1],
        [1, 0]
    )

    a_pruned_desc = TensorDescriptor.from_tensor(A_pruned, A_pruned.shape, a_pruned_layout)
    a_compressed_desc = TensorDescriptor.from_tensor(A_compressed, A_compressed.shape, a_compressed_layout)

    compression_ignore_wgmma_kernel[(1, )](
        a_pruned_desc, a_compressed_desc,
        64, 128, 
        a_pruned_reg_layout, a_sliced_layout,
        INSTR_SHAPE_N, num_warps=num_warps)


if __name__ == "__main__":
    # print("Benchmarking WGMMA")
    # print("==================")
    torch.set_printoptions(threshold=10_000, linewidth=200, edgeitems=16)
    M, N, K = 64, 16, 128
    num_warps = 4
    A = torch.randn(M, K, device="cuda", dtype=torch.float16)
    B = torch.randn(K, N, device="cuda", dtype=torch.float16)
    C = torch.zeros((M, N), device="cuda", dtype=torch.float32)

    A_pruned = prune_2_4(A)
    A, E = compress_dense_to_sparse(A_pruned)
    E = E.view(M // 16, K)

    D = torch.empty_like(C, dtype=torch.float16)

    A_compressed = torch.empty_like(A, dtype = torch.float16)

    # print("INSTR_SHAPE_N time (us)")
    for INSTR_SHAPE_N in [16]:
        # small_mma(A, B, C, E, D, INSTR_SHAPE_N, num_warps)
        # D_ref = torch.matmul(A_pruned, B)

        # torch.testing.assert_close(D_ref, D, rtol=1e-3, atol=1e-1)

        compression_ignore_wgmma(A_pruned, A_compressed, INSTR_SHAPE_N, num_warps)

        # fn = lambda: small_mma(A, B, C, D, INSTR_SHAPE_N, num_warps)
        # ms = triton.testing.do_bench(fn)
        # print(f"{INSTR_SHAPE_N:>13} {ms*1000:>9.2f}")

        torch.testing.assert_close(A_compressed, A, rtol = 0, atol = 0)
    print()


    