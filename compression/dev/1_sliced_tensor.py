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
def compression_ignore_wgmma_kernel(
    a_pruned_desc, a_0_desc, a_1_desc, a_2_desc, a_3_desc,
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

    a_0_smem = gl.allocate_shared_memory(a_0_desc.dtype, a_0_desc.block_type.shape, a_sliced_layout)
    a_1_smem = gl.allocate_shared_memory(a_1_desc.dtype, a_1_desc.block_type.shape, a_sliced_layout)
    a_2_smem = gl.allocate_shared_memory(a_2_desc.dtype, a_0_desc.block_type.shape, a_sliced_layout)
    a_3_smem = gl.allocate_shared_memory(a_3_desc.dtype, a_0_desc.block_type.shape, a_sliced_layout)

    # 2. split A into 4 and load them to register
    idx_layout: gl.constexpr = gl.BlockedLayout(
        [1, 4],
        [4, 8],
        [num_warps, 1],
        [1, 0]
    )
    col_idx = gl.arange(0, BLOCK_K//4, gl.SliceLayout(0, idx_layout))[None, :]*4
    row_idx = gl.arange(0, BLOCK_M, gl.SliceLayout(1, idx_layout))[:, None]*0
    slice_idx = row_idx + col_idx

    # slice_idx = gl.convert_layout(slice_idx, idx_layout)

    # print(slice_idx.type)

    a_pruned = a_pruned_smem.load(a_pruned_reg_layout)

    a_0_reg = a_pruned.gather(slice_idx, 1)
    a_1_reg = a_pruned.gather(slice_idx+1, 1)
    a_2_reg = a_pruned.gather(slice_idx+2, 1)
    a_3_reg = a_pruned.gather(slice_idx+3, 1)

    a_0_smem.store(a_0_reg)
    a_1_smem.store(a_1_reg)
    a_2_smem.store(a_2_reg)
    a_3_smem.store(a_3_reg)

    fence_async_shared()

    tma.async_copy_shared_to_global(a_0_desc, [0, 0], a_0_smem)
    tma.async_copy_shared_to_global(a_1_desc, [0, 0], a_1_smem)
    tma.async_copy_shared_to_global(a_2_desc, [0, 0], a_2_smem)
    tma.async_copy_shared_to_global(a_3_desc, [0, 0], a_3_smem)
    tma.store_wait(pendings = 0)


def compression_ignore_wgmma(A_pruned, A_compressed, A_0, A_1, A_2, A_3, INSTR_SHAPE_N, num_warps=4):

    a_pruned_layout = gl.NVMMASharedLayout.get_default_for(A_pruned.shape, gl.float16)
    a_compressed_layout = gl.NVMMASharedLayout.get_default_for(A_compressed.shape, gl.float16)
    a_sliced_layout = gl.NVMMASharedLayout.get_default_for(A_0.shape, gl.float16)

    a_pruned_reg_layout = gl.BlockedLayout(     # 1 thread stores 4 elements in the pruned A matrix. 
        [64 // num_warps, 1],                                 # [1, 4] is used to capture a row of 4 elements
        [1, 32],                                # since the matrix is row-majored
        [num_warps, 1],
        [1, 0]
    )

    a_pruned_desc = TensorDescriptor.from_tensor(A_pruned, A_pruned.shape, a_pruned_layout)
    a_compressed_desc = TensorDescriptor.from_tensor(A_compressed, A_compressed.shape, a_compressed_layout)
    a_0_desc = TensorDescriptor.from_tensor(A_0, A_0.shape, a_sliced_layout)
    a_1_desc = TensorDescriptor.from_tensor(A_1, A_1.shape, a_sliced_layout)
    a_2_desc = TensorDescriptor.from_tensor(A_2, A_2.shape, a_sliced_layout)
    a_3_desc = TensorDescriptor.from_tensor(A_3, A_3.shape, a_sliced_layout)

    compression_ignore_wgmma_kernel[(1, )](
        a_pruned_desc, a_0_desc, a_1_desc, a_2_desc, a_3_desc,
        64, 128, 
        a_pruned_reg_layout, a_sliced_layout,
        INSTR_SHAPE_N, num_warps=num_warps)


if __name__ == "__main__":
    os.environ["MLIR_ENABLE_DUMP"]="1"
    os.environ["MLIR_DUMP_PATH"] = "./MLIR_DUMP"
    os.environ["TRITON_ALWAYS_COMPILE"]="1"
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
    A_0 = torch.zeros((M, K//4), dtype = torch.float16, device = "cuda")
    A_1 = torch.zeros((M, K//4), dtype = torch.float16, device = "cuda")
    A_2 = torch.zeros((M, K//4), dtype = torch.float16, device = "cuda")
    A_3 = torch.zeros((M, K//4), dtype = torch.float16, device = "cuda")

    # print("Original A_pruned (First Row, Cols 0-15):")
    # print(A_pruned[0, 0:16])

    # print("INSTR_SHAPE_N time (us)")
    for INSTR_SHAPE_N in [16]:
        # small_mma(A, B, C, E, D, INSTR_SHAPE_N, num_warps)
        # D_ref = torch.matmul(A_pruned, B)

        # torch.testing.assert_close(D_ref, D, rtol=1e-3, atol=1e-1)

        compression_ignore_wgmma(A_pruned, A_compressed, A_0, A_1, A_2, A_3, INSTR_SHAPE_N, num_warps)

        # fn = lambda: small_mma(A, B, C, D, INSTR_SHAPE_N, num_warps)
        # ms = triton.testing.do_bench(fn)
        # print(f"{INSTR_SHAPE_N:>13} {ms*1000:>9.2f}")

        # Print the first row, columns 0 through 15 of the original matrix

        # Print the first row of A_sliced (which should match columns 0, 4, 8, 12 of the original)
        # print("\nA_sliced (First Row, first 4 extracted elements):")
        # print(A_0[0, 0:4])
        # print(A_1[0, 0:4])
        # print(A_2[0, 0:4])
        # print(A_3[0, 0:4])
    print()


    