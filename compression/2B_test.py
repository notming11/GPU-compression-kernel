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
    a_pruned_desc, b_desc, c_desc, e_desc, d_desc,  #
    a_compressed_layout: gl.constexpr,
    BLOCK_M: gl.constexpr, BLOCK_K: gl.constexpr, 
    INSTR_SHAPE_N: gl.constexpr, num_warps: gl.constexpr):

    bar = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
    mbarrier.init(bar, count=1)

    # Load A, B, C and E tiles.
    a_pruned_smem = gl.allocate_shared_memory(a_pruned_desc.dtype, a_pruned_desc.block_type.shape, a_pruned_desc.layout)
    b_smem = gl.allocate_shared_memory(b_desc.dtype, b_desc.block_type.shape, b_desc.layout)
    e_smem = gl.allocate_shared_memory(e_desc.dtype, e_desc.block_type.shape, e_desc.layout)
    c_smem = gl.allocate_shared_memory(c_desc.dtype, c_desc.block_type.shape, c_desc.layout)

    mbarrier.expect(bar, a_pruned_desc.block_type.nbytes + b_desc.block_type.nbytes + e_desc.block_type.nbytes + c_desc.block_type.nbytes)
    tma.async_copy_global_to_shared(a_pruned_desc, [0, 0], bar, a_pruned_smem)
    tma.async_copy_global_to_shared(b_desc, [0, 0], bar, b_smem)
    tma.async_copy_global_to_shared(e_desc, [0, 0], bar, e_smem)
    tma.async_copy_global_to_shared(c_desc, [0, 0], bar, c_smem)
    mbarrier.wait(bar, phase=0)
    mbarrier.invalidate(bar)

    ############################
    # new code for compression #
    ############################
    compress_shape: gl.constexpr = (a_pruned_desc.block_type.shape[0], a_pruned_desc.block_type.shape[1]//2)
    a_compressed_smem = gl.allocate_shared_memory(a_pruned_desc.dtype, compress_shape, a_compressed_layout)

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

    a_0 = a_pruned.gather(slice_idx, 1)
    a_1 = a_pruned.gather(slice_idx+1, 1)
    a_2 = a_pruned.gather(slice_idx+2, 1)
    a_3 = a_pruned.gather(slice_idx+3, 1)

    nz0 = gl.where(a_0 != 0, a_0, gl.where(a_1 != 0, a_1, a_2))
    nz1 = gl.where(a_3 != 0, a_3, gl.where(a_2 != 0, a_2, a_1))

    a_compressed = gl.join(nz0, nz1)
    a_compressed = a_compressed.reshape(BLOCK_M, BLOCK_K // 2)
    a_compressed_smem.store(a_compressed)
    ######################################################################

    m: gl.constexpr = 16
    k: gl.constexpr = 32
    n: gl.constexpr = INSTR_SHAPE_N
    warps_per_cta: gl.constexpr = [num_warps, 1]

    c_layout: gl.constexpr = gl.NVMMADistributedLayout(
        version=[3, 0],
        warps_per_cta=warps_per_cta,
        instr_shape=[m, n, k],
    )

    e_reg_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=0,
        parent=c_layout,
        k_width=32 // e_desc.dtype.primitive_bitwidth,
        meta=1,
    )

    e = e_smem.load(e_reg_layout)
    c = c_smem.load(c_layout)

    d = warpgroup_mma(a_compressed_smem, b_smem, c, e=e, is_async=True, use_acc=True)
    d = warpgroup_mma_wait(num_outstanding=0, deps=(d, ))

    d_smem = gl.allocate_shared_memory(d_desc.dtype, d_desc.block_type.shape, d_desc.layout)
    d_smem.store(d.to(gl.float16))
    fence_async_shared()
    tma.async_copy_shared_to_global(d_desc, [0, 0], d_smem)
    tma.store_wait(pendings=0)

def small_mma(A_pruned, B, C, E, D, INSTR_SHAPE_N, num_warps=4):
    a_pruned_layout = gl.NVMMASharedLayout.get_default_for(A_pruned.shape, gl.float16)
    b_layout = gl.NVMMASharedLayout.get_default_for(B.shape, gl.float16)
    e_layout = gl.NVMMASharedLayout.get_default_for(E.shape, gl.int16)
    c_layout = gl.NVMMASharedLayout.get_default_for(C.shape, gl.float32)
    d_layout = gl.NVMMASharedLayout.get_default_for(C.shape, gl.float16)
    a_compressed_layout = gl.NVMMASharedLayout.get_default_for((A_pruned.shape[0], A_pruned.shape[1] // 2), gl.float16)

    a_pruned_desc = TensorDescriptor.from_tensor(A_pruned, A_pruned.shape, a_pruned_layout)
    b_desc = TensorDescriptor.from_tensor(B, B.shape, b_layout)
    c_desc = TensorDescriptor.from_tensor(C, C.shape, c_layout)
    e_desc = TensorDescriptor.from_tensor(E, E.shape, e_layout)
    d_desc = TensorDescriptor.from_tensor(D, D.shape, d_layout)

    small_mma_kernel[(1, )](
        a_pruned_desc, b_desc, c_desc, e_desc, d_desc,  #
        a_compressed_layout,
        64, 64,
        INSTR_SHAPE_N, num_warps=num_warps)

if __name__ == "__main__":
    # os.environ["MLIR_ENABLE_DUMP"]="1"
    # os.environ["MLIR_DUMP_PATH"] = "./MLIR_DUMP"
    # os.environ["TRITON_ALWAYS_COMPILE"]="1"
    print("Benchmarking WGMMA")
    print("==================")
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
        small_mma(A_pruned, B, C, E, D, INSTR_SHAPE_N, num_warps)
        D_ref = torch.matmul(A_pruned, B)

        torch.testing.assert_close(D_ref, D, rtol=1e-3, atol=1e-1)

        # fn = lambda: small_mma(A, B, C, D, INSTR_SHAPE_N, num_warps)
        # ms = triton.testing.do_bench(fn)
        # print(f"{INSTR_SHAPE_N:>13} {ms*1000:>9.2f}")
    print()