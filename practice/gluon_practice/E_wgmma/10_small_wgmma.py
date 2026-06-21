import torch
import triton
import itertools
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

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
def small_wgmma_kernel(a_desc, b_desc, c_desc, d_desc, INSTR_SHAPE_N: gl.constexpr, num_warps: gl.constexpr):
    # load data into s memory
    bar = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
    mbarrier.init(bar, count = 1)

    a_smem = gl.allocate_shared_memory(a_desc.dtype, a_desc.block_type.shape, a_desc.layout)
    b_smem = gl.allocate_shared_memory(b_desc.dtype, b_desc.block_type.shape, b_desc.layout)
    c_smem = gl.allocate_shared_memory(c_desc.dtype, c_desc.block_type.shape, c_desc.layout)

    mbarrier.expect(bar, a_desc.block_type.nbytes + b_desc.block_type.nbytes + c_desc.block_type.nbytes)
    tma.async_copy_global_to_shared(a_desc, [0, 0], bar, a_smem)
    tma.async_copy_global_to_shared(b_desc, [0, 0], bar, b_smem)
    tma.async_copy_global_to_shared(c_desc, [0, 0], bar, c_smem)
    mbarrier.wait(bar, phase = 0)
    mbarrier.invalidate(bar)

    # initilize wgmma
    m: gl.constexpr = 16
    n: gl.constexpr = INSTR_SHAPE_N
    k: gl.constexpr = 256 // a_desc.dtype.primitive_bitwidth
    warps_per_cta: gl.constexpr = [num_warps, 1]

    # layout of c in register (must load c into register)
    c_layout: gl.constexpr = gl.NVMMADistributedLayout(
        version = [3, 0],
        warps_per_cta = warps_per_cta,
        instr_shape = [m, n, k]
    )
    c = c_smem.load(c_layout)

    # layout of a in register (optional to load a into register)
    a_reg_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index = 0,
        parent = c_layout,
        k_width = 32 // a_desc.dtype.primitive_bitwidth
    )

    a = a_smem.load(a_reg_layout)
    # a can also be in shared memory
    # a = a_smem

    # b must be in shared memory
    # use wgmma
    d = warpgroup_mma(a, b_smem, c, is_async = True, use_acc = True)
    d = warpgroup_mma_wait(num_outstanding = 0, deps = (d, ))

    # store d
    d_smem = gl.allocate_shared_memory(d_desc.dtype, d_desc.block_type.shape, d_desc.layout)
    d_smem.store(d)
    fence_async_shared()        # prevent race condition when tma write d_smem to d_desc
    tma.async_copy_shared_to_global(d_desc, [0, 0], d_smem)
    tma.store_wait(pendings = 0)

def small_wgmma(A, B, C, D, INSTR_SHAPE_N, num_warps=4):
    a_layout = gl.NVMMASharedLayout.get_default_for(A.shape, gl.float16)
    b_layout = gl.NVMMASharedLayout.get_default_for(B.shape, gl.float16)
    cd_layout = gl.NVMMASharedLayout.get_default_for(C.shape, gl.float32)

    a_desc = TensorDescriptor.from_tensor(A, A.shape, a_layout)
    b_desc = TensorDescriptor.from_tensor(B, B.shape, b_layout)
    c_desc = TensorDescriptor.from_tensor(C, C.shape, cd_layout)
    d_desc = TensorDescriptor.from_tensor(D, D.shape, cd_layout)

    small_wgmma_kernel[(1, )](
        a_desc, b_desc, c_desc, d_desc,  #
        INSTR_SHAPE_N, num_warps=num_warps)

def test_small_wgmma(M, N, K, INSTR_SHAPE_N, num_warps):
    maxN = max(N // triton.cdiv(num_warps, triton.cdiv(M, 16)), 8)
    if INSTR_SHAPE_N > maxN:
        pytest.skip(f"INSTR_SHAPE_N={INSTR_SHAPE_N} is too large for M={M}, N={N}, num_warps={num_warps}")

    A = torch.randn(M, K, device="cuda", dtype=torch.float16)
    B = torch.randn(K, N, device="cuda", dtype=torch.float16)
    C = torch.randn(M, N, device="cuda", dtype=torch.float32)
    D = torch.empty_like(C)
    small_wgmma(A, B, C, D, INSTR_SHAPE_N, num_warps)

if __name__ == "__main__":
    print("Benchmarking WGMMA")
    print("==================")
    M, N, K = 64, 128, 128
    num_warps = 4
    A = torch.randn(M, K, device="cuda", dtype=torch.float16)
    B = torch.randn(K, N, device="cuda", dtype=torch.float16)
    C = torch.randn(M, N, device="cuda", dtype=torch.float32)
    D = torch.empty_like(C)

    print("INSTR_SHAPE_N time (us)")
    for INSTR_SHAPE_N in [16, 32, 64, 128]:
        fn = lambda: small_wgmma(A, B, C, D, INSTR_SHAPE_N, num_warps)
        ms = triton.testing.do_bench(fn)
        print(f"{INSTR_SHAPE_N:>13} {ms*1000:>9.2f}")
    print()