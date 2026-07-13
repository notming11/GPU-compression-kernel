import torch
import os
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.nvidia.hopper import TensorDescriptor
from triton.experimental.gluon.language.nvidia.hopper import (
    tma,
    mbarrier,
    fence_async_shared,
)

os.environ["MLIR_ENABLE_DUMP"] = "1"
os.environ["MLIR_DUMP_PATH"] = "./MLIR_DUMP/test"
os.environ["TRITON_ALWAYS_COMPILE"] = "1"

num_warps = 4

m: gl.constexpr = 16
k: gl.constexpr = 32
n: gl.constexpr = 16

warps_per_cta: gl.constexpr = [num_warps, 1]

c_layout: gl.constexpr = gl.NVMMADistributedLayout(
    version=[3, 0],
    warps_per_cta=warps_per_cta,
    instr_shape=[m, n, k],
)

e_warp_bases: gl.constexpr = (
    [[1, 0], [2, 0]]
    if num_warps == 4
    else (
        [[1, 0], [2, 0], [0, 0]] if num_warps == 8 else [[1, 0], [2, 0], [0, 0], [0, 0]]
    )
)
e_intermediate_layout: gl.constexpr = gl.DistributedLinearLayout(
    reg_bases=[[0, 1], [0, 2], [0, 64], [4, 0]],
    lane_bases=[[0, 0], [0, 4], [0, 8], [0, 16], [0, 32]],
    warp_bases=e_warp_bases,
    block_bases=[],
    shape=[8, 128],
)

e_begin_layout: gl.constexpr = gl.DistributedLinearLayout(
    reg_bases=[[0, 1], [0, 2], [0, 4]],
    lane_bases=[[0, 0], [0, 0], [0, 8], [0, 16], [0, 32]],
    warp_bases=[[1, 0], [2, 0]],
    block_bases=[],
    shape=[4, 64],
)

e_end_layout: gl.constexpr = gl.DistributedLinearLayout(
    reg_bases=[[0, 1], [0, 2], [0, 4]],
    lane_bases=[[0, 0], [0, 0], [0, 8], [0, 16], [0, 32]],
    warp_bases=[[1, 0], [2, 0]],
    block_bases=[],
    shape=[4, 64],
)

BLOCK_M = 64
BLOCK_K = 64
block_shape = [BLOCK_M // 16, BLOCK_K]  # [4, 64]

smem_layout = gl.NVMMASharedLayout.get_default_for(block_shape, gl.int16)


@gluon.jit
def test_kernel(a_desc, c_desc, BLOCK_M: gl.constexpr, BLOCK_K: gl.constexpr):
    # Allocate shared memory for input and output tiles
    a_smem = gl.allocate_shared_memory(gl.int16, a_desc.block_type.shape, a_desc.layout)
    c_smem = gl.allocate_shared_memory(gl.int16, c_desc.block_type.shape, c_desc.layout)

    # Allocate and initialize mbarrier
    bars = gl.allocate_shared_memory(gl.int64, [1, 1], mbarrier.MBarrierLayout())
    bar = bars.index(0)
    mbarrier.init(bar, count=1)

    # TMA load: global -> shared
    mbarrier.expect(bar, a_desc.block_type.nbytes)
    tma.async_copy_global_to_shared(a_desc, [0, 0], bar, a_smem)
    mbarrier.wait(bar, 0)

    # Load from shared memory to registers with e_begin_layout
    e = a_smem.load(e_begin_layout)

    # Layout conversions to observe in MLIR
    e = gl.convert_layout(e, e_intermediate_layout)
    e = gl.convert_layout(e, e_end_layout)

    # Store result to shared memory, then TMA store: shared -> global
    c_smem.store(e)
    fence_async_shared()
    tma.async_copy_shared_to_global(c_desc, [0, 0], c_smem)
    tma.store_wait(pendings=0)


a_tensor = torch.randn(block_shape[0], block_shape[1], device="cuda", dtype=torch.float16).to(torch.int16)
c_tensor = torch.zeros(block_shape[0], block_shape[1], device="cuda", dtype=torch.int16)

a_desc = TensorDescriptor.from_tensor(a_tensor, block_shape, smem_layout)
c_desc = TensorDescriptor.from_tensor(c_tensor, block_shape, smem_layout)

test_kernel[(1,)](a_desc, c_desc, BLOCK_M, BLOCK_K)
print("done")
