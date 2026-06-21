import torch
import triton
import importlib
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

from triton.experimental.gluon.nvidia.hopper import TensorDescriptor
from triton.experimental.gluon.language.nvidia.hopper import tma, mbarrier, fence_async_shared
from triton.experimental.gluon.language.nvidia.ampere import async_copy as cp

# 1d copy with tma
@gluon.jit
def copy_1d_tma_kernel(x_desc, y_desc, BLOCK: gl.constexpr):
    pid = gl.program_id(0)

    smem_layout: gl.constexpr = x_desc.layout
    smem = gl.allocate_shared_memory(x_desc.dtype, [BLOCK], smem_layout)

    # setup barrier
    bar = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
    mbarrier.init(bar, count=1)
    mbarrier.expect(bar, x_desc.block_type.nbytes)

    # load to share memory
    tma.async_copy_global_to_shared(x_desc, [pid*BLOCK], bar, smem)

    # wait for completion
    mbarrier.wait(bar, phase = 0)

    # kill barrier
    mbarrier.invalidate(bar)

    # actually copy data
    tma.async_copy_shared_to_global(y_desc, [pid * BLOCK], smem)

    # check if tma stores is finished
    tma.store_wait(pendings = 0)

def copy_1d_tma(x, y, BLOCK):
    # initialize layout
    block_shape = [BLOCK]
    layout = gl.NVMMASharedLayout.get_default_for(block_shape, gl.float32)

    # initialize tensor descriptor
    x_desc = TensorDescriptor.from_tensor(x, block_shape, layout)
    y_desc = TensorDescriptor.from_tensor(y, block_shape, layout)

    grid = (triton.cdiv(x.numel(), BLOCK), )
    copy_1d_tma_kernel[grid](x_desc, y_desc, BLOCK)

# ampere baseline
@gluon.jit
def copy_1d_async_kernel(
    x_ptr, y_ptr, N, BLOCK_SIZE: gl.constexpr, layout: gl.constexpr
):

    pid = gl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + gl.arange(0, BLOCK_SIZE, layout = layout)
    mask = offsets < N

    # layout of shared memory
    smem_layout: gl.constexpr = gl.SwizzledSharedLayout(vec = 1, per_phase = 1, max_phase = 1, order=[0])
    x_smem = gl.allocate_shared_memory(gl.float32, [BLOCK_SIZE], layout = smem_layout)

    cp.async_copy_global_to_shared(x_smem, x_ptr + offsets, mask = mask)
    cp.commit_group()

    cp.wait_group(0)

    x = x_smem.load(layout)

    gl.store(y_ptr + offsets, x, mask = mask)

def copy_1d_async(x, y, N, layout):
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(N, BLOCK_SIZE), )
    copy_1d_async_kernel[grid](x, y, N, BLOCK_SIZE = 1024, layout = layout)

# benchmarking
def benchmark():
    N = 1024 * 1024 * 64  # 64M elements
    x = torch.randn(N, device='cuda', dtype=torch.float32)
    y_tma = torch.empty_like(x)
    y_async = torch.empty_like(x)

    BLOCK = 1024
    layout = gl.BlockedLayout(
        size_per_thread = [16],
        threads_per_warp = [32],
        warps_per_cta = [4],
        order = [0]
    )

    # 1. Check Correctness
    copy_1d_tma(x, y_tma, BLOCK)
    copy_1d_async(x, y_async, N, layout)

    print(f"TMA Correctness:   {torch.allclose(x, y_tma)}")
    print(f"Async Correctness: {torch.allclose(x, y_async)}")

    # 2. Benchmark
    # Function to calculate GB/s (1 read + 1 write)
    def gbps(ms):
        return (2 * x.numel() * x.element_size()) / (ms * 1e6)

    ms_tma = triton.testing.do_bench(lambda: copy_1d_tma(x, y_tma, BLOCK))
    ms_async = triton.testing.do_bench(lambda: copy_1d_async(x, y_async, N, layout))

    print("-" * 30)
    print(f"TMA Bandwidth:   {gbps(ms_tma):.2f} GB/s")
    print(f"Async Bandwidth: {gbps(ms_async):.2f} GB/s")


if __name__ == "__main__":
    benchmark()

