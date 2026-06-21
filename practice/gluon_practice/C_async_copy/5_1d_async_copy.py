# test 1d async copy

import torch
import triton
import triton.language as tl
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia.ampere import async_copy as cp

# baseline gluon without async copy
@gluon.jit
def elementwise_add_kernel(
    x_ptr, y_ptr, z_ptr, N, BLOCK_SIZE: gl.constexpr, layout: gl.constexpr
):

    pid = gl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + gl.arange(0, BLOCK_SIZE, layout = layout)
    mask = offsets < N


    x = gl.load(x_ptr + offsets, mask = mask)
    y = gl.load(y_ptr + offsets, mask = mask)

    z = x + y

    gl.store(z_ptr + offsets, z, mask = mask)

# gluon with async copy
@gluon.jit
def elementwise_add_async_kernel(
    x_ptr, y_ptr, z_ptr, N, BLOCK_SIZE: gl.constexpr, layout: gl.constexpr
):

    pid = gl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + gl.arange(0, BLOCK_SIZE, layout = layout)
    mask = offsets < N

    # layout of shared memory
    smem_layout: gl.constexpr = gl.SwizzledSharedLayout(vec = 1, per_phase = 1, max_phase = 1, order=[0])
    x_smem = gl.allocate_shared_memory(gl.float32, [BLOCK_SIZE], layout = smem_layout)
    y_smem = gl.allocate_shared_memory(gl.float32, [BLOCK_SIZE], layout = smem_layout)

    cp.async_copy_global_to_shared(x_smem, x_ptr + offsets, mask = mask)
    cp.async_copy_global_to_shared(y_smem, y_ptr + offsets, mask = mask)
    cp.commit_group()

    cp.wait_group(0)

    x = x_smem.load(layout)
    y = y_smem.load(layout)

    z = x + y

    gl.store(z_ptr + offsets, z, mask = mask)

# launch kernels
def elementwise_add(x, y, z, N, layout):
    grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE']),)
    elementwise_add_kernel[grid](x, y, z, N, BLOCK_SIZE = 1024, layout = layout)

def elementwise_add_async(x, y, z, N, layout):
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(N, BLOCK_SIZE), )
    elementwise_add_async_kernel[grid](x, y, z, N, BLOCK_SIZE = 1024, layout = layout)

# benchmarking
def run_benchmark():
    print(f"{'Elements (N)':>15} | {'Standard Load (GB/s)':>25} | {'Async Copy (GB/s)':>25} | {'Percentage':>15}")
    print("-" * 94)

    layout = gl.BlockedLayout(
        size_per_thread = [16],
        threads_per_warp = [32],
        warps_per_cta = [4],
        order = [0]
    )
    
    # Test sizes from 2^20 (~1M) to 2^27 (~134M)
    for i in range(20, 28):
        N = 2**i
        x = torch.randn(N, device='cuda', dtype=torch.float32)
        y = torch.randn(N, device='cuda', dtype=torch.float32)
        z = torch.empty_like(x)
        
        # Helper to calculate GB/s based on 3 memory operations (Read X, Read Y, Write Z)
        def get_gbps(ms):
            return (3 * x.numel() * x.element_size()) / ms * 1e-6
        
        # Benchmark Baseline
        ms_base = triton.testing.do_bench(lambda: elementwise_add(x, y, z, N, layout))
        gbps_base = get_gbps(ms_base)
        
        # Benchmark Async Copy
        ms_async = triton.testing.do_bench(lambda: elementwise_add_async(x, y, z, N, layout))
        gbps_async = get_gbps(ms_async)
        
        print(f"{N:>15} | {gbps_base:>25.2f} | {gbps_async:>25.2f} | {((gbps_async-gbps_base)/gbps_base * 100):>15.2f}%")

if __name__ == '__main__':
    # N = 1024
    # x = torch.randn(N, device='cuda', dtype=torch.float32)
    # y = torch.randn(N, device='cuda', dtype=torch.float32)
    # z = torch.empty_like(x)
    # layout = gl.BlockedLayout(
    #     size_per_thread = [16],
    #     threads_per_warp = [32],
    #     warps_per_cta = [4],
    #     order = [0]
    # )
    
    # print("Testing Baseline Compilation...")
    # elementwise_add(x, y, z, N, layout)
    # print("Baseline Compiled Successfully!\n")
    
    # print("Testing Async Compilation...")
    # elementwise_add_async(x, y, z, N, layout)
    # print("Async Compiled Successfully!")
    run_benchmark()
