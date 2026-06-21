# test 2d async copy

import torch
import triton
import triton.language as tl
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia.ampere import async_copy as cp

# baseline 2d copy
@gluon.jit
def copy_2d_kernel(
    x_ptr, y_ptr, 
    M, N,
    xstride_M, xstride_N,
    ystride_M, ystride_N,
    BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr,
    layout: gl.constexpr
):

    pid = gl.program_id(0)
    offsets_M = pid*BLOCK_M + gl.arange(0, BLOCK_M, layout = gl.SliceLayout(1, layout))

    x_ptrs = x_ptr + xstride_M * offsets_M[:, None]
    y_ptrs = y_ptr + ystride_M * offsets_M[:, None]

    for i in range(0, N, BLOCK_N):
        offsets_N = i + gl.arange(0, BLOCK_N, layout = gl.SliceLayout(0, layout))
        mask = (offsets_M[:, None] < M) & (offsets_N[None, :] < N)

        x = gl.load(x_ptrs + xstride_N * offsets_N[None, :], mask = mask)

        gl.store(y_ptrs + ystride_N * offsets_N[None, :], x, mask = mask)

def copy_2d(x, y, layout, BLOCK_M, BLOCK_N):
    M, N = x.shape
    grid = (triton.cdiv(M, BLOCK_M),)
    copy_2d_kernel[grid](
        x, y,
        M, N,
        *x.stride(),
        *y.stride(),
        BLOCK_M, BLOCK_N,
        layout
    )

# async 2d copy
@gluon.jit
def async_copy_2d_kernel(
    x_ptr, y_ptr, 
    M, N,
    xstride_M, xstride_N,
    ystride_M, ystride_N,
    BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr,
    layout: gl.constexpr, smem_layout: gl.constexpr
):

    pid = gl.program_id(0)
    offsets_M = pid * BLOCK_M + gl.arange(0, BLOCK_M, layout = gl.SliceLayout(1, layout))

    x_ptrs = x_ptr + xstride_M * offsets_M[:, None]
    y_ptrs = y_ptr + ystride_M * offsets_M[:, None]

    x_smem = gl.allocate_shared_memory(gl.float32, [BLOCK_M, BLOCK_N], layout = smem_layout)

    for i in range(0, N, BLOCK_N):
        offsets_N = i + gl.arange(0, BLOCK_N, layout = gl.SliceLayout(0, layout))
        mask = (offsets_M[:, None] < M) & (offsets_N[None, :] < N)


        cp.async_copy_global_to_shared(x_smem, x_ptrs + xstride_N * offsets_N[None, :], mask = mask)
        cp.commit_group()
        cp.wait_group(0)

        x = x_smem.load(layout)

        gl.store(y_ptrs + ystride_N * offsets_N[None, :], x, mask = mask)

def async_copy_2d(x, y, layout, smem_layout, BLOCK_M, BLOCK_N):
    M, N = x.shape
    grid = (triton.cdiv(M, BLOCK_M),)
    async_copy_2d_kernel[grid](
        x, y,
        M, N,
        *x.stride(),
        *y.stride(),
        BLOCK_M, BLOCK_N,
        layout, smem_layout
    )

# Benchmarking
def run_benchmark():
    print(f"{'Elements (MxN)':>15} | {'Standard Load (GB/s)':>25} | {'Async Copy (GB/s)':>25} | {'Percentage':>15}")
    print("-" * 94)

    BLOCK_M = 64
    BLOCK_N = 64
    layout = gl.BlockedLayout(
        size_per_thread = [1, 1],
        threads_per_warp = [1, 32],
        warps_per_cta = [1, 4],
        order = [1, 0]
    )
    smem_layout = gl.SwizzledSharedLayout(
        vec = 1, per_phase = 1, max_phase = 1, order = [1, 0]
    )

    for i in range(10, 16):
        M = 2**i
        N = 2**i
        x = torch.randn(M, N, device='cuda', dtype=torch.float32)
        y = torch.randn(M, N, device='cuda', dtype=torch.float32)
        
        # Helper to calculate GB/s based on 3 memory operations (Read X, Read Y, Write Z)
        def get_gbps(ms):
            return (2 * x.numel() * x.element_size()) / ms * 1e-6
        
        # Benchmark Baseline
        ms_base = triton.testing.do_bench(lambda: copy_2d(x, y, layout, BLOCK_M, BLOCK_N))
        gbps_base = get_gbps(ms_base)
        
        # Benchmark Async Copy
        ms_async = triton.testing.do_bench(lambda: async_copy_2d(x, y, layout, smem_layout, BLOCK_M, BLOCK_N))
        gbps_async = get_gbps(ms_async)
        
        print(f"{N:>15} | {gbps_base:>25.2f} | {gbps_async:>25.2f} | {((gbps_async-gbps_base)/gbps_base * 100):>15.2f}%")

if __name__ == "__main__":
    run_benchmark()