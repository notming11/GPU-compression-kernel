# test pipelining

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

# helper function for pipelining
@gluon.jit
def write_smem(write_idx, 
    x_smem, x_ptrs, offsets_M, 
    M, N, xstride_N, 
    BLOCK_N: gl.constexpr, num_buf: gl.constexpr, layout: gl.constexpr
):
    offsets_N = write_idx * BLOCK_N + gl.arange(0, BLOCK_N, layout = gl.SliceLayout(0, layout))
    mask = (offsets_M < M)[:, None] & (offsets_N < N)[None, :]
    cp.async_copy_global_to_shared(x_smem.index(write_idx % num_buf), x_ptrs + xstride_N * offsets_N[None, :], mask)
    cp.commit_group()

@gluon.jit
def read_smem(read_idx, 
    x_smem, y_ptrs, offsets_M, 
    M, N, ystride_N, 
    BLOCK_N: gl.constexpr, num_buf: gl.constexpr, layout: gl.constexpr
):
    x = x_smem.index(read_idx % num_buf).load(layout)
    offsets_N = read_idx * BLOCK_N + gl.arange(0, BLOCK_N, layout = gl.SliceLayout(0, layout))
    mask = (offsets_M < M)[:, None] & (offsets_N < N)[None, :]
    gl.store(y_ptrs + ystride_N * offsets_N[None, :], x, mask)

# async 2d copy with pipelining
@gluon.jit
def pipeline_async_copy_2d_kernel(
    x_ptr, y_ptr, 
    M, N,
    xstride_M, xstride_N,
    ystride_M, ystride_N,
    BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr,
    layout: gl.constexpr, smem_layout: gl.constexpr, num_buf: gl.constexpr
):

    pid = gl.program_id(0)
    offsets_M = pid * BLOCK_M + gl.arange(0, BLOCK_M, layout = gl.SliceLayout(1, layout))

    x_ptrs = x_ptr + xstride_M * offsets_M[:, None]
    y_ptrs = y_ptr + ystride_M * offsets_M[:, None]

    x_smem = gl.allocate_shared_memory(gl.float32, [num_buf, BLOCK_M, BLOCK_N], layout = smem_layout)

    write_idx = 0
    read_idx = 0

    for _ in gl.static_range(num_buf-1):
        write_smem(write_idx, x_smem, x_ptrs, offsets_M, M, N, xstride_N, BLOCK_N, num_buf, layout)
        write_idx += 1

    for _ in range(gl.cdiv(N, BLOCK_N) - (num_buf - 1)):
        write_smem(write_idx, x_smem, x_ptrs, offsets_M, M, N, xstride_N, BLOCK_N, num_buf, layout)
        write_idx += 1

        cp.wait_group(num_buf - 1)

        read_smem(read_idx, x_smem, y_ptrs, offsets_M, M, N, ystride_N, BLOCK_N, num_buf, layout)
        read_idx += 1

    for i in gl.static_range(num_buf - 1):
        cp.wait_group(num_buf - 2 - i)
        read_smem(read_idx, x_smem, y_ptrs, offsets_M, M, N, ystride_N, BLOCK_N, num_buf, layout)
        read_idx += 1

def pipeline_async_copy_2d(x, y, layout, smem_layout, BLOCK_M, BLOCK_N, num_buf):
    M, N = x.shape
    grid = (triton.cdiv(M, BLOCK_M),)
    pipeline_async_copy_2d_kernel[grid](
        x, y,
        M, N,
        *x.stride(),
        *y.stride(),
        BLOCK_M, BLOCK_N,
        layout, smem_layout, num_buf
    )



# Benchmarking
def run_benchmark():
    print(f"{'Elements (MxN)':>15} | {'Standard Load (GB/s)':>25} | {'Async Copy (GB/s)':>25} | {'Pipeline Copy (GB/s)':>25}")
    print("-" * 100)

    BLOCK_M = 64
    BLOCK_N = 64
    num_buf = 2
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

        # Benchmark Pipeline
        ms_pipeline = triton.testing.do_bench(lambda: pipeline_async_copy_2d(x, y, layout, smem_layout, BLOCK_M, BLOCK_N, num_buf))
        gbps_pipeline = get_gbps(ms_pipeline)

        if not torch.equal(x, y):
            print("FAILED")
        
        print(f"{N:>15} | {gbps_base:>25.2f} | {gbps_async:>25.2f} | {gbps_pipeline:>25.2f}")

if __name__ == "__main__":
    run_benchmark()