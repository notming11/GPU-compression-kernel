# try 2d layout
import torch
import triton
import triton.language as tl
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

# gluon code
@gluon.jit
def gluon_memcpy_2d_kernel(x_ptr, y_ptr, M, N, stride_row, stride_col, BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr, layout: gl.constexpr):
    pid_M = gl.program_id(0)
    pid_N = gl.program_id(1)

    start_M = pid_M * BLOCK_M
    start_N = pid_N * BLOCK_N
    
    row_idx = gl.arange(0, BLOCK_M, layout = gl.SliceLayout(dim=1, parent = layout)) + start_M
    col_idx = gl.arange(0, BLOCK_N, layout = gl.SliceLayout(dim=0, parent = layout)) + start_N

    mask_row = row_idx < M
    mask_col = col_idx < N

    offset = (row_idx[:, None] * stride_row) + (col_idx[None, :] * stride_col)
    mask = mask_row[:, None] & mask_col[None, :]

    x = gl.load(x_ptr + offset, mask = mask)
    gl.store(y_ptr + offset, x, mask = mask)

def run_gluon_2d(x, layout_type):
    y = torch.empty_like(x)
    M, N = x.shape
    BLOCK_M, BLOCK_N = 128, 128

    if layout_type == 'optimized':
        layout = gl.BlockedLayout(
            size_per_thread=[1, 4],
            threads_per_warp=[4, 8],
            warps_per_cta=[4, 1],
            order = [1, 0]
        )
    else:
        layout = gl.BlockedLayout(
            size_per_thread=[1, 4],
            threads_per_warp=[32, 1],
            warps_per_cta=[4, 1],
            order = [0, 1]
        )

    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    gluon_memcpy_2d_kernel[grid](x, y, M, N, x.stride(0), x.stride(1), BLOCK_M = BLOCK_M, BLOCK_N = BLOCK_N, layout = layout)
    return y
    
# triton baseline
@triton.jit
def triton_memcpy_2d_kernel(
    x_ptr, y_ptr, 
    M, N, 
    stride_row, stride_col, 
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    start_m = pid_m * BLOCK_M
    start_n = pid_n * BLOCK_N

    row_idx = start_m + tl.arange(0, BLOCK_M)
    col_idx = start_n + tl.arange(0, BLOCK_N)

    mask_row = row_idx < M
    mask_col = col_idx < N
    # Logical AND for 2D masking
    mask = mask_row[:, None] & mask_col[None, :]

    offsets = (row_idx[:, None] * stride_row) + (col_idx[None, :] * stride_col)

    x = tl.load(x_ptr + offsets, mask=mask)
    tl.store(y_ptr + offsets, x, mask=mask)

def run_triton_2d(x):
    y = torch.empty_like(x)
    M, N = x.shape
    BLOCK_M, BLOCK_N = 128, 128
    
    # 2D Grid calculation
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(N, meta['BLOCK_N']))
    triton_memcpy_2d_kernel[grid](x, y, M, N, x.stride(0), x.stride(1), BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N)
    return y

# benchmarks
def run_manual_benchmark():
    # Sweep sizes from 1024x1024 to 8192x8192
    sizes = [1024 * i for i in range(1, 9)]
    quantiles = [0.5, 0.2, 0.8]
    
    print(f"\n{'Matrix (M x M)':<15} | {'Torch GB/s':<12} | {'Triton GB/s':<12} | {'Gluon Opt GB/s':<15} | {'Gluon Naive GB/s'}")
    print("-" * 80)

    for size in sizes:
        M, N = size, size
        # PyTorch creates Row-Major tensors by default
        x = torch.randn(M, N, device='cuda', dtype=torch.float32)
        
        # GB/s calculation
        gbps = lambda ms: (2 * x.numel() * x.element_size()) / ms * 1e-6

        # Run benchmarks
        ms_torch, _, _ = triton.testing.do_bench(lambda: x.clone(), quantiles=quantiles)
        ms_triton, _, _ = triton.testing.do_bench(lambda: run_triton_2d(x), quantiles=quantiles)
        ms_g_opt, _, _ = triton.testing.do_bench(lambda: run_gluon_2d(x, 'optimized'), quantiles=quantiles)
        ms_g_naive, _, _ = triton.testing.do_bench(lambda: run_gluon_2d(x, 'naive'), quantiles=quantiles)

        # Print the row
        print(f"{f'{size}x{size}':<15} | {gbps(ms_torch):<12.3f} | {gbps(ms_triton):<12.3f} | {gbps(ms_g_opt):<15.3f} | {gbps(ms_g_naive):<15.3f}")
        
        # Prevent Out-Of-Memory errors on the GPU
        del x
        torch.cuda.empty_cache()

def run_contiguous_benchmark():
    # Sweep sizes from 1024x1024 to 8192x8192
    sizes = [1024 * i for i in range(1, 9)]
    quantiles = [0.5, 0.2, 0.8]
    
    print(f"\n{'Transposed (x.T)':<16} | {'Torch .contiguous':<18} | {'Triton Baseline':<18} | {'Gluon Zero-Copy'}")
    print("-" * 75)

    for size in sizes:
        M, N = size, size
        # Create standard tensor, then logically transpose it
        x = torch.randn(M, N, device='cuda', dtype=torch.float32)
        x_t = x.T 
        
        # GB/s calculation
        gbps = lambda ms: (2 * x.numel() * x.element_size()) / ms * 1e-6

        # 1. PyTorch Contiguous Tax
        # PyTorch allocates new VRAM and physically copies the data
        ms_torch, _, _ = triton.testing.do_bench(lambda: x_t.contiguous(), quantiles=quantiles)
        
        # 2. Triton Baseline
        # Triton tries to read the non-contiguous tensor directly and suffers terrible uncoalesced memory latency
        ms_triton, _, _ = triton.testing.do_bench(lambda: run_triton_2d(x_t), quantiles=quantiles)
        
        # 3. Gluon Zero-Copy
        # We pass the transposed tensor, but we tell Gluon to use the 'naive' order=[0, 1] layout.
        # Gluon dynamically adapts its thread mapping to read the physical memory perfectly contiguously!
        ms_g_zero_copy, _, _ = triton.testing.do_bench(lambda: run_gluon_2d(x_t, 'naive'), quantiles=quantiles)

        # Print the row
        print(f"{f'{size}x{size}':<16} | {gbps(ms_torch):<18.3f} | {gbps(ms_triton):<18.3f} | {gbps(ms_g_zero_copy):<18.3f}")
        
        # Prevent Out-Of-Memory errors on the GPU
        del x, x_t
        torch.cuda.empty_cache()

if __name__ == "__main__":
    print("Warming up GPU and starting 2D benchmark...")
    # run_manual_benchmark()
    run_contiguous_benchmark()
    print("Benchmark complete.")