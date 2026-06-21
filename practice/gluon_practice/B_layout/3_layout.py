# Simple vector scalling to learn 1d layout

import torch
import triton
import triton.language as tl
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

# Triton baseline
@triton.jit
def triton_scalling_kernel(x_ptr, y_ptr, a, num_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid*BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    tl.store(y_ptr + offsets, x * a, mask = mask)

def triton_scalling(x: torch.Tensor, a):
    output = torch.empty_like(x)
    grid = lambda meta: (triton.cdiv(x.numel(), meta['BLOCK_SIZE']), )
    triton_scalling_kernel[grid](x, output, a, x.numel(), BLOCK_SIZE=2048)
    return output

# Gluon 
@gluon.jit
def gluon_scalling_kernel(x_ptr, y_ptr, a, num_elements, BLOCK_SIZE: gl.constexpr, layout: gl.constexpr):
    pid = gl.program_id(0)

    start = pid * BLOCK_SIZE
    idx = gl.arange(0, BLOCK_SIZE, layout = layout)
    
    offsets = start + idx
    x_ptrs = x_ptr + offsets
    mask = offsets < num_elements

    x = gl.load(x_ptrs, mask = mask)
    y_ptrs = y_ptr + offsets
    gl.store(y_ptrs, x * a, mask = mask)

def gluon_scalling(x, a, R):
    layout = gl.BlockedLayout(
        size_per_thread = [R],
        threads_per_warp = [32],
        warps_per_cta = [4],
        order = [0]
    )

    output = torch.empty_like(x)
    BLOCK_SIZE = 2048
    grid = (triton.cdiv(x.numel(), BLOCK_SIZE),)

    gluon_scalling_kernel[grid](x, output, a, x.numel(), BLOCK_SIZE = BLOCK_SIZE, layout = layout)
    return output


# Benchmarking
def run_manual_benchmark():
    # Sweep sizes from 2^20 (~1M) to 2^25 (~33M) elements
    sizes = [2**i for i in range(20, 26)]
    a = 2.5
    quantiles = [0.5, 0.2, 0.8]
    
    # Print the table header
    print(f"\n{'N (Elements)':<12} | {'Torch GB/s':<12} | {'Triton GB/s':<12} | {'Gluon R=1':<12} | {'Gluon R=4':<12}")
    print("-" * 75)

    for N in sizes:
        # Allocate tensors
        x = torch.randn(N, device='cuda', dtype=torch.float32)
        
        # GB/s calculation: (Reads + Writes) / time
        calc_gbps = lambda ms: (2 * x.numel() * x.element_size()) / ms * 1e-6

        # 1. Benchmark PyTorch
        ms_torch, _, _ = triton.testing.do_bench(lambda: x * a, quantiles=quantiles)
        
        # 2. Benchmark Triton
        ms_triton, _, _ = triton.testing.do_bench(lambda: triton_scalling(x, a), quantiles=quantiles)
        
        # 3. Benchmark Gluon (R=1)
        ms_g1, _, _ = triton.testing.do_bench(lambda: gluon_scalling(x, a, R=1), quantiles=quantiles)
        
        # 4. Benchmark Gluon (R=4)
        ms_g4, _, _ = triton.testing.do_bench(lambda: gluon_scalling(x, a, R=4), quantiles=quantiles)

        # Print the row
        print(f"{N:<12} | {calc_gbps(ms_torch):<12.3f} | {calc_gbps(ms_triton):<12.3f} | {calc_gbps(ms_g1):<12.3f} | {calc_gbps(ms_g4):<12.3f}")
        
        # Free memory before the next size iteration
        del x
        torch.cuda.empty_cache()

if __name__ == "__main__":
    print("Warming up GPU and starting manual benchmark...")
    run_manual_benchmark()
    print("Benchmark complete.")