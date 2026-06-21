
import os
import importlib
import torch
import triton

# Crucial: Disable constant recompilation so do_bench measures GPU execution, not compiler overhead
os.environ["TRITON_ALWAYS_COMPILE"] = "0"

# Dynamically import small_mma from 2B_test.py to safely handle the leading digit in the filename
try:
    kernel_module = importlib.import_module("2B_test")
    small_mma = kernel_module.small_mma
except ImportError:
    raise ImportError(
        "Could not import 'small_mma' from '2B_test.py'. "
        "Please confirm that '2B_test.py' is present in your current directory."
    )

from prune import prune_2_4
from compress_2_4 import compress_dense_to_sparse

def benchmark_fixed_shape():
    # Explicit dimensions requested
    M, N, K = 64, 16, 128
    num_warps = 4
    INSTR_SHAPE_N = 16

    print(f"Initializing arrays for dimensions: M={M}, N={N}, K={K}...\n")

    # 1. Allocate Tensors
    A = torch.randn(M, K, device="cuda", dtype=torch.float16)
    B = torch.randn(K, N, device="cuda", dtype=torch.float16)
    C = torch.zeros((M, N), device="cuda", dtype=torch.float32)
    D = torch.empty((M, N), device="cuda", dtype=torch.float16)

    # 2. Structure 2:4 Sparsity and generate metadata matrix E
    A_pruned = prune_2_4(A)
    _, E = compress_dense_to_sparse(A_pruned)
    E = E.view(M // 16, K)

    # 3. Validation and Warmup Pass
    small_mma(A_pruned, B, C, E, D, INSTR_SHAPE_N, num_warps)
    D_ref = torch.matmul(A_pruned, B)
    
    try:
        torch.testing.assert_close(D_ref, D, rtol=1e-3, atol=1e-1)
        print("✓ Validation PASSED: Custom Triton inline compression matches PyTorch result.")
    except AssertionError as e:
        print("✗ Validation FAILED: Discrepancy detected between Triton kernel output and PyTorch reference!")
        print(e)
        return

    # 4. Benchmark PyTorch Matmul
    fn_pytorch = lambda: torch.matmul(A_pruned, B)
    ms_pytorch = triton.testing.do_bench(fn_pytorch)

    # 5. Benchmark your Inline-Compressing WGMMA Triton Kernel
    fn_triton = lambda: small_mma(A_pruned, B, C, E, D, INSTR_SHAPE_N, num_warps)
    ms_triton = triton.testing.do_bench(fn_triton)

    # 6. Calculate Dense-Equivalent Throughput (TFLOPS)
    ops = 2 * M * N * K
    dense_tflops = (ops / (ms_triton / 1000.0)) / 1e12

    print("\n" + "="*65)
    print(f"{'Target Framework/Kernel':<32} | {'Latency / Throughput':<25}")
    print("="*65)
    print(f"{'PyTorch matmul (Dense Baseline)':<32} | {ms_pytorch:.4f} ms")
    print(f"{'Triton Runtime Sparse WGMMA':<32} | {ms_triton:.4f} ms")
    print(f"{'Dense-Equivalent Performance':<32} | {dense_tflops:.2f} TFLOPS")
    print("="*65)

if __name__ == "__main__":
    benchmark_fixed_shape()