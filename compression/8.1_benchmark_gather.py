import os
import importlib
import torch
import triton

os.environ["TRITON_ALWAYS_COMPILE"] = "0"

# 1. Dynamically import '1_sliced_tensor.py' to avoid standard Python syntax errors with leading digits
try:
    sliced_tensor_module = importlib.import_module("1_sliced_tensor")
    compression_ignore_wgmma = sliced_tensor_module.compression_ignore_wgmma
except ImportError:
    raise ImportError(
        "Could not import 'compression_ignore_wgmma' from '1_sliced_tensor.py'. "
        "Make sure '1_sliced_tensor.py' exists in your current working directory."
    )

# 2. Import your existing workspace routines
from prune import prune_2_4
from compress_2_4 import compress_dense_to_sparse


def run_compression_benchmark():
    # Force Triton to compile cleanly during benchmarking bounds
    os.environ["TRITON_ALWAYS_COMPILE"] = "1"
    
    # A list of diverse matrix configurations (M, K) to see how performance scales
    shapes = [
        (64, 128),
    ]
    
    num_warps = 4
    INSTR_SHAPE_N = 16
    
    print(f"{'Shape (M x K)':<18} | {'PyTorch Ref (ms)':<18} | {'Triton Kernel (ms)':<20} | {'Speedup':<10}")
    print("-" * 75)
    
    for M, K in shapes:
        # Generate random FP16 input matrix and apply 2:4 structured sparsity
        A = torch.randn(M, K, device="cuda", dtype=torch.float16)
        A_pruned = prune_2_4(A)
        
        # Pre-allocate output tensors to avoid overheads inside timing hooks
        # compress_dense_to_sparse returns a half-sized K matrix (M, K // 2)
        A_compressed = torch.zeros((M, K // 2), device="cuda", dtype=torch.float16)
        A_0 = torch.zeros((M, K // 4), device="cuda", dtype=torch.float16)
        A_1 = torch.zeros((M, K // 4), device="cuda", dtype=torch.float16)
        A_2 = torch.zeros((M, K // 4), device="cuda", dtype=torch.float16)
        A_3 = torch.zeros((M, K // 4), device="cuda", dtype=torch.float16)
        
        # --- Warmup executions ---
        _ = compress_dense_to_sparse(A_pruned)
        compression_ignore_wgmma(A_pruned, A_compressed, A_0, A_1, A_2, A_3, INSTR_SHAPE_N, num_warps)
        
        # --- Benchmark: PyTorch Reference Compression ---
        # triton.testing.do_bench takes care of GPU stream sync and warmups automatically
        fn_ref = lambda: compress_dense_to_sparse(A_pruned)
        ref_ms = triton.testing.do_bench(fn_ref)
        
        # --- Benchmark: Custom Triton Runtime Slicing Kernel ---
        fn_triton = lambda: compression_ignore_wgmma(
            A_pruned, A_compressed, A_0, A_1, A_2, A_3, INSTR_SHAPE_N, num_warps
        )
        triton_ms = triton.testing.do_bench(fn_triton)
        
        # Calculate performance speedup
        speedup = ref_ms / triton_ms if triton_ms > 0 else float('inf')
        
        print(f"{f'{M} x {K}':<18} | {ref_ms:<18.4f} | {triton_ms:<20.4f} | {speedup:.2f}x")


if __name__ == "__main__":
    print("====================================================")
    print("   Benchmarking Runtime Compression Overheads       ")
    print("====================================================")
    torch.set_printoptions(precision=4, sci_mode=False)
    run_compression_benchmark()
