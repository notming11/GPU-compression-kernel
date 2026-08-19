import os
import sys
import importlib.util
import torch
import triton
from prune import prune_2_4
from compress_2_4 import compress_dense_to_sparse


def run_pipeline_benchmark():
    print("Loading modules dynamically...")
    
    # --- Safe Import for 5_compression_loop.py ---
    try:
        v5 = importlib.import_module("5_compression_loop")
    except ImportError:
        raise ImportError("Could not find '5_compression_loop.py' in the current working directory.")
        
    # --- Force-Load 5.1 via its explicit filesystem path ---
    v5_1_path = "./5.1_compression_loop_with_convert.py"
    if not os.path.exists(v5_1_path):
        raise ImportError(f"Could not find physical file '{v5_1_path}' in the current working directory.")
        
    try:
        # Bypasses Python's dot-parsing rules by mapping the file pathway directly to a clean module alias
        spec = importlib.util.spec_from_file_location("compression_v5_1", v5_1_path)
        v5_1 = importlib.util.module_from_spec(spec)
        sys.modules["compression_v5_1"] = v5_1
        spec.loader.exec_module(v5_1)
    except Exception as e:
        raise ImportError(f"Failed to load '5.1_compression_loop_with_convert.py' directly: {e}")

    try:
        import gluon_loop as baseline
    except ImportError:
        raise ImportError("baseline fail to include")

    # Mirroring the precise workload variations defined in your test configurations
    test_configs = [
        # (M, N, K, BLOCK_M, BLOCK_N, BLOCK_K, TRANSPOSE_B, num_warps)
        (768, 768, 768, 64, 64, 128, False, 4),
        (768, 768, 768, 64, 64, 128, True, 4),
        (768, 768, 768, 64, 64, 128, False, 8),
        (768, 768, 768, 64, 64, 128, True, 8),
        (768, 768, 768, 128, 128, 128, False, 4),
        (768, 768, 768, 128, 128, 128, True, 4),
        (768, 768, 768, 128, 128, 128, False, 8),
        (768, 768, 768, 128, 128, 128, True, 8),
        (4096, 4096, 4096, 64, 64, 128, False, 4),
        (4096, 4096, 4096, 64, 64, 128, True, 4),
        (4096, 4096, 4096, 64, 64, 128, False, 8),
        (4096, 4096, 4096, 64, 64, 128, True, 8),
        (4096, 4096, 4096, 128, 128, 128, False, 4),
        (4096, 4096, 4096, 128, 128, 128, True, 4),
        (4096, 4096, 4096, 128, 128, 128, False, 8),
        (4096, 4096, 4096, 128, 128, 128, True, 8),
    ]

    print("\n" + "="*95)
    print(f"{'Config (M, N, K, BM, BN, BK, TB, W)':<40} | {'v5 Time (ms)':<13} | {'v5.1 Time (ms)':<15} | {'dense time':<15} | {"sparse time":<15}")
    print("="*95)

    for config in test_configs:
        M, N, K, BLOCK_M, BLOCK_N, BLOCK_K, TRANSPOSE_B, num_warps = config
        
        # Initialize dense inputs and generate static 2:4 structured sparsity pattern
        A = torch.randn(M, K, device="cuda", dtype=torch.float16)
        A_pruned = prune_2_4(A)
        A_compressed, E = compress_dense_to_sparse(A)
        E = E.view(M // 16, K)
        
        B = torch.randn((N, K) if TRANSPOSE_B else (K, N), device="cuda", dtype=torch.float16)
        
        # Separate output tracking targets for safe assertions
        C_v5 = torch.empty(M, N, device="cuda", dtype=torch.float16)
        C_v5_1 = torch.empty(M, N, device="cuda", dtype=torch.float16)
        C_dense = torch.empty(M, N, device="cuda", dtype=torch.float16)
        C_sparse = torch.empty(M, N, device="cuda", dtype=torch.float16)

        A_compressed_v5 = torch.empty((M, K // 2), device="cuda", dtype=torch.float16)
        E_v5 = torch.empty((M // 16, K), device="cuda", dtype=torch.int16)

        A_compressed_v5_1 = torch.empty((M, K // 2), device="cuda", dtype=torch.float16)
        E_v5_1 = torch.empty((M // 16, K), device="cuda", dtype=torch.int16)

        # Functional validation pass to guarantee profile integrity
        try:
            v5.sparse_compress_blocked_matmul(A_pruned, B, C_v5, A_compressed_v5, E_v5, BLOCK_M, BLOCK_N, BLOCK_K, TRANSPOSE_B, num_warps)
            v5_1.sparse_compress_blocked_matmul(A_pruned, B, C_v5_1, A_compressed_v5_1, E_v5_1, BLOCK_M, BLOCK_N, BLOCK_K, TRANSPOSE_B, num_warps)
            baseline.blocked_matmul(A, B, C_dense, BLOCK_M, BLOCK_N, BLOCK_K, TRANSPOSE_B, num_warps)
            baseline.sparse_blocked_matmul(A, E, B, C_sparse, BLOCK_M, BLOCK_N, BLOCK_K, TRANSPOSE_B, num_warps)
            
            # Ensure parity before taking speed measurements
            torch.testing.assert_close(C_v5, C_v5_1, rtol=1e-3, atol=1e-1)
        except Exception as e:
            print(f"Skipping config {config} due to execution runtime error: {e}")
            continue

        # Isolate profiling loops using lambda wrappers
        fn_v5 = lambda: v5.sparse_compress_blocked_matmul(A_pruned, B, C_v5, A_compressed_v5, E_v5, BLOCK_M, BLOCK_N, BLOCK_K, TRANSPOSE_B, num_warps)
        fn_v5_1 = lambda: v5_1.sparse_compress_blocked_matmul(A_pruned, B, C_v5_1, A_compressed_v5_1, E_v5_1, BLOCK_M, BLOCK_N, BLOCK_K, TRANSPOSE_B, num_warps)
        fn_dense = lambda: baseline.blocked_matmul(A, B, C_dense, BLOCK_M, BLOCK_N, BLOCK_K, TRANSPOSE_B, num_warps)
        fn_sparse = lambda: baseline.sparse_blocked_matmul(A, E, B, C_sparse, BLOCK_M, BLOCK_N, BLOCK_K, TRANSPOSE_B, num_warps)

        # Profile hardware execution durations
        ms_v5 = triton.testing.do_bench(fn_v5)
        ms_v5_1 = triton.testing.do_bench(fn_v5_1)
        ms_dense = triton.testing.do_bench(fn_dense)
        ms_sparse = triton.testing.do_bench(fn_sparse)
        
        # speedup = ms_v5 / ms_v5_1 if ms_v5_1 > 0 else 0.0
        
        config_str = f"({M},{N},{K},{BLOCK_M},{BLOCK_N},{BLOCK_K},{str(TRANSPOSE_B)[0]},{num_warps})"
        print(f"{config_str:<40} | {ms_v5:>13.4f} | {ms_v5_1:>15.4f} | {ms_dense:>15.4f} | {ms_sparse:>15.4f}")

    print("="*95)

if __name__ == "__main__":
    run_pipeline_benchmark()