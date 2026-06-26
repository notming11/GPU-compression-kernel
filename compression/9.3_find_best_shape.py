import importlib
import torch
import triton
import itertools
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 1. Choose a high-capacity path on your storage cluster
SCRATCH_WORKSPACE = "compiler_scratch"

# 2. Force create the workspace structures
os.makedirs(SCRATCH_WORKSPACE, exist_ok=True)
os.makedirs(os.path.join(SCRATCH_WORKSPACE, "triton_cache"), exist_ok=True)
os.makedirs(os.path.join(SCRATCH_WORKSPACE, "cuda_cache"), exist_ok=True)

# 3. OVERRIDE TRITON (Must happen before 'import triton')
os.environ["TRITON_CACHE_DIR"] = os.path.join(SCRATCH_WORKSPACE, "triton_cache")

# 4. OVERRIDE NVIDIA PTXAS TEMPORARY FILE DUMPS
os.environ["TMPDIR"] = SCRATCH_WORKSPACE
os.environ["TMP"] = SCRATCH_WORKSPACE
os.environ["TEMP"] = SCRATCH_WORKSPACE

# 5. OVERRIDE PYTORCH / CUDA JIT BINARY PAYLOAD CACHES
os.environ["CUDA_CACHE_PATH"] = os.path.join(SCRATCH_WORKSPACE, "cuda_cache")
os.environ["TORCH_HOME"] = os.path.join(SCRATCH_WORKSPACE, "cuda_cache")

from prune import prune_2_4
from compress_2_4 import compress_dense_to_sparse

def matmul_get_configs():
    return [
        triton.Config(
            {
                "BLOCK_SIZE_M": BM,
                "BLOCK_SIZE_N": BN,
                "BLOCK_SIZE_K": BK,
                "num_buffers": buffers,
            },
            num_warps=warps,
        )
        for BM, BN, BK in [[64, 64, 128], [64, 64, 256], [64, 128, 128], [128, 64, 128], [64, 64, 64], [64, 128, 64], [128, 128, 64]] 
        for buffers in (3, 4, 5, 6, 7)
        for warps in (4, 8, 16)
    ]

def benchmark_kernels(M, N, K, BLOCK_M, BLOCK_N, BLOCK_K, num_buffers, num_warps):
    A = torch.randn((M, K), device="cuda", dtype=torch.float16)
    B = torch.randn((K, N), device="cuda", dtype=torch.float16)
    C = torch.zeros((M, N), device="cuda", dtype=torch.float16)
    
    A_pruned = prune_2_4(A) 
    A_comp, E = compress_dense_to_sparse(A_pruned)
    
    def to_tflops(ms):
        return (2 * M * N * K) / (ms * 1e-3 * 1e12) if ms else 0.0

    ms_dense, ms_runtime, ms_precomp = None, None, None
    tflops_dense, tflops_runtime, tflops_precomp = None, None, None

    try:
        ms_dense = triton.testing.do_bench(
            lambda: dense_matmul(A, B, C, BLOCK_M, BLOCK_N, BLOCK_K, num_buffers, num_warps, PersistentTileScheduler)
        )
        tflops_dense = to_tflops(ms_dense)
    except Exception as e:
        # print(f"dense failed on {BLOCK_M}x{BLOCK_N}x{BLOCK_K}, w:{num_warps}, b:{num_buffers}. Error: {e}")
        pass

    try:
        ms_precomp = triton.testing.do_bench(
            lambda: pre_compressed_sparse_matmul(A_comp, E, B, C, BLOCK_M, BLOCK_N, BLOCK_K, num_buffers, num_warps, PersistentTileScheduler)
        )
        tflops_precomp = to_tflops(ms_precomp)
    except Exception as e:
        # print(f"Precomp failed on {BLOCK_M}x{BLOCK_N}x{BLOCK_K}, w:{num_warps}, b:{num_buffers}. Error: {e}")
        pass

    try:
        ms_runtime = triton.testing.do_bench(
            lambda: runtime_compression_sparse_matmul(A_pruned, B, C, BLOCK_M, BLOCK_N, BLOCK_K, num_buffers, num_warps, PersistentTileScheduler)
        )
        tflops_runtime = to_tflops(ms_runtime)
    except Exception as e:
        # print(f"runtime failed on {BLOCK_M}x{BLOCK_N}x{BLOCK_K}, w:{num_warps}, b:{num_buffers}. Error: {e}")
        pass

    if ms_runtime is not None and ms_precomp is not None and ms_precomp > 0:
        overhead = (ms_runtime / ms_precomp - 1.0) * 100.0
    else:
        overhead = float('nan')

    return {
        "dense_tflops": tflops_dense,
        "runtime_tflops": tflops_runtime,
        "precomp_tflops": tflops_precomp,
        "overhead_pct": overhead
    }

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Input the version (7/7.1/7.2/7.3/7.5) and N")
        sys.exit(1)
    version = sys.argv[1]
    N=int(sys.argv[2])
    # dim = [768, 1024, 1536, 2048, 3072, 4096, 6144, 8192, 12288, 16384, 24576, 32768, 49152]
    # shapes = [(i, N, j) for i in dim for j in dim if [i, j] not in [[49152, 32768], [32768, 49152]]]
    shapes = [(16384, N, 49152)]

    paths = {
        "7" : "./7_compression_pipeline.py", 
        "7.1" : "./7.1_compression_pipeline_with_convert.py", 
        "7.2" : "./7.2_compression_pipeline_no_gather.py", 
        "7.3" : "./7.3_compression_pipeline_reduce.py",
        "7.5" : "./7.5_compression_pipeline_no_ldmatrix.py"
    }
    
    try:
        v7_path = paths[version]
        if not os.path.exists(v7_path):
            raise ImportError(f"Could not find physical file '{v7_path}' in the current working directory.")
        
        spec = importlib.util.spec_from_file_location("compression_v7", v7_path)
        comp_pipeline_7 = importlib.util.module_from_spec(spec)
        sys.modules["compression_v7"] = comp_pipeline_7
        spec.loader.exec_module(comp_pipeline_7)

        gluon_pipeline = importlib.import_module("gluon_pipeline")

        dense_matmul = gluon_pipeline.persistent_matmul_pipelined
        pre_compressed_sparse_matmul = gluon_pipeline.sparse_persistent_matmul_pipelined
        runtime_compression_sparse_matmul = comp_pipeline_7.sparse_persistent_matmul_pipelined
        PersistentTileScheduler = comp_pipeline_7.PersistentTileScheduler

    except ImportError as e:
        print(f"Error importing modules: {e}")
        print("Make sure 7_compession_pipeline.py, gluon_pipeline.py, prune.py, and compress_2_4.py are in the current directory.")
        exit(1)

    shapes = sorted(shapes, key=lambda x: x[0] * x[1] * x[2], reverse=True)
    configs = matmul_get_configs()
    
    data_log = []
    
    # --- GLOBAL TRACKERS FOR PEAK RATIO ---
    global_max_ratio = 0.0
    best_ratio_shape = None
    best_ratio_config = None
    
    print("\nStarting manual autotuning search...")
    print("-" * 80)
    
    for i, (M, N, K) in enumerate(shapes):
        shape_str = f"{M}-{N}-{K}"
        
        best_runtime_tflops = 0.0
        best_runtime_config = None
        
        best_dense_tflops = 0.0
        best_dense_config = None

        best_precomp_tflops = 0.0
        best_precomp_config = None
        
        for config in configs:
            bm = config.kwargs["BLOCK_SIZE_M"]
            bn = config.kwargs["BLOCK_SIZE_N"]
            bk = config.kwargs["BLOCK_SIZE_K"]
            num_buffers = config.kwargs["num_buffers"]
            num_warps = config.num_warps
            
            metrics = benchmark_kernels(
                M=M, N=N, K=K, 
                BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk, 
                num_buffers=num_buffers, num_warps=num_warps,
            )
            
            dense_tflops = metrics.get("dense_tflops") or 0.0
            runtime_tflops = metrics.get("runtime_tflops") or 0.0
            precomp_tflops = metrics.get("precomp_tflops") or 0.0
            
            data_log.append({
                "Shape": shape_str,
                "Dense_TFLOPS": dense_tflops,
                "Runtime_TFLOPS": runtime_tflops,
                "Precomp_TFLOPS": precomp_tflops
            })

            config_details = f"BM={bm:3}, BN={bn:3}, BK={bk:3}, warps={num_warps:2}, buffers={num_buffers}"

            # Update shape-specific bests
            if runtime_tflops > best_runtime_tflops:
                best_runtime_tflops = runtime_tflops
                best_runtime_config = config_details

            if dense_tflops > best_dense_tflops:
                best_dense_tflops = dense_tflops
                best_dense_config = config_details
                
            if precomp_tflops > best_precomp_tflops:
                best_precomp_tflops = precomp_tflops
                best_precomp_config = config_details

        if best_dense_tflops > 0:
            if best_runtime_tflops / best_dense_tflops > global_max_ratio:
                global_max_ratio = best_runtime_tflops / best_dense_tflops
                best_ratio_shape = shape_str
                best_ratio_config = best_runtime_config

        # print(global_max_ratio)

        print(f"[{i+1}/{len(shapes)}] Shape {shape_str}:")
        print(f"  -> Best Dense   : {best_dense_tflops:7.2f} TFLOPS | Config: {best_dense_config}")
        print(f"  -> Best Runtime : {best_runtime_tflops:7.2f} TFLOPS | Config: {best_runtime_config}")
        print(f"  -> Best Precomp : {best_precomp_tflops:7.2f} TFLOPS | Config: {best_precomp_config}")
        print("-" * 80)

    # --- PRINT THE GLOBAL WINNER AT THE END ---
    print("\n" + "=" * 80)
    print("🏆 GLOBAL WINNER: HIGHEST RUNTIME vs DENSE RATIO")
    print("=" * 80)
    if global_max_ratio > 0:
        print(f"Shape  : {best_ratio_shape}")
        print(f"Ratio  : {global_max_ratio:.3f}x speedup")
        print(f"Layout : {best_ratio_config}")
    else:
        print("No valid ratios found (dense TFLOPS was 0).")
    print("=" * 80 + "\n")
