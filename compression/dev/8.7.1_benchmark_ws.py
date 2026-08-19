import argparse
import importlib
import importlib.util
import itertools
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import triton
import time

# 1. Choose a high-capacity path on your storage cluster
SCRATCH_WORKSPACE = "compiler_scratch"

JOB_ID = str(os.getpid())

# 2. Force create the workspace structures (isolated per job)
os.makedirs(SCRATCH_WORKSPACE, exist_ok=True)
os.makedirs(os.path.join(SCRATCH_WORKSPACE, f"triton_cache_{JOB_ID}"), exist_ok=True)
os.makedirs(os.path.join(SCRATCH_WORKSPACE, f"cuda_cache_{JOB_ID}"), exist_ok=True)

# 3. OVERRIDE TRITON (Must happen before 'import triton')
os.environ["TRITON_CACHE_DIR"] = os.path.join(SCRATCH_WORKSPACE, f"triton_cache_{JOB_ID}")

# 4. OVERRIDE NVIDIA PTXAS TEMPORARY FILE DUMPS
os.environ["TMPDIR"] = SCRATCH_WORKSPACE
os.environ["TMP"] = SCRATCH_WORKSPACE
os.environ["TEMP"] = SCRATCH_WORKSPACE

# 5. OVERRIDE PYTORCH / CUDA JIT BINARY PAYLOAD CACHES
os.environ["CUDA_CACHE_PATH"] = os.path.join(SCRATCH_WORKSPACE, f"cuda_cache_{JOB_ID}")
os.environ["TORCH_HOME"] = os.path.join(SCRATCH_WORKSPACE, f"cuda_cache_{JOB_ID}")

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

from prune import prune_2_4
from compress_2_4 import compress_dense_to_sparse

# Import our targets
import gluon_ws_dense
import gluon_ws_sparse


def to_tflops(ms, M, N, K):
    return (2 * M * N * K) / (ms * 1e-3 * 1e12) if ms else 0.0

def benchmark_kernels_ws(M, N, K, comp_module, version):
    A = torch.randn((M, K), device="cuda", dtype=torch.float16)
    B = torch.randn((K, N), device="cuda", dtype=torch.float16)
    C = torch.zeros((M, N), device="cuda", dtype=torch.float16)

    A_pruned = prune_2_4(A)
    A_comp, E = compress_dense_to_sparse(A_pruned)
    E = E.view(M // 16, K) # As in gluon_ws_sparse.py

    start_time = time.perf_counter()
    # 1. Dense WS
    try:
        ms_dense_ws = triton.testing.do_bench_cudagraph(lambda: gluon_ws_dense.run_ws_matmul(A, B), rep=1000)
        tflops_dense_ws = to_tflops(ms_dense_ws, M, N, K)
    except Exception as e:
        tflops_dense_ws = None
        torch.cuda.synchronize()
    
    end_time = time.perf_counter()
    print(end_time - start_time)
    
    start_time = time.perf_counter()
    # 2. Sparse Pre-compressed WS
    try:
        ms_sparse_ws = triton.testing.do_bench_cudagraph(lambda: gluon_ws_sparse.run_sparse_ws_matmul(A_comp, E, B), rep=1000)
        tflops_sparse_ws = to_tflops(ms_sparse_ws, M, N, K)
    except Exception as e:
        tflops_sparse_ws = None
        torch.cuda.synchronize()
    end_time = time.perf_counter()
    print(end_time - start_time)

    start_time = time.perf_counter()
    # 3. Sparse Runtime Compression WS (Dynamic Version)
    try:
        if version in ["10.1"]:
            print("compressed")
            ms_runtime_ws = triton.testing.do_bench_cudagraph(lambda: comp_module.run_sparse_ws_matmul(A_comp, E, B), rep=1000)
        elif version in ["7.8.1", "7.8.2"]:
            print("dense")
            ms_runtime_ws = triton.testing.do_bench_cudagraph(lambda: comp_module.run_sparse_ws_matmul(A, B), rep=1000)
        elif version in ["11.1"]:
            print("2 kernel")
            ms_runtime_ws = triton.testing.do_bench_cudagraph(lambda: comp_module.run_2_kernel_ws_matmul(A, B), rep=1000)
        else:
            print("pruned")
            ms_runtime_ws = triton.testing.do_bench_cudagraph(lambda: comp_module.run_sparse_ws_matmul(A_pruned, B), rep=1000)
            
        tflops_runtime_ws = to_tflops(ms_runtime_ws, M, N, K)
    except Exception as e:
        torch.cuda.synchronize()

    end_time = time.perf_counter()
    print(end_time - start_time)
    
    return {
        "Dense_WS_TFLOPS": tflops_dense_ws,
        "Sparse_Precomp_WS_TFLOPS": tflops_sparse_ws,
        "Runtime_WS_TFLOPS": tflops_runtime_ws,
    }


def plot_benchmark_results(df_raw, N, version):
    if df_raw.empty:
        print("No valid data points to plot.")
        return

    # Extract performance
    df_peak = df_raw.copy()

    # Initialize subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=False)

    # 1. Define the numerical x-coordinates early so both plots can use it
    x = np.arange(len(df_peak["Shape"]))

    # 2. Plot against 'x' instead of df_peak["Shape"] to keep axes aligned perfectly
    ax1.plot(x, df_peak["Dense_WS_TFLOPS"], marker="o", linewidth=2, label="Dense WS Baseline", color="#2b5c8f")
    ax1.plot(x, df_peak["Sparse_Precomp_WS_TFLOPS"], marker="^", linewidth=2, label="Precomp Sparse WS", color="#2ca02c")
    ax1.plot(x, df_peak["Runtime_WS_TFLOPS"], marker="s", linewidth=2, label=f"Runtime WS ({version})", color="#d95f02")

    ax1.set_ylabel("TFLOPS", fontsize=12, fontweight="bold")
    ax1.set_title(f"Warp-Specialized vs Non-WS Compression Performance (N={N}, version={version})", fontsize=14, fontweight="bold", pad=15)
    ax1.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="none")
    ax1.grid(True, linestyle="--", alpha=0.5)
    
    # 3. Apply the slice [::7] to show a tick every 7 shapes on ax1
    ax1.set_xticks(x[::7])
    ax1.set_xticklabels(df_peak["Shape"].tolist()[::7], rotation=35, ha="right", fontsize=9)

    width = 0.35

    # Speedup relative to Dense WS
    df_peak["Speedup_Precomp_WS"] = df_peak["Sparse_Precomp_WS_TFLOPS"] / df_peak["Dense_WS_TFLOPS"]
    df_peak["Speedup_Runtime_WS"] = df_peak["Runtime_WS_TFLOPS"] / df_peak["Dense_WS_TFLOPS"]
    
    print(df_peak["Speedup_Runtime_WS"].mean())

    ax2.bar(x - width/2, df_peak["Speedup_Runtime_WS"], width, label=f"{version} Runtime WS vs Dense WS", color="#729ece")
    ax2.bar(x + width/2, df_peak["Speedup_Precomp_WS"], width, label="Precomp WS vs Dense WS", color="#e15759")

    ax2.axhline(1.0, color="#7f7f7f", linestyle="--", linewidth=1.2, alpha=0.8)

    ax2.set_ylabel("Ratio (Speedup vs Dense WS)", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Matrix Shapes (M-N-K)", fontsize=12, fontweight="bold", labelpad=10)
    
    # 4. Apply the exact same slicing to ax2
    ax2.set_xticks(x[::7])
    ax2.set_xticklabels(df_peak["Shape"].tolist()[::7], rotation=35, ha="right", fontsize=9)
    
    ax2.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="none")
    ax2.grid(True, linestyle="--", alpha=0.3)
    ax2.set_ylim([0.7, 1.7])

    plt.tight_layout()
    output_image = f"Benchmark/v{version}/v{version}_Benchmark_{N}.png"
    os.makedirs(f"Benchmark/v{version}", exist_ok=True)
    plt.savefig(output_image, dpi=300, bbox_inches="tight")
    print(f"\n[INFO] Optimization charts successfully compiled and saved to '{output_image}'")
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Input the version (e.g. 7.6) and N (e.g. 4096)")
        sys.exit(1)
        
    version = sys.argv[1]
    N = int(sys.argv[2])
    
    dim = [
        768, 1024, 1536, 2048, 3072, 4096, 6144, 8192, 12288, 16384, 24576, 32768, 49152,
        # 768
    ]
    # shapes = [
    #     (i, N, j)
    #     for i in dim
    #     for j in dim
    # ]
    shapes = [
        (512, N, j)
        for j in dim
    ]
    
    paths = {
        "7.6": "./7.6_compression_ws.py",
        "7.6.1": "./7.6.1_compression_ws_outstanding_mmas.py",
        "7.6.2": "./7.6.2_compression_ws_register_buffer.py",
        "7.6.3": "./7.6.3_compress_ws_2_partition.py",
        "7.6.4": "./7.6.4_compression_ws_optimization.py",
        "7.7.1": "./7.7.1_ws_seperate_warp_4_buf.py",
        "7.8.1": "./7.8.1_prune_ws.py",
        "7.8.2": "./7.8.2_prune_ws_2_partition.py",
        "10.1": "./10.1_prune_acc.py",
        "11.1": "./11.1_2_kernel_baseline.py",
        # Add future WS iterations here if needed
    }
    
    try:
        v_path = paths[version]
        if not os.path.exists(v_path):
            raise ImportError(
                f"Could not find physical file '{v_path}' in the current working directory."
            )

        spec = importlib.util.spec_from_file_location(f"comp_{version.replace('.', '_')}", v_path)
        comp_module = importlib.util.module_from_spec(spec)
        sys.modules[f"comp_{version.replace('.', '_')}"] = comp_module
        spec.loader.exec_module(comp_module)

    except Exception as e:
        print(f"Error importing modules: {e}")
        sys.exit(1)

    shapes = sorted(shapes, key=lambda x: x[0] * x[1] * x[2], reverse=True)
    
    data_log = []
    
    for idx, (M, N, K) in enumerate(shapes):
        shape_str = f"{M}-{N}-{K}"
        print(f"start {shape_str} ({idx+1}/{len(shapes)})", flush = True)
        
        metrics = benchmark_kernels_ws(M, N, K, comp_module, version)
        
        data_log.append({
            "Shape": shape_str,
            **metrics
        })
        print(f"finish {shape_str} -> Dense: {metrics['Dense_WS_TFLOPS']}, Precomp WS: {metrics['Sparse_Precomp_WS_TFLOPS']}, {version} WS: {metrics['Runtime_WS_TFLOPS']}")

        if M <= 768 and N <= 8192:
            best_dense = gluon_ws_dense.ws_kernel_autotune_768.best_config
            best_sparse = gluon_ws_sparse.sparse_ws_kernel_autotune_768.best_config
        else: 
            best_dense = gluon_ws_dense.ws_kernel_autotune_trimmed.best_config
            best_sparse = gluon_ws_sparse.sparse_ws_kernel_autotune_trimmed.best_config
        
        if version == "11.1":
            best_runtime = getattr(comp_module.compress_2_4_autotune, "best_config", "Kernel Failed / Not Set")
        else:
            best_runtime = getattr(comp_module.sparse_ws_kernel_autotune, "best_config", "Kernel Failed / Not Set")            
        
        print(f"  Dense   best config: {best_dense.kwargs}, num_warps={best_dense.num_warps}")
        print(f"  Sparse  best config: {best_sparse.kwargs}, num_warps={best_sparse.num_warps}")
        if isinstance(best_runtime, str):
            print(f"  Runtime best config: {best_runtime}", flush=True)
        else:
            print(f"  Runtime best config: {best_runtime.kwargs}, num_warps={best_runtime.num_warps}", flush=True)

    df_raw = pd.DataFrame(data_log)
    plot_benchmark_results(df_raw, N, version)