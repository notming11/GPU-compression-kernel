import importlib
import torch
import triton
import itertools
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import importlib.util

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

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

from prune import prune_2_4
from compress_2_4 import compress_dense_to_sparse

# Import our targets
import gluon_ws_dense
import gluon_ws_sparse

spec_76 = importlib.util.spec_from_file_location("comp_76", "7.6_compression_ws.py")
comp_76 = importlib.util.module_from_spec(spec_76)
sys.modules["comp_76"] = comp_76
spec_76.loader.exec_module(comp_76)

def to_tflops(ms, M, N, K):
    return (2 * M * N * K) / (ms * 1e-3 * 1e12) if ms else 0.0

def benchmark_kernels_ws(M, N, K):
    A = torch.randn((M, K), device="cuda", dtype=torch.float16)
    B = torch.randn((K, N), device="cuda", dtype=torch.float16)
    C = torch.zeros((M, N), device="cuda", dtype=torch.float16)

    A_pruned = prune_2_4(A)
    A_comp, E = compress_dense_to_sparse(A_pruned)
    E = E.view(M // 16, K) # As in gluon_ws_sparse.py

    # 1. Dense WS
    try:
        ms_dense_ws = triton.testing.do_bench(lambda: gluon_ws_dense.run_ws_matmul(A, B))
        tflops_dense_ws = to_tflops(ms_dense_ws, M, N, K)
    except Exception as e:
        tflops_dense_ws = None

    # print("finish dense ws")

    # 2. Sparse Pre-compressed WS
    try:
        ms_sparse_ws = triton.testing.do_bench(lambda: gluon_ws_sparse.run_sparse_ws_matmul(A_comp, E, B))
        tflops_sparse_ws = to_tflops(ms_sparse_ws, M, N, K)
    except Exception as e:
        tflops_sparse_ws = None

    # print("finish sparse ws")

    # 3. Sparse Runtime Compression WS (7.6)
    try:
        ms_runtime_ws = triton.testing.do_bench(lambda: comp_76.run_sparse_ws_matmul(A_pruned, B))
        tflops_runtime_ws = to_tflops(ms_runtime_ws, M, N, K)
    except Exception as e:
        tflops_runtime_ws = None

    # print("finish compression ws")
    return {
        "Dense_WS_TFLOPS": tflops_dense_ws,
        "Sparse_Precomp_WS_TFLOPS": tflops_sparse_ws,
        "Runtime_WS_TFLOPS": tflops_runtime_ws,
    }


def plot_benchmark_results(df_raw, N):
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
    ax1.plot(x, df_peak["Runtime_WS_TFLOPS"], marker="s", linewidth=2, label="Runtime WS (7.6)", color="#d95f02")

    ax1.set_ylabel("TFLOPS", fontsize=12, fontweight="bold")
    ax1.set_title(f"Warp-Specialized vs Non-WS Compression Performance (N={N})", fontsize=14, fontweight="bold", pad=15)
    ax1.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="none")
    ax1.grid(True, linestyle="--", alpha=0.5)
    
    # 3. Apply the slice [::7] to show a tick every 7 shapes on ax1
    ax1.set_xticks(x[::7])
    ax1.set_xticklabels(df_peak["Shape"].tolist()[::7], rotation=35, ha="right", labelsize=9)

    width = 0.2

    # Speedup relative to Dense WS
    df_peak["Speedup_Precomp_WS"] = df_peak["Sparse_Precomp_WS_TFLOPS"] / df_peak["Dense_WS_TFLOPS"]
    df_peak["Speedup_Runtime_WS"] = df_peak["Runtime_WS_TFLOPS"] / df_peak["Dense_WS_TFLOPS"]
    
    print(df_peak["Speedup_Runtime_WS"].mean())

    ax2.bar(x - width/2, df_peak["Speedup_Runtime_WS"], width, label="7.6 Runtime WS vs Dense WS", color="#729ece")
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
    output_image = f"Benchmark/v7.6/v7.6_Benchmark_{N}.png"
    os.makedirs("Benchmark/v7.6", exist_ok=True) # Fixed this line so it actually creates the nested directory
    plt.savefig(output_image, dpi=300, bbox_inches="tight")
    print(f"\n[INFO] Optimization charts successfully compiled and saved to '{output_image}'")
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Input N (e.g. 4096)")
        sys.exit(1)
    N = int(sys.argv[1])
    dim = [
        768, 1024, 1536, 2048, 3072, 4096, 6144, 8192, 12288, 16384, 24576, 32768, 49152,
    ]
    shapes = [
        (i, N, j)
        for i in dim
        for j in dim
    ]

    shapes = sorted(shapes, key=lambda x: x[0] * x[1] * x[2], reverse=True)
    
    data_log = []
    
    # print("="*80)
    # print(f"Benchmarking WS Kernels for N={N}")
    # print("="*80)
    
    for idx, (M, N, K) in enumerate(shapes):
        shape_str = f"{M}-{N}-{K}"
        print(f"start {shape_str} ({idx+1}/{len(shapes)})", flush = True)
        
        metrics = benchmark_kernels_ws(M, N, K)
        
        data_log.append({
            "Shape": shape_str,
            **metrics
        })
        print(f"finish {shape_str} -> Dense: {metrics['Dense_WS_TFLOPS']}, Precomp WS: {metrics['Sparse_Precomp_WS_TFLOPS']}, 7.6 WS: {metrics['Runtime_WS_TFLOPS']}")

        best_dense = gluon_ws_dense.ws_kernel_autotune.best_config
        best_sparse = gluon_ws_sparse.sparse_ws_kernel_autotune.best_config
        best_runtime = comp_76.sparse_ws_kernel_autotune.best_config
        print(f"  Dense   best config: {best_dense.kwargs}, num_warps={best_dense.num_warps}")
        print(f"  Sparse  best config: {best_sparse.kwargs}, num_warps={best_sparse.num_warps}")
        print(f"  Runtime best config: {best_runtime.kwargs}, num_warps={best_runtime.num_warps}", flush=True)

    df_raw = pd.DataFrame(data_log)
    plot_benchmark_results(df_raw, N)
