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

# ---------------------------------------------------------------------------
# WORKSPACE & ENVIRONMENT CONFIGURATION
# ---------------------------------------------------------------------------
SCRATCH_WORKSPACE = "compiler_scratch"
JOB_ID = str(os.getpid())

os.makedirs(SCRATCH_WORKSPACE, exist_ok=True)
os.makedirs(os.path.join(SCRATCH_WORKSPACE, f"triton_cache_{JOB_ID}"), exist_ok=True)
os.makedirs(os.path.join(SCRATCH_WORKSPACE, f"cuda_cache_{JOB_ID}"), exist_ok=True)

os.environ["TRITON_CACHE_DIR"] = os.path.join(SCRATCH_WORKSPACE, f"triton_cache_{JOB_ID}")
os.environ["TMPDIR"] = SCRATCH_WORKSPACE
os.environ["TMP"] = SCRATCH_WORKSPACE
os.environ["TEMP"] = SCRATCH_WORKSPACE
os.environ["CUDA_CACHE_PATH"] = os.path.join(SCRATCH_WORKSPACE, f"cuda_cache_{JOB_ID}")
os.environ["TORCH_HOME"] = os.path.join(SCRATCH_WORKSPACE, f"cuda_cache_{JOB_ID}")
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# ---------------------------------------------------------------------------
# BASELINE IMPORTS (UPDATE PATHS AS NEEDED)
# ---------------------------------------------------------------------------
from prune import prune_2_4
from compress_2_4 import compress_dense_to_sparse
import gluon_ws_dense
import gluon_ws_sparse

# ===========================================================================
# PLACEHOLDER FOR YOUR KERNEL IMPORTS
# ===========================================================================
# Option A: Direct import if in the same folder / Python path
# import fused_post_compress_module
# import separated_post_compress_module

# Option B: Dynamic file paths (Fill in your absolute or relative paths here)
FUSED_KERNEL_PATH = "/home/notming/links/scratch/compression/kernels/10.1_prune_acc.py"
SEPARATED_KERNEL_PATH = "/home/notming/links/scratch/compression/dev/10.2_prune_acc_2_kernel.py"
# ===========================================================================


def load_module_from_path(module_name, file_path):
    """Utility to dynamically import a module from an explicit file path."""
    if not os.path.exists(file_path):
        raise ImportError(f"Could not find target file at: '{file_path}'")
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def to_tflops(ms, M, N, K):
    return (2 * M * N * K) / (ms * 1e-3 * 1e12) if ms else 0.0


def safe_bench(fn, rep=100, use_cudagraph=True):
    """Helper to benchmark callables cleanly with CUDA Graph or standard Triton timer."""
    try:
        if use_cudagraph:
            return triton.testing.do_bench_cudagraph(fn, rep=rep)
        else:
            return triton.testing.do_bench(fn, warmup=25, rep=rep)
    except Exception as e:
        print(f"[BENCHMARK ERROR]: {e}")
        torch.cuda.synchronize()
        return None


def benchmark_post_compression_kernels(M, N, K, fused_module, separated_module):
    # Prepare input matrices
    A = torch.randn((M, K), device="cuda", dtype=torch.float16)
    B = torch.randn((K, N), device="cuda", dtype=torch.float16)

    # Pre-prune and pre-compress input matrix A for sparse matmul input requirements
    A_pruned = prune_2_4(A)
    A_comp, E = compress_dense_to_sparse(A_pruned)
    E = E.view(M // 16, K)

    # 1. Dense WS Triton Baseline
    ms_dense_ws = safe_bench(lambda: gluon_ws_dense.run_ws_matmul(A, B), use_cudagraph=True)
    tflops_dense_ws = to_tflops(ms_dense_ws, M, N, K)

    # 2. Sparse Pre-compressed WS Triton Baseline (Inputs precompressed, standard dense out)
    ms_sparse_ws = safe_bench(lambda: gluon_ws_sparse.run_sparse_ws_matmul(A_comp, E, B), use_cudagraph=True)
    tflops_sparse_ws = to_tflops(ms_sparse_ws, M, N, K)

    # 3. Fused Kernel: Sparse Matmul + Output Prune/Compress in 1 Kernel
    ms_fused_postcomp = safe_bench(
        lambda: fused_module.run_sparse_ws_matmul(A_comp, E, B, tune=True),
        use_cudagraph=True
    )
    tflops_fused_postcomp = to_tflops(ms_fused_postcomp, M, N, K)

    # 4. Separated Kernel: Sparse Matmul Kernel -> Dense C -> Compression Kernel
    ms_separated_postcomp = safe_bench(
        lambda: separated_module.run_matmul_then_compress_separate(A_comp, E, B, tune=True),
        use_cudagraph=True
    )
    tflops_separated_postcomp = to_tflops(ms_separated_postcomp, M, N, K)

    return {
        "Dense_WS_TFLOPS": tflops_dense_ws,
        "Sparse_Precomp_WS_TFLOPS": tflops_sparse_ws,
        "Fused_Postcomp_TFLOPS": tflops_fused_postcomp,
        "Separated_Postcomp_TFLOPS": tflops_separated_postcomp,
    }


def plot_benchmark_results(df_raw, N):
    if df_raw.empty:
        print("No valid data points to plot.")
        return

    df = df_raw.copy()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=False)

    x = np.arange(len(df["Shape"]))

    # Upper Plot: TFLOPS Comparison
    ax1.plot(x, df["Dense_WS_TFLOPS"], marker="o", linewidth=2, label="Dense WS Baseline", color="#2b5c8f")
    ax1.plot(x, df["Sparse_Precomp_WS_TFLOPS"], marker="^", linewidth=2, label="Precomp Sparse WS", color="#2ca02c")
    ax1.plot(x, df["Fused_Postcomp_TFLOPS"], marker="s", linewidth=2, label="Fused Post-Compress Kernel", color="#d95f02")
    ax1.plot(x, df["Separated_Postcomp_TFLOPS"], marker="d", linewidth=2, label="Separated 2-Kernel Pipeline", color="#9467bd", linestyle="--")

    ax1.set_ylabel("TFLOPS", fontsize=12, fontweight="bold")
    ax1.set_title(f"Post-Compression Matmul Kernel Comparison (N={N})", fontsize=14, fontweight="bold", pad=15)
    ax1.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="none")
    ax1.grid(True, linestyle="--", alpha=0.5)

    ax1.set_xticks(x[::7])
    ax1.set_xticklabels(df["Shape"].tolist()[::7], rotation=35, ha="right", fontsize=9)

    # Lower Plot: Relative Speedups vs Dense Baseline
    width = 0.35
    df["Speedup_Fused_vs_Dense"] = df["Fused_Postcomp_TFLOPS"] / df["Dense_WS_TFLOPS"]
    df["Speedup_Separated_vs_Dense"] = df["Separated_Postcomp_TFLOPS"] / df["Dense_WS_TFLOPS"]

    ax2.bar(x - width / 2, df["Speedup_Fused_vs_Dense"], width, label="Fused Post-Compress vs Dense WS", color="#e6550d")
    ax2.bar(x + width / 2, df["Speedup_Separated_vs_Dense"], width, label="Separated 2-Kernel vs Dense WS", color="#756bb1")

    ax2.axhline(1.0, color="#7f7f7f", linestyle="--", linewidth=1.2, alpha=0.8)
    ax2.set_ylabel("Speedup Ratio", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Matrix Shapes (M-N-K)", fontsize=12, fontweight="bold", labelpad=10)

    ax2.set_xticks(x[::7])
    ax2.set_xticklabels(df["Shape"].tolist()[::7], rotation=35, ha="right", fontsize=9)

    ax2.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="none")
    ax2.grid(True, linestyle="--", alpha=0.3)

    plt.tight_layout()
    output_image = f"/home/notming/links/scratch/compression/results/plots/v10.1/v10.1_benchmark_{N}.png"
    plt.savefig(output_image, dpi=300, bbox_inches="tight")
    print(f"\n[INFO] Benchmark plot successfully saved to '{output_image}'")
    plt.show()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python benchmark.py <N>")
        sys.exit(1)

    N = int(sys.argv[1])

    # Dynamic loading of custom modules
    try:
        fused_module = load_module_from_path("fused_postcomp", FUSED_KERNEL_PATH)
        separated_module = load_module_from_path("separated_postcomp", SEPARATED_KERNEL_PATH)
    except Exception as e:
        print(f"Error loading target kernel modules: {e}")
        sys.exit(1)

    dim = [
        768, 1024, 1536, 2048, 3072, 4096, 6144, 8192, 12288, 16384, 24576, 32768, 49152
        # 768,
    ]

    shapes = [
        (i, N, j)
        for i in dim
        for j in dim
    ]

    shapes = sorted(shapes, key=lambda x: x[0] * x[1] * x[2], reverse=True)
    data_log = []

    for idx, (M, N, K) in enumerate(shapes):
        shape_str = f"{M}-{N}-{K}"
        print(f"Benchmarking {shape_str} ({idx + 1}/{len(shapes)})...", flush=True)

        metrics = benchmark_post_compression_kernels(M, N, K, fused_module, separated_module)

        data_log.append({
            "Shape": shape_str,
            **metrics
        })

        print(f"  Dense WS TFLOPS:             {metrics['Dense_WS_TFLOPS']:.2f}")
        print(f"  Precomp Sparse WS TFLOPS:    {metrics['Sparse_Precomp_WS_TFLOPS']:.2f}")
        print(f"  Fused Postcomp TFLOPS:       {metrics['Fused_Postcomp_TFLOPS']:.2f}")
        print(f"  Separated Postcomp TFLOPS:   {metrics['Separated_Postcomp_TFLOPS']:.2f}\n")

    df_raw = pd.DataFrame(data_log)
    plot_benchmark_results(df_raw, N)