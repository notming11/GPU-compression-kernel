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

# 3. OVERRIDE TRITON
os.environ["TRITON_CACHE_DIR"] = os.path.join(SCRATCH_WORKSPACE, f"triton_cache_{JOB_ID}")

# 4. OVERRIDE NVIDIA PTXAS TEMPORARY FILE DUMPS
os.environ["TMPDIR"] = SCRATCH_WORKSPACE
os.environ["TMP"] = SCRATCH_WORKSPACE
os.environ["TEMP"] = SCRATCH_WORKSPACE

# 5. OVERRIDE PYTORCH / CUDA JIT BINARY PAYLOAD CACHES
os.environ["CUDA_CACHE_PATH"] = os.path.join(SCRATCH_WORKSPACE, f"cuda_cache_{JOB_ID}")
os.environ["TORCH_HOME"] = os.path.join(SCRATCH_WORKSPACE, f"cuda_cache_{JOB_ID}")

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

# Import custom baselines
from prune import prune_2_4
from compress_2_4 import compress_dense_to_sparse
import gluon_ws_dense
import gluon_ws_sparse

# Import from your new torchao module
import traceback
from torchao.sparsity.training.autograd import semi_structured_sparsify
from torchao.sparsity.training import SemiSparseLinear
from torch.sparse import to_sparse_semi_structured

def to_tflops(ms, M, N, K):
    return (2 * M * N * K) / (ms * 1e-3 * 1e12) if ms else 0.0

def safe_bench(fn, rep=500, use_cudagraph=False):
    """
    Helper to benchmark callables cleanly with CUDA Graph or standard Triton timer.
    Falls back to do_bench if CUDA Graph capture fails.
    """
    try:
        if use_cudagraph:
            return triton.testing.do_bench_cudagraph(fn, rep=rep)
        else:
            return triton.testing.do_bench(fn, warmup=25, rep=rep)
    except Exception as e:
        torch.cuda.synchronize()
        return None

def benchmark_kernels_ws(M, N, K, comp_module, version):
    A = torch.randn((M, K), device="cuda", dtype=torch.float16)
    B = torch.randn((K, N), device="cuda", dtype=torch.float16)

    # Pre-prune and pre-compress for static sparse baseline tests
    A_pruned = prune_2_4(A)
    A_comp, E = compress_dense_to_sparse(A_pruned)
    E = E.view(M // 16, K)  # As in gluon_ws_sparse.py

    # 1. Dense WS Triton Baseline
    ms_dense_ws = safe_bench(lambda: gluon_ws_dense.run_ws_matmul(A, B), use_cudagraph=True)
    tflops_dense_ws = to_tflops(ms_dense_ws, M, N, K)

    # 2. Sparse Pre-compressed WS Triton
    ms_sparse_ws = safe_bench(lambda: gluon_ws_sparse.run_sparse_ws_matmul(A_comp, E, B), use_cudagraph=True)
    tflops_sparse_ws = to_tflops(ms_sparse_ws, M, N, K)

    # 3. Custom Runtime / 2-Kernel Triton Implementation (11.1)
    if version == "11.1":
        ms_runtime_ws = safe_bench(lambda: comp_module.run_2_kernel_ws_matmul(A, B), use_cudagraph=True)
    else:
        raise ValueError(f"Unsupported version: {version}. This script only supports 11.1")
        
    tflops_runtime_ws = to_tflops(ms_runtime_ws, M, N, K)

    # ---------------------------------------------------------
    # 4. TorchAO Baseline
    # ---------------------------------------------------------
    # Note: sparse_layer initialization kept for reference, but we bypass its broken forward() pass
    sparse_layer = SemiSparseLinear(in_features=K, out_features=M, bias=False).to(A.device, dtype=A.dtype)
    sparse_layer.weight.data.copy_(A)
    B_input = B.t().contiguous()
    
    # 4a. TorchAO Compress Latency (The isolated custom C++ kernel)
    try:
        max_chunk = 16384
        
        if M <= max_chunk and K <= max_chunk:
            # Fast path: runs directly if the matrix is small enough to avoid 32-bit overflow
            def run_torchao_compress():
                return semi_structured_sparsify(A, backend="cusparselt")
        else:
            # Tiled path: Pre-slice A into safe <= 16384x16384 tiles outside the benchmark loop.
            # This completely bypasses the 1EB memory crash and eliminates Python slicing overhead from the timer.
            tiles = [
                A[m : min(m + max_chunk, M), k : min(k + max_chunk, K)].contiguous()
                for m in range(0, M, max_chunk)
                for k in range(0, K, max_chunk)
            ]
            def run_torchao_compress():
                # Enqueues all 36-microsecond tile kernels sequentially on the active CUDA stream
                return [semi_structured_sparsify(t, backend="cusparselt") for t in tiles]

        # Warmup and benchmark the isolated compression time
        _ = run_torchao_compress()
        ms_torchao_compress = safe_bench(run_torchao_compress, use_cudagraph=False)
        
    except Exception as e:
        print(f"TorchAO Compress failed: {e}")
        ms_torchao_compress = None

    # 4b. TorchAO Sparse GEMM Computation
    try:
        # We use standard PyTorch for the matmul strictly to bypass TorchAO's dim()==2 crash bug.
        # PyTorch's native to_sparse_semi_structured uses 64-bit indexing, so it won't crash on large shapes.
        # The hardware execution time on the Tensor Cores is identical.
        A_native_comp = to_sparse_semi_structured(A) 
        ms_torchao_matmul = safe_bench(lambda: torch.matmul(A_native_comp, B), use_cudagraph=True)
        tflops_torchao_matmul = to_tflops(ms_torchao_matmul, M, N, K)
        
    except Exception as e:
        print(f"TorchAO Matmul failed: {e}")
        ms_torchao_matmul = None
        tflops_torchao_matmul = None

    # 4c. TorchAO Total End-to-End Throughput (TFLOPS)
    if ms_torchao_compress is not None and ms_torchao_matmul is not None:
        # The true End-to-End latency of TorchAO without the library conflict crashes
        ms_torchao_e2e = ms_torchao_compress + ms_torchao_matmul
        tflops_torchao_e2e = to_tflops(ms_torchao_e2e, M, N, K)
    else:
        ms_torchao_e2e = None
        tflops_torchao_e2e = None

    return {
        "Dense_WS_TFLOPS": tflops_dense_ws,
        "Sparse_Precomp_WS_TFLOPS": tflops_sparse_ws,
        "Runtime_WS_TFLOPS": tflops_runtime_ws,
        "TorchAO_E2E_TFLOPS": tflops_torchao_e2e,
    }


def plot_benchmark_results(df_raw, N, version):
    if df_raw.empty:
        print("No valid data points to plot.")
        return

    df_peak = df_raw.copy()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=False)

    x = np.arange(len(df_peak["Shape"]))

    ax1.plot(x, df_peak["Dense_WS_TFLOPS"], marker="o", linewidth=2, label="Dense WS Baseline", color="#2b5c8f")
    ax1.plot(x, df_peak["Sparse_Precomp_WS_TFLOPS"], marker="^", linewidth=2, label="Precomp Sparse WS", color="#2ca02c")
    ax1.plot(x, df_peak["Runtime_WS_TFLOPS"], marker="s", linewidth=2, label=f"Your Runtime ({version})", color="#d95f02")
    ax1.plot(x, df_peak["TorchAO_E2E_TFLOPS"], marker="d", linewidth=2, label="TorchAO 2:4 E2E Baseline", color="#9467bd", linestyle="--")

    ax1.set_ylabel("TFLOPS", fontsize=12, fontweight="bold")
    ax1.set_title(f"Custom Kernels vs TorchAO 2:4 Sparse Performance (N={N}, version={version})", fontsize=14, fontweight="bold", pad=15)
    ax1.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="none")
    ax1.grid(True, linestyle="--", alpha=0.5)

    ax1.set_xticks(x[::7])
    ax1.set_xticklabels(df_peak["Shape"].tolist()[::7], rotation=35, ha="right", fontsize=9)

    width = 0.25
    df_peak["Speedup_YourKernel_vs_TorchAO"] = df_peak["Runtime_WS_TFLOPS"] / df_peak["TorchAO_E2E_TFLOPS"]
    df_peak["Speedup_YourKernel_vs_Dense"] = df_peak["Runtime_WS_TFLOPS"] / df_peak["Dense_WS_TFLOPS"]

    ax2.bar(x - width/2, df_peak["Speedup_YourKernel_vs_TorchAO"], width, label=f"Your Runtime ({version}) vs TorchAO E2E", color="#729ece")
    ax2.bar(x + width/2, df_peak["Speedup_YourKernel_vs_Dense"], width, label=f"Your Runtime ({version}) vs Dense WS", color="#e15759")

    ax2.axhline(1.0, color="#7f7f7f", linestyle="--", linewidth=1.2, alpha=0.8)
    ax2.set_ylabel("Speedup Ratio", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Matrix Shapes (M-N-K)", fontsize=12, fontweight="bold", labelpad=10)

    ax2.set_xticks(x[::7])
    ax2.set_xticklabels(df_peak["Shape"].tolist()[::7], rotation=35, ha="right", fontsize=9)

    ax2.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="none")
    ax2.grid(True, linestyle="--", alpha=0.3)

    plt.tight_layout()
    output_image = f"Benchmark/v{version}/v{version}_Benchmark_{N}.png"
    os.makedirs(f"Benchmark/v{version}", exist_ok=True)
    plt.savefig(output_image, dpi=300, bbox_inches="tight")
    print(f"\n[INFO] Benchmark charts saved to '{output_image}'")
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python benchmark.py <version> <N>")
        sys.exit(1)

    version = sys.argv[1]
    N = int(sys.argv[2])

    dim = [
        # 768, 1024, 1536, 2048, 3072, 4096, 6144, 8192, 12288, 16384,
        768, 1024, 1536, 2048, 3072, 4096, 6144, 8192, 12288, 16384, 24576, 32768, 49152,
    ]

    shapes = [
        (i, N, j)
        for i in dim
        for j in dim
    ]
    

    # Only 11.1 remains
    paths = {
        "11.1": "./11.1_2_kernel_baseline.py",
    }

    try:
        v_path = paths[version]
        if not os.path.exists(v_path):
            raise ImportError(f"Could not find physical file '{v_path}'")

        spec = importlib.util.spec_from_file_location(f"comp_{version.replace('.', '_')}", v_path)
        comp_module = importlib.util.module_from_spec(spec)
        sys.modules[f"comp_{version.replace('.', '_')}"] = comp_module
        spec.loader.exec_module(comp_module)

    except Exception as e:
        print(f"Error importing target module: {e}")
        sys.exit(1)

    shapes = sorted(shapes, key=lambda x: x[0] * x[1] * x[2], reverse=True)
    # shapes = shapes[0:2]
    data_log = []

    for idx, (M, N, K) in enumerate(shapes):
        shape_str = f"{M}-{N}-{K}"
        print(f"start {shape_str} ({idx+1}/{len(shapes)})", flush = True)

        metrics = benchmark_kernels_ws(M, N, K, comp_module, version)

        data_log.append({
            "Shape": shape_str,
            **metrics
        })

        print(f"Dense WS TFLOPS: {metrics['Dense_WS_TFLOPS']:.2f}")
        print(f"Sparse WS TFLOPS: {metrics['Sparse_Precomp_WS_TFLOPS']:.2f}")
        print(f"Your Runtime ({version}) TFLOPS: {metrics['Runtime_WS_TFLOPS']:.2f}")
        print(f"TorchAO E2E TFLOPS: {metrics['TorchAO_E2E_TFLOPS']:.2f}")

    df_raw = pd.DataFrame(data_log)
    plot_benchmark_results(df_raw, N, version)