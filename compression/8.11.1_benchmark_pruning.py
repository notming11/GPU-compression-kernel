import argparse
import importlib
import importlib.util
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import triton

# ==============================================================================
# 1. Environment Overrides
# ==============================================================================
SCRATCH_WORKSPACE = "compiler_scratch"
JOB_ID = str(os.getpid())

os.makedirs(SCRATCH_WORKSPACE, exist_ok=True)
os.makedirs(os.path.join(SCRATCH_WORKSPACE, f"triton_cache_{JOB_ID}"), exist_ok=True)
os.makedirs(os.path.join(SCRATCH_WORKSPACE, f"cuda_cache_{JOB_ID}"), exist_ok=True)

os.environ["TRITON_CACHE_DIR"] = os.path.join(
    SCRATCH_WORKSPACE, f"triton_cache_{JOB_ID}"
)
os.environ["TMPDIR"] = SCRATCH_WORKSPACE
os.environ["TMP"] = SCRATCH_WORKSPACE
os.environ["TEMP"] = SCRATCH_WORKSPACE
os.environ["CUDA_CACHE_PATH"] = os.path.join(SCRATCH_WORKSPACE, f"cuda_cache_{JOB_ID}")
os.environ["TORCH_HOME"] = os.path.join(SCRATCH_WORKSPACE, f"cuda_cache_{JOB_ID}")
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# TorchAO Import
from torchao.sparsity.training.autograd import semi_structured_sparsify

from common import (
    WGMMA,
    GroupedPersistentTileScheduler
)

# ==============================================================================
# 2. Benchmarking Utilities
# ==============================================================================
def to_gbps(ms, M, K):
    """
    Calculates Memory Bandwidth (GB/s).
    Total Bytes = Read Dense (M*K*2) + Write Comp (M*(K/2)*2) + Write Meta ((M/16)*K*2)
    Total Bytes = 3.125 * M * K
    """
    if not ms:
        return 0.0
    bytes_processed = (2 + 1 + 0.125) * M * K
    return bytes_processed / (ms * 1e6)  # (bytes / (ms * 1e-3)) / 1e9 -> GB/s


import traceback


def safe_bench(fn, rep=100, use_cudagraph=False):
    try:
        if use_cudagraph:
            return triton.testing.do_bench_cudagraph(fn, rep=rep)
        else:
            return triton.testing.do_bench(fn, warmup=25, rep=rep)
    except Exception as e:
        # Stop hiding the error! Print it so we know exactly why it failed.
        print("\n[safe_bench ERROR CAUGHT]")
        traceback.print_exc()
        torch.cuda.synchronize()
        return None


def benchmark_compression(M, K, comp_module):
    A = torch.randn((M, K), device="cuda", dtype=torch.float16)

    # ---------------------------------------------------------
    # 1. TorchAO C++ Fast Kernel (Baseline)
    # ---------------------------------------------------------
    try:
        max_chunk = 16384

        if M <= max_chunk and K <= max_chunk:
            # Single-pass for safe dimensions
            def run_torchao():
                return semi_structured_sparsify(A, backend="cusparselt")

        else:
            # Pre-slice A into safe <= 16384x16384 tiles outside the benchmark loop
            # to eliminate Python slicing overhead from the GPU timer.
            tiles = [
                A[m : min(m + max_chunk, M), k : min(k + max_chunk, K)]
                for m in range(0, M, max_chunk)
                for k in range(0, K, max_chunk)
            ]

            def run_torchao():
                # Enqueues all tile kernels sequentially on the CUDA stream
                # Returns a plain Python list (no torch.cat call!)
                return [
                    semi_structured_sparsify(t, backend="cusparselt") for t in tiles
                ]

        # Warmup
        _ = run_torchao()

        # Benchmark
        ms_torchao = safe_bench(run_torchao, use_cudagraph=False)
        gbps_torchao = to_gbps(ms_torchao, M, K)
    except Exception as e:
        print(f"TorchAO Failed: {e}")
        ms_torchao, gbps_torchao = None, None

    # ---------------------------------------------------------
    # 2. Custom Triton TMA Kernel (Loaded dynamically)
    # ---------------------------------------------------------
    try:
        a_compressed = torch.empty((M, K // 2), device=A.device, dtype=torch.float16)
        e = torch.empty((M // 16, K), device=A.device, dtype=torch.int16)
        
        dummy_block = [1, 1]
        dummy_layout_f16 = comp_module.gl.NVMMASharedLayout.get_default_for(dummy_block, comp_module.gl.float16)
        dummy_layout_i16 = comp_module.gl.NVMMASharedLayout.get_default_for(dummy_block, comp_module.gl.int16)
        
        a_desc = comp_module.TensorDescriptor.from_tensor(A, dummy_block, dummy_layout_f16)
        a_compressed_desc = comp_module.TensorDescriptor.from_tensor(a_compressed, dummy_block, dummy_layout_f16)
        e_desc = comp_module.TensorDescriptor.from_tensor(e, dummy_block, dummy_layout_i16)

        def run_custom():
            # CRITICAL FIX: Persistent schedulers require a 1D grid bounded by the number of SMs
            def grid_prune(meta):
                # num_sms = torch.cuda.get_device_properties("cuda").multi_processor_count
                # num_pid = triton.cdiv(M, meta["BLOCK_SIZE_M"]) * triton.cdiv(K, meta["BLOCK_SIZE_K"])
                # return (min(num_sms, num_pid), )
                return (triton.cdiv(M, meta["BLOCK_SIZE_M"]), triton.cdiv(K, meta["BLOCK_SIZE_K"]))

            comp_module.compress_2_4_autotune[grid_prune](
                a_desc, a_compressed_desc, e_desc,
                # comp_module.GroupedPersistentTileScheduler(8),
                M, K
            )
        
        # Warmup (triggers autotuning)
        run_custom()
        
        # Extract and print the best configuration found by the autotuner
        best_cfg = comp_module.compress_2_4_autotune.best_config
        print(f"  -> Best Custom Config: BM={best_cfg.kwargs['BLOCK_SIZE_M']}, "
              f"BK={best_cfg.kwargs['BLOCK_SIZE_K']}, "
              f"warps={best_cfg.num_warps}")

        # Benchmark the optimal configuration
        ms_custom = safe_bench(run_custom, use_cudagraph=False)
        gbps_custom = to_gbps(ms_custom, M, K)
    except Exception as e:
        print(f"Custom Triton Kernel Failed: {e}")
        import traceback
        traceback.print_exc()
        ms_custom, gbps_custom = None, None

    return {
        "TorchAO_ms": ms_torchao,
        "TorchAO_GBps": gbps_torchao,
        "Custom_ms": ms_custom,
        "Custom_GBps": gbps_custom,
    }


# ==============================================================================
# 3. Plotting
# ==============================================================================
def plot_compression_results(df_raw, version):
    if df_raw.empty:
        print("No valid data points to plot.")
        return

    df = df_raw.copy()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=False)

    x = np.arange(len(df["Shape"]))

    # Plot 1: Memory Bandwidth (Higher is better)
    ax1.plot(
        x,
        df["TorchAO_GBps"],
        marker="d",
        linewidth=2,
        label="TorchAO C++",
        color="#9467bd",
        linestyle="--",
    )
    ax1.plot(
        x,
        df["Custom_GBps"],
        marker="s",
        linewidth=2,
        label=f"Your TMA Kernel ({version})",
        color="#d95f02",
    )

    ax1.set_ylabel("Memory Bandwidth (GB/s)", fontsize=12, fontweight="bold")
    ax1.set_title(
        f"Prune & Compress Performance (version={version})",
        fontsize=14,
        fontweight="bold",
        pad=15,
    )
    ax1.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="none")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.set_xticks(x)
    ax1.set_xticklabels(df["Shape"], rotation=45, ha="right", fontsize=9)

    # Plot 2: Speedup Ratio (Latency based)
    width = 0.4
    df["Speedup_Custom_vs_TorchAO"] = df["TorchAO_ms"] / df["Custom_ms"]

    ax2.bar(
        x,
        df["Speedup_Custom_vs_TorchAO"],
        width,
        label="Speedup: Custom vs TorchAO",
        color="#729ece",
    )

    ax2.axhline(
        1.0,
        color="#e15759",
        linestyle="--",
        linewidth=1.5,
        alpha=0.8,
        label="Baseline (1.0x)",
    )
    ax2.set_ylabel("Speedup Ratio", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Matrix Shapes (M-K)", fontsize=12, fontweight="bold", labelpad=10)
    ax2.set_xticks(x)
    ax2.set_xticklabels(df["Shape"], rotation=45, ha="right", fontsize=9)

    ax2.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="none")
    ax2.grid(True, axis="y", linestyle="--", alpha=0.3)

    plt.tight_layout()
    out_dir = f"Benchmark/v{version}"
    os.makedirs(out_dir, exist_ok=True)
    output_image = f"{out_dir}/v{version}_Compression_Benchmark.png"
    plt.savefig(output_image, dpi=300, bbox_inches="tight")
    print(f"\n[INFO] Benchmark charts saved to '{output_image}'")
    plt.show()


# ==============================================================================
# 4. Main Execution Pipeline
# ==============================================================================
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python benchmark_compress.py <version>")
        sys.exit(1)

    version = sys.argv[1]

    # Map version to physical file
    paths = {
        "11.1": "./11.1_2_kernel_baseline.py",
    }

    try:
        v_path = paths[version]
        if not os.path.exists(v_path):
            raise ImportError(f"Could not find physical file '{v_path}'")

        spec = importlib.util.spec_from_file_location(
            f"comp_{version.replace('.', '_')}", v_path
        )
        comp_module = importlib.util.module_from_spec(spec)
        sys.modules[f"comp_{version.replace('.', '_')}"] = comp_module
        spec.loader.exec_module(comp_module)
    except Exception as e:
        print(f"Error importing target module '{version}': {e}")
        sys.exit(1)

    # Filter massive shapes to avoid TorchAO's 1EB indexing crash
    dim = [
        # 768, 1024, 1536, 2048, 3072, 4096, 6144, 8192, 12288, 16384,
        768,
        1024,
        1536,
        2048,
        3072,
        4096,
        6144,
        8192,
        12288,
        16384,
        24576,
        32768,
        49152,
    ]
    shapes = [
        (M, K)
        for M in dim
        for K in dim
        # if M * K <= (16384 * 16384)
    ]
    shapes = sorted(shapes, key=lambda x: x[0] * x[1], reverse=True)

    data_log = []

    print(
        f"Starting Isolated Prune/Compress Benchmark (Testing {len(shapes)} shapes)..."
    )
    for idx, (M, K) in enumerate(shapes):
        shape_str = f"{M}-{K}"
        print(f"\n[{idx+1}/{len(shapes)}] Benchmarking {shape_str}...", flush=True)

        metrics = benchmark_compression(M, K, comp_module)

        data_log.append({"Shape": shape_str, **metrics})

        if metrics["TorchAO_ms"] is not None:
            print(
                f"  TorchAO C++: {metrics['TorchAO_ms']:.4f} ms ({metrics['TorchAO_GBps']:.2f} GB/s)"
            )
        if metrics["Custom_ms"] is not None:
            print(
                f"  Custom TMA:  {metrics['Custom_ms']:.4f} ms ({metrics['Custom_GBps']:.2f} GB/s)"
            )

    df_raw = pd.DataFrame(data_log)
    plot_compression_results(df_raw, version)
