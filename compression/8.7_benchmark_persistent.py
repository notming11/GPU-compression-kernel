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
    def valid(BM, BN, BK, warps, buffers):
        SB = buffers == 4
        # Shared Memory
        smem_bytes = 2 * (
                (buffers * BM * BK) +
                ((buffers + SB) * BK * BN) +
                ((1 - SB) * BM * BN)
        ) + (8 * buffers)

        if smem_bytes > 232448: return False

        # STEALB
        if SB and 2 * BN * BK < BM * BN: return False
        if SB and BM > BK: return False

        if (BM * BN) >= 65536 and warps < 12:  # 256x256 blocks require at least 3 warp groups
            return False
        if (BM * BN) <= 4096 and warps > 8:    # Tiny blocks will starve 12 or 16 warps
            return False

        elements_per_thread = (BM * BN) / (warps * 32)
        if elements_per_thread > 256:
            return False

        return True

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
        for BM in (64, 128, 256)
        for BN in (64, 128, 256)
        for BK in (64, 128, 256)
        for buffers in (3, 4, 5, 6, 7)
        for warps in (4, 8, 16)
        if valid(BM, BN, BK, warps, buffers)
    ]

# def matmul_get_configs():
#     return [
#         triton.Config(
#             {
#                 "BLOCK_SIZE_M": BM,
#                 "BLOCK_SIZE_N": BN,
#                 "BLOCK_SIZE_K": BK,
#                 "num_buffers": buffers,
#             },
#             num_warps=warps,
#         )
#         for BM, BN, BK in [[64, 64, 128], [64, 64, 256], [64, 128, 128], [128, 64, 128], [64, 64, 64], [64, 128, 64], [128, 128, 64]] 
#         for buffers in (3, 4, 5, 6, 7)
#         for warps in (4, 8, 16)
#     ]


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
            lambda: dense_matmul(
                A,
                B,
                C,
                BLOCK_M,
                BLOCK_N,
                BLOCK_K,
                num_buffers,
                num_warps,
                PersistentTileScheduler,
            )
        )
        tflops_dense = to_tflops(ms_dense)
    except Exception as e:
        # print(f"dense failed on {BLOCK_M}x{BLOCK_N}x{BLOCK_K}, w:{num_warps}, b:{num_buffers}. Error: {e}")
        pass

    try:
        ms_precomp = triton.testing.do_bench(
            lambda: pre_compressed_sparse_matmul(
                A_comp,
                E,
                B,
                C,
                BLOCK_M,
                BLOCK_N,
                BLOCK_K,
                num_buffers,
                num_warps,
                PersistentTileScheduler,
            )
        )
        tflops_precomp = to_tflops(ms_precomp)
    except Exception as e:
        # print(f"Precomp failed on {BLOCK_M}x{BLOCK_N}x{BLOCK_K}, w:{num_warps}, b:{num_buffers}. Error: {e}")
        pass

    try:
        ms_runtime = triton.testing.do_bench(
            lambda: runtime_compression_sparse_matmul(
                A_pruned,
                B,
                C,
                BLOCK_M,
                BLOCK_N,
                BLOCK_K,
                num_buffers,
                num_warps,
                PersistentTileScheduler,
            )
        )
        tflops_runtime = to_tflops(ms_runtime)
    except Exception as e:
        # print(f"runtime failed on {BLOCK_M}x{BLOCK_N}x{BLOCK_K}, w:{num_warps}, b:{num_buffers}. Error: {e}")
        pass

    if ms_runtime is not None and ms_precomp is not None and ms_precomp > 0:
        overhead = (ms_runtime / ms_precomp - 1.0) * 100.0
    else:
        overhead = float("nan")

    return {
        "dense_tflops": tflops_dense,
        "runtime_tflops": tflops_runtime,
        "precomp_tflops": tflops_precomp,
        "overhead_pct": overhead,
    }


def plot_benchmark_results(df_raw, N, version):
    """Filters for peak performance per shape and generates comparative plots."""
    if df_raw.empty:
        print("No valid data points to plot.")
        return

    # Group by Shape and extract the MAXIMUM TFLOPS (Autotuning Peak)
    df_peak = (
        df_raw.groupby("Shape", sort=False)
        .agg({"Dense_TFLOPS": "max", "Runtime_TFLOPS": "max", "Precomp_TFLOPS": "max"})
        .reset_index()
    )

    # Calculate Speedup Ratios relative to Dense Baseline
    df_peak["Speedup_Runtime"] = df_peak["Runtime_TFLOPS"] / df_peak["Dense_TFLOPS"]
    df_peak["Speedup_Precomp"] = df_peak["Precomp_TFLOPS"] / df_peak["Dense_TFLOPS"]

    print(df_peak["Speedup_Runtime"].mean())

    # Initialize subplots (similar structure to the reference image)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=False)

    # ----------------------------------------------------
    # Plot 1: Line Chart (Absolute Performance in TFLOPS)
    # ----------------------------------------------------
    ax1.plot(
        df_peak["Shape"],
        df_peak["Dense_TFLOPS"],
        marker="o",
        linewidth=2,
        label="Dense Baseline",
        color="#2b5c8f",
    )
    ax1.plot(
        df_peak["Shape"],
        df_peak["Runtime_TFLOPS"],
        marker="s",
        linewidth=2,
        label="Runtime Compression",
        color="#d95f02",
    )
    ax1.plot(
        df_peak["Shape"],
        df_peak["Precomp_TFLOPS"],
        marker="^",
        linewidth=2,
        label="Pre-Compressed Sparse",
        color="#2ca02c",
    )

    ax1.set_ylabel("TFLOPS", fontsize=12, fontweight="bold")
    ax1.set_title(
        f"Kernel Performance Across Matrix Shapes (N={N}, version={version})",
        fontsize=14,
        fontweight="bold",
        pad=15,
    )
    ax1.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="none")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.tick_params(axis="x", rotation=35, labelsize=9)

    # ----------------------------------------------------
    # Plot 2: Bar Chart (Speedup Ratio vs Dense Baseline)
    # ----------------------------------------------------
    x = np.arange(len(df_peak["Shape"]))
    width = 0.35

    ax2.bar(
        x - width / 2,
        df_peak["Speedup_Runtime"],
        width,
        label="Runtime vs Dense",
        color="#729ece",
    )
    ax2.bar(
        x + width / 2,
        df_peak["Speedup_Precomp"],
        width,
        label="Pre-Comp vs Dense",
        color="#e15759",
    )

    # Baseline reference line at 1.0x speedup
    ax2.axhline(1.0, color="#7f7f7f", linestyle="--", linewidth=1.2, alpha=0.8)

    ax2.set_ylabel("Ratio (Speedup)", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Matrix Shapes (M-N-K)", fontsize=12, fontweight="bold", labelpad=10)
    ax2.set_xticks(x)
    ax2.set_xticklabels(df_peak["Shape"], rotation=35, ha="right", fontsize=9)
    ax2.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="none")
    ax2.grid(True, linestyle="--", alpha=0.3)

    # Polish layouts
    plt.tight_layout()
    output_image = f"Benchmark/v{version}_{N}.png"
    plt.savefig(output_image, dpi=300, bbox_inches="tight")
    print(
        f"\n[INFO] Optimization charts successfully compiled and saved to '{output_image}'"
    )
    plt.show()


if __name__ == "__main__":
    # Feel free to add more configurations here to match the density of your target plot!
    if len(sys.argv) < 3:
        print("Input the version (7/7.1/7.2/7.3/7.4/7.5) and N")
        sys.exit(1)
    version = sys.argv[1]
    N = int(sys.argv[2])
    dim = [
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
        (i, N, j)
        for i in dim
        for j in dim
    ]

    paths = {
        "7" : "./7_compression_pipeline.py", 
        "7.1" : "./7.1_compression_pipeline_with_convert.py", 
        "7.2" : "./7.2_compression_pipeline_no_gather.py", 
        "7.3" : "./7.3_compression_pipeline_reduce.py",
        "7.3.1": "./7.3.1_compression_pipeline_reduce_interlayout.py",
        "7.4" : "./7.4_compression_pipeline_ptx_prototype.py",
        "7.5" : "./7.5_compression_pipeline_no_ldmatrix.py"
    }
    try:
        v7_path = paths[version]
        if not os.path.exists(v7_path):
            raise ImportError(
                f"Could not find physical file '{v7_path}' in the current working directory."
            )

        spec = importlib.util.spec_from_file_location("compression_v7", v7_path)
        comp_pipeline_7 = importlib.util.module_from_spec(spec)
        sys.modules["compression_v7"] = comp_pipeline_7
        spec.loader.exec_module(comp_pipeline_7)

        gluon_pipeline = importlib.import_module("gluon_pipeline")

        dense_matmul = gluon_pipeline.persistent_matmul_pipelined
        pre_compressed_sparse_matmul = gluon_pipeline.sparse_persistent_matmul_pipelined
        runtime_compression_sparse_matmul = (
            comp_pipeline_7.sparse_persistent_matmul_pipelined
        )
        PersistentTileScheduler = comp_pipeline_7.PersistentTileScheduler

    except ImportError as e:
        print(f"Error importing modules: {e}")
        print(
            "Make sure 7_compession_pipeline.py, gluon_pipeline.py, prune.py, and compress_2_4.py are in the current directory."
        )
        exit(1)

    shapes = sorted(shapes, key=lambda x: x[0] * x[1] * x[2], reverse=True)
    configs = matmul_get_configs()

    # Array to log structured data points for plotting
    data_log = []

    header_fmt = (
        "| {:<14} | {:<14} | {:<5} | {:<4} | {:<12} | {:<12} | {:<12} | {:<11} |"
    )
    row_fmt = "| {:<14} | {:<14} | {:<5} | {:<4} | {:<12.2f} | {:<12.2f} | {:<12.2f} | {:<10.1f}% |"

    # print("\n" + "="*113)
    # print(header_fmt.format("Shape (M-N-K)", "Tile (M-N-K)", "Warps", "Bufs", "Dense TFLOP", "Runtime TFL", "Precomp TFL", "Overhead"))
    # print("="*113)
    i = 0
    for M, N, K in shapes:
        shape_str = f"{M}-{N}-{K}"
        print(f"start {shape_str}")
        max_dense = None
        max_precomp = None
        max_runtime = None
        dense_config = None
        precomp_config = None
        runtime_config = None
        for config in configs:
            bm = config.kwargs["BLOCK_SIZE_M"]
            bn = config.kwargs["BLOCK_SIZE_N"]
            bk = config.kwargs["BLOCK_SIZE_K"]
            num_buffers = config.kwargs["num_buffers"]
            num_warps = config.num_warps

            # print(config)

            metrics = benchmark_kernels(
                M=M,
                N=N,
                K=K,
                BLOCK_M=bm,
                BLOCK_N=bn,
                BLOCK_K=bk,
                num_buffers=num_buffers,
                num_warps=num_warps,
            )

            if (
                metrics["dense_tflops"] is not None
                and metrics["runtime_tflops"] is not None
                and metrics["precomp_tflops"] is not None
            ):

                if max_dense is None or metrics["dense_tflops"] > max_dense:
                    max_dense = metrics["dense_tflops"]
                    dense_config = config
                if max_precomp is None or metrics["precomp_tflops"] > max_precomp:
                    max_precomp = metrics["precomp_tflops"]
                    precomp_config = config
                if max_runtime is None or metrics["runtime_tflops"] > max_runtime:
                    max_runtime = metrics["runtime_tflops"]
                    runtime_config = config

                # Append payload to our plotting list
                data_log.append(
                    {
                        "Shape": shape_str,
                        "Dense_TFLOPS": metrics["dense_tflops"],
                        "Runtime_TFLOPS": metrics["runtime_tflops"],
                        "Precomp_TFLOPS": metrics["precomp_tflops"],
                    }
                )

        print(f"finish {shape_str}, ({i+1}/{len(shapes)}) -> Max Dense: {max_dense:.2f}, Max Precomp: {max_precomp:.2f}, Max Runtime: {max_runtime:.2f}")
        print(f"dense config: {dense_config}")
        print(f"precomp config: {precomp_config}")
        print(f"runtime config: {runtime_config}", flush = True)
        i += 1

    # Convert logged metrics into a DataFrame and visualize
    df_raw = pd.DataFrame(data_log)
    plot_benchmark_results(df_raw, N, version)
