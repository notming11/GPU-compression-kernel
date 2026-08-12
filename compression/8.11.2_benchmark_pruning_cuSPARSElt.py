import argparse
import importlib
import importlib.util
import os
import sys
import traceback
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import triton
from torch.utils.cpp_extension import load_inline

# ==============================================================================
# 1. Environment Overrides
# ==============================================================================
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

# ==============================================================================
# 2. PyTorch C++ Extension for cuSPARSELt
# ==============================================================================
print("[INFO] Compiling/Loading cuSPARSELt PyTorch Extension...")

CUSPARSELT_INCLUDE = os.environ.get("CUSPARSELT_INCLUDE", "/path/to/cusparselt/include")
CUSPARSELT_LIB = os.environ.get("CUSPARSELT_LIB", "/path/to/cusparselt/lib64")

cusparselt_cpp_source = """
#include <torch/extension.h>
#include <cusparseLt.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <c10/cuda/CUDAStream.h>
#include <iostream>
#include <algorithm>
#include <stdexcept>

#define CHECK_CUSPARSELT(call)                                                  \\
    do {                                                                        \\
        cusparseStatus_t status = call;                                         \\
        if (status != CUSPARSE_STATUS_SUCCESS) {                                \\
            std::cerr << "cuSPARSELt error at " << __FILE__ << ":" << __LINE__  \\
                      << " code: " << status << std::endl;                      \\
            throw std::runtime_error("cuSPARSELt failure");                     \\
        }                                                                       \\
    } while (0)

// --- GLOBAL STATE FOR BENCHMARKING ---
static cusparseLtHandle_t g_handle;
static cusparseLtMatDescriptor_t g_matA, g_matB, g_matC;
static cusparseLtMatmulDescriptor_t g_matmul;
static cusparseLtMatmulAlgSelection_t g_alg_sel;
static cusparseLtMatmulPlan_t g_plan;
static bool g_initialized = false;

// Precomputed tile parameters
static size_t g_tile_compressed_size = 0;
static size_t g_tile_compress_buffer_size = 0;
static torch::Tensor g_compress_buffer;
static const int64_t MAX_CHUNK = 16384;

void init_benchmark_state(int tile_m, int tile_k, int tile_n) {
    if (g_initialized) return;

    CHECK_CUSPARSELT(cusparseLtInit(&g_handle));

    // Initialize descriptors for the standard 16384x16384 (or smaller) tile shape
    CHECK_CUSPARSELT(cusparseLtStructuredDescriptorInit(
        &g_handle, &g_matA, tile_m, tile_k, tile_k, 16, CUDA_R_16F, CUSPARSE_ORDER_ROW, CUSPARSELT_SPARSITY_50_PERCENT));
    CHECK_CUSPARSELT(cusparseLtDenseDescriptorInit(
        &g_handle, &g_matB, tile_k, tile_n, tile_n, 16, CUDA_R_16F, CUSPARSE_ORDER_ROW));
    CHECK_CUSPARSELT(cusparseLtDenseDescriptorInit(
        &g_handle, &g_matC, tile_m, tile_n, tile_n, 16, CUDA_R_16F, CUSPARSE_ORDER_ROW));

    CHECK_CUSPARSELT(cusparseLtMatmulDescriptorInit(
        &g_handle, &g_matmul, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &g_matA, &g_matB, &g_matC, &g_matC, CUSPARSE_COMPUTE_16F));
    CHECK_CUSPARSELT(cusparseLtMatmulAlgSelectionInit(
        &g_handle, &g_alg_sel, &g_matmul, CUSPARSELT_MATMUL_ALG_DEFAULT));
    CHECK_CUSPARSELT(cusparseLtMatmulPlanInit(
        &g_handle, &g_plan, &g_matmul, &g_alg_sel));

    // Query compressed size per tile ONCE during initialization
    CHECK_CUSPARSELT(cusparseLtSpMMACompressedSize(
        &g_handle, &g_plan, &g_tile_compressed_size, &g_tile_compress_buffer_size));

    // Allocate single workspace buffer for tile compression
    if (g_tile_compress_buffer_size > 0) {
        auto options = torch::TensorOptions().device(torch::kCUDA).dtype(torch::kUInt8);
        g_compress_buffer = torch::empty({static_cast<int64_t>(g_tile_compress_buffer_size)}, options);
    }

    g_initialized = true;
}

void teardown_benchmark_state() {
    if (!g_initialized) return;
    CHECK_CUSPARSELT(cusparseLtMatmulPlanDestroy(&g_plan));
    CHECK_CUSPARSELT(cusparseLtDestroy(&g_handle));

    g_compress_buffer = torch::Tensor();
    g_initialized = false;
}

// Optimized Tiled Compression Execution
torch::Tensor compress_tiled_forward(torch::Tensor A) {
    TORCH_CHECK(A.is_cuda(), "Input tensor A must be on CUDA");
    TORCH_CHECK(A.dtype() == torch::kFloat16, "Input tensor A must be Float16");

    const int64_t M = A.size(0);
    const int64_t K = A.size(1);

    // Fetch active PyTorch CUDA stream (prevents Stream 0 global barriers)
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    // Calculate total number of tiles
    const int64_t num_m_tiles = (M + MAX_CHUNK - 1) / MAX_CHUNK;
    const int64_t num_k_tiles = (K + MAX_CHUNK - 1) / MAX_CHUNK;
    const int64_t total_tiles = num_m_tiles * num_k_tiles;

    // Allocate output tensor to hold all compressed tile buffers
    auto options = torch::TensorOptions().device(A.device()).dtype(torch::kUInt8);
    auto compressed_output = torch::empty({total_tiles, static_cast<int64_t>(g_tile_compressed_size)}, options);

    void* workspace_ptr = (g_tile_compress_buffer_size > 0) ? g_compress_buffer.data_ptr() : nullptr;

    int64_t tile_idx = 0;
    for (int64_t m = 0; m < M; m += MAX_CHUNK) {
        for (int64_t k = 0; k < K; k += MAX_CHUNK) {
            // Slice tile sub-view directly without extra allocations
            auto tile = A.slice(0, m, std::min(m + MAX_CHUNK, M))
                         .slice(1, k, std::min(k + MAX_CHUNK, K))
                         .contiguous();

            const __half* d_tile_dense = reinterpret_cast<const __half*>(tile.data_ptr<at::Half>());
            uint8_t* d_tile_compressed = compressed_output[tile_idx].data_ptr<uint8_t>();

            // Enqueue tile compression kernel onto the active stream
            CHECK_CUSPARSELT(cusparseLtSpMMACompress(
                &g_handle,
                &g_plan,
                d_tile_dense,
                d_tile_compressed,
                workspace_ptr,
                stream
            ));

            tile_idx++;
        }
    }

    return compressed_output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("init_benchmark_state", &init_benchmark_state, "Initialize cuSPARSELt tile state");
    m.def("teardown_benchmark_state", &teardown_benchmark_state, "Teardown cuSPARSELt tile state");
    m.def("compress_tiled_forward", &compress_tiled_forward, "Compress dense matrix into 2:4 sparse layout with C++ tiling");
}
"""

ext_build_dir = os.path.join(SCRATCH_WORKSPACE, f"torch_ext_{JOB_ID}")
os.makedirs(ext_build_dir, exist_ok=True)

# JIT Compile the extension
# Note: "functions" parameter removed because PYBIND11_MODULE is defined in the C++ source
cusparselt_ext = load_inline(
    name="cusparselt_ext",
    cpp_sources=cusparselt_cpp_source,
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-arch=sm_90", "-O3"],
    extra_include_paths=[CUSPARSELT_INCLUDE],
    extra_ldflags=[f"-L{CUSPARSELT_LIB}", "-lcusparseLt"],
    build_directory=ext_build_dir,
    with_cuda=True,
)


# ==============================================================================
# 3. Benchmarking Utilities
# ==============================================================================
def to_gbps(ms, M, K):
    if not ms:
        return 0.0
    bytes_processed = (2 + 1 + 0.125) * M * K
    return bytes_processed / (ms * 1e6)  


def safe_bench(fn, rep=100, use_cudagraph=False):
    try:
        if use_cudagraph:
            return triton.testing.do_bench_cudagraph(fn, rep=rep)
        else:
            return triton.testing.do_bench(fn, warmup=25, rep=rep)
    except Exception as e:
        print("\n[safe_bench ERROR CAUGHT]")
        traceback.print_exc()
        torch.cuda.synchronize()
        return None


def benchmark_compression(M, K, comp_module):
    A = torch.randn((M, K), device="cuda", dtype=torch.float16)

    # ---------------------------------------------------------
    # 1. cuSPARSELt C++ Native Tiled Baseline
    # ---------------------------------------------------------
    try:
        # Initialize tile descriptors using standard safe chunk size (<= 16384)
        tile_m = min(M, 16384)
        tile_k = min(K, 16384)
        tile_n = 16  # Dense matrix descriptor dimension

        cusparselt_ext.init_benchmark_state(tile_m, tile_k, tile_n)

        def run_cusparselt():
            return cusparselt_ext.compress_tiled_forward(A)

        # Warmup
        _ = run_cusparselt()

        # Benchmark using standard CUDA events (do_bench)
        ms_baseline = safe_bench(run_cusparselt, use_cudagraph=False)
        gbps_baseline = to_gbps(ms_baseline, M, K)
        
        # Clean up library state
        cusparselt_ext.teardown_benchmark_state()
        
    except Exception as e:
        print(f"cuSPARSELt Failed: {e}")
        traceback.print_exc()
        ms_baseline, gbps_baseline = None, None

    # ---------------------------------------------------------
    # 2. Custom Triton TMA Kernel
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
            def grid_prune(meta):
                return (triton.cdiv(M, meta["BLOCK_SIZE_M"]), triton.cdiv(K, meta["BLOCK_SIZE_K"]))

            comp_module.compress_2_4_autotune[grid_prune](
                a_desc, a_compressed_desc, e_desc,
                M, K
            )
        
        run_custom()
        
        best_cfg = comp_module.compress_2_4_autotune.best_config
        print(f"  -> Best Custom Config: BM={best_cfg.kwargs['BLOCK_SIZE_M']}, "
              f"BK={best_cfg.kwargs['BLOCK_SIZE_K']}, "
              f"warps={best_cfg.num_warps}")

        ms_custom = safe_bench(run_custom, use_cudagraph=False)
        gbps_custom = to_gbps(ms_custom, M, K)
    except Exception as e:
        print(f"Custom Triton Kernel Failed: {e}")
        traceback.print_exc()
        ms_custom, gbps_custom = None, None

    return {
        "cuSPARSELt_ms": ms_baseline,
        "cuSPARSELt_GBps": gbps_baseline,
        "Custom_ms": ms_custom,
        "Custom_GBps": gbps_custom,
    }


# ==============================================================================
# 4. Plotting
# ==============================================================================
def plot_compression_results(df_raw, version):
    if df_raw.empty:
        print("No valid data points to plot.")
        return

    df = df_raw.copy()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=False)

    x = np.arange(len(df["Shape"]))

    # Plot 1: Memory Bandwidth
    ax1.plot(
        x, df["cuSPARSELt_GBps"], marker="d", linewidth=2,
        label="NVIDIA cuSPARSELt C++ (Tiled)", color="#9467bd", linestyle="--"
    )
    ax1.plot(
        x, df["Custom_GBps"], marker="s", linewidth=2,
        label=f"Your TMA Kernel ({version})", color="#d95f02"
    )

    ax1.set_ylabel("Memory Bandwidth (GB/s)", fontsize=12, fontweight="bold")
    ax1.set_title(f"Prune & Compress Performance (version={version})", fontsize=14, fontweight="bold", pad=15)
    ax1.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="none")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.set_xticks(x)
    ax1.set_xticklabels(df["Shape"], rotation=45, ha="right", fontsize=9)

    # Plot 2: Speedup Ratio
    width = 0.4
    df["Speedup_Custom_vs_cuSPARSELt"] = df["cuSPARSELt_ms"] / df["Custom_ms"]

    ax2.bar(
        x, df["Speedup_Custom_vs_cuSPARSELt"], width,
        label="Speedup: Custom vs cuSPARSELt", color="#729ece"
    )

    ax2.axhline(1.0, color="#e15759", linestyle="--", linewidth=1.5, alpha=0.8, label="Baseline (1.0x)")
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


# ==============================================================================
# 5. Main Execution Pipeline
# ==============================================================================
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python benchmark_compress.py <version>")
        sys.exit(1)

    version = sys.argv[1]

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

    dim = [
        # 768, 1024, 1536, 2048, 3072, 4096, 6144, 8192, 12288, 16384,
        # 24576, 32768, 49152,
        49152
    ]
    shapes = sorted([(M, K) for M in dim for K in dim], key=lambda x: x[0] * x[1], reverse=True)

    data_log = []

    print(f"Starting Isolated Prune/Compress Benchmark (Testing {len(shapes)} shapes)...")
    for idx, (M, K) in enumerate(shapes):
        shape_str = f"{M}-{K}"
        print(f"\n[{idx+1}/{len(shapes)}] Benchmarking {shape_str}...", flush=True)

        metrics = benchmark_compression(M, K, comp_module)
        data_log.append({"Shape": shape_str, **metrics})

        if metrics["cuSPARSELt_ms"] is not None:
            print(f"  cuSPARSELt C++: {metrics['cuSPARSELt_ms']:.4f} ms ({metrics['cuSPARSELt_GBps']:.2f} GB/s)")
        if metrics["Custom_ms"] is not None:
            print(f"  Custom TMA:   {metrics['Custom_ms']:.4f} ms ({metrics['Custom_GBps']:.2f} GB/s)")
            
        torch.cuda.empty_cache()

    df_raw = pd.DataFrame(data_log)
    plot_compression_results(df_raw, version)