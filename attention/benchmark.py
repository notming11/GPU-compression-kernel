import argparse
import importlib.util
import os
import sys
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import triton

# ---------------------------------------------------------------------------
# WORKSPACE & ENVIRONMENT OVERRIDES
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
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

# ---------------------------------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------------------------------
def to_attention_tflops(ms: float, seq_len: int, head_dim: int, batch: int = 1, num_heads: int = 1) -> float:
    """
    Standard FlashAttention FLOP calculation:
      - Q * K^T: 2 * batch * num_heads * seq_len * seq_len * head_dim
      - Softmax * V: 2 * batch * num_heads * seq_len * seq_len * head_dim
      Total FLOPs = 4 * B * H * S^2 * D
    """
    if not ms or ms <= 0:
        return 0.0
    flops = 4.0 * batch * num_heads * (seq_len ** 2) * head_dim
    return flops / (ms * 1e-3 * 1e12)


def _invoke_kernel_module(module, q, k, v, tune: bool = True):
    """Helper to flexibly invoke whichever function entrypoint exists in the module."""
    if hasattr(module, "run_fa3_kernel"):
        return module.run_fa3_kernel(q, k, v, tune=tune)
    elif hasattr(module, "gluon_attention_forward"):
        return module.gluon_attention_forward(q, k, v)
    elif hasattr(module, "gluon_fa3_forward"):
        return module.gluon_fa3_forward(q, k, v)
    else:
        raise AttributeError(f"Could not find an execution function in module '{module.__name__}'")


def benchmark_fa3_kernel(seq_len: int, head_dim: int, fa3_3part_module, fa3_4part_module, tune: bool = True, rep: int = 1000):
    NUM_HEADS = 16
    BATCH_SIZE = max(1, 16384 // seq_len)
    
    # Allocate inputs
    Q_4d = torch.randn((BATCH_SIZE, NUM_HEADS, seq_len, head_dim), device="cuda", dtype=torch.float16)
    K_4d = torch.randn((BATCH_SIZE, NUM_HEADS, seq_len, head_dim), device="cuda", dtype=torch.float16)
    V_4d = torch.randn((BATCH_SIZE, NUM_HEADS, seq_len, head_dim), device="cuda", dtype=torch.float16)

    # 1. Benchmark PyTorch SDPA using CUDA graphs
    try:
        ms_torch = triton.testing.do_bench_cudagraph(
            lambda: torch.nn.functional.scaled_dot_product_attention(Q_4d, K_4d, V_4d),
            rep=rep
        )
        tflops_torch = to_attention_tflops(ms_torch, seq_len, head_dim, BATCH_SIZE, NUM_HEADS)
    except Exception as e:
        print(f"PyTorch SDPA benchmark failed at SEQ_LEN={seq_len}: {e}")
        ms_torch, tflops_torch = None, None
        torch.cuda.synchronize()

    # 2. Benchmark Triton FA3 (3 Partition) Kernel
    try:
        _ = _invoke_kernel_module(fa3_3part_module, Q_4d, K_4d, V_4d, tune=tune)
        torch.cuda.synchronize()
        
        ms_triton_3part = triton.testing.do_bench_cudagraph(
            lambda: _invoke_kernel_module(fa3_3part_module, Q_4d, K_4d, V_4d, tune=tune), 
            rep=rep
        )
        tflops_triton_3part = to_attention_tflops(
            ms_triton_3part, seq_len, head_dim, batch=BATCH_SIZE, num_heads=NUM_HEADS
        )
    except Exception as e:
        print(f"Triton FA3 (3-Part) benchmark failed at SEQ_LEN={seq_len}: {e}")
        ms_triton_3part, tflops_triton_3part = None, None

    # # 3. Benchmark Triton FA3 (4 Partition) Kernel
    # try:
    #     _ = _invoke_kernel_module(fa3_4part_module, Q_4d, K_4d, V_4d, tune=tune)
    #     torch.cuda.synchronize()
        
    #     ms_triton_4part = triton.testing.do_bench_cudagraph(
    #         lambda: _invoke_kernel_module(fa3_4part_module, Q_4d, K_4d, V_4d, tune=tune), 
    #         rep=rep
    #     )
    #     tflops_triton_4part = to_attention_tflops(
    #         ms_triton_4part, seq_len, head_dim, batch=BATCH_SIZE, num_heads=NUM_HEADS
    #     )
    # except Exception as e:
    #     print(f"Triton FA3 (4-Part) benchmark failed at SEQ_LEN={seq_len}: {e}")
    #     ms_triton_4part, tflops_triton_4part = None, None

    return {
        "PyTorch_SDPA_TFLOPS": tflops_torch,
        "Triton_3Part_TFLOPS": tflops_triton_3part,
        # "Triton_4Part_TFLOPS": tflops_triton_4part,
        "PyTorch_ms": ms_torch,
        "Triton_3Part_ms": ms_triton_3part,
        # "Triton_4Part_ms": ms_triton_4part,
    }


def plot_benchmark_results(df_peak: pd.DataFrame, head_dim: int, output_dir: str = "Benchmark"):
    if df_peak.empty:
        print("No valid data points to plot.")
        return

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 9), sharex=True)
    x = np.arange(len(df_peak["SEQ_LEN"]))

    # 1. Throughput Plot (TFLOPS)
    ax1.plot(x, df_peak["PyTorch_SDPA_TFLOPS"], marker="o", linewidth=2.5, label="PyTorch SDPA (Native FA2)", color="#2b5c8f")
    ax1.plot(x, df_peak["Triton_3Part_TFLOPS"], marker="s", linewidth=2.5, label="Custom Triton FA3 (3 Partition)", color="#d95f02")
    # ax1.plot(x, df_peak["Triton_4Part_TFLOPS"], marker="^", linewidth=2.5, label="Custom Triton FA3 (4 Partition)", color="#7570b3")

    ax1.set_ylabel("Throughput (TFLOPS)", fontsize=11, fontweight="bold")
    ax1.set_title(f"FlashAttention Throughput vs Sequence Length (HEAD_DIM={head_dim}, FP16)", fontsize=13, fontweight="bold", pad=12)
    ax1.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="none")
    ax1.grid(True, linestyle="--", alpha=0.5)

    # 2. Relative Speedup Ratio Plot vs PyTorch SDPA
    df_peak["Speedup_3Part"] = df_peak["Triton_3Part_TFLOPS"] / df_peak["PyTorch_SDPA_TFLOPS"]
    # df_peak["Speedup_4Part"] = df_peak["Triton_4Part_TFLOPS"] / df_peak["PyTorch_SDPA_TFLOPS"]
    
    width = 0.35
    bars1 = ax2.bar(x - width/2, df_peak["Speedup_3Part"], width, label="3-Partition vs SDPA", color="#729ece", edgecolor="#2b5c8f", alpha=0.85)
    # bars2 = ax2.bar(x + width/2, df_peak["Speedup_4Part"], width, label="4-Partition vs SDPA", color="#e7298a", edgecolor="#7570b3", alpha=0.85)
    
    ax2.axhline(1.0, color="#d62728", linestyle="--", linewidth=1.5, label="Parity (1.0x)")

    # Add text labels on top of bars
    for bars in [bars1, 
                #  bars2
                 ]:
        for bar in bars:
            height = bar.get_height()
            if not np.isnan(height) and height > 0:
                ax2.annotate(f"{height:.2f}x",
                             xy=(bar.get_x() + bar.get_width() / 2, height),
                             xytext=(0, 3), textcoords="offset points",
                             ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    ax2.set_ylabel("Speedup vs PyTorch SDPA", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Sequence Length (SEQ_LEN)", fontsize=11, fontweight="bold", labelpad=8)
    ax2.set_xticks(x)
    ax2.set_xticklabels(df_peak["SEQ_LEN"].tolist(), fontsize=10)
    ax2.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="none")
    ax2.grid(True, linestyle="--", alpha=0.3)

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    output_image = os.path.join(output_dir, f"FA3_Benchmark_HEAD_DIM_{head_dim}.png")
    plt.savefig(output_image, dpi=300, bbox_inches="tight")
    print(f"\n[INFO] Benchmark chart saved successfully to '{output_image}'")
    plt.show()
    
def get_best_config(module):
    """Extracts best_config from module autotuner cache or direct attributes."""
    # 1. Check dynamic autotune cache dict (used in gluon_attention_forward.py)
    cache = getattr(module, "_autotune_cache", {})
    for autotuner in cache.values():
        if getattr(autotuner, "best_config", None) is not None:
            return autotuner.best_config

    # 2. Direct attribute check (used in GEMM benchmark modules)
    for name in ["fa3_autotune_kernel", "sparse_ws_kernel_autotune", "fa3_warp_specialized_kernel"]:
        obj = getattr(module, name, None)
        if obj and getattr(obj, "best_config", None) is not None:
            return obj.best_config

    return "Kernel Failed / Not Set"

# ---------------------------------------------------------------------------
# MAIN EXECUTION
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark Triton FA3 Kernels (3-Part vs 4-Part)")
    parser.add_argument("--module-3part", type=str, default="./gluon_attention_forward.py", help="Path to your 3-partition Triton script")
    parser.add_argument("--module-4part", type=str, default="./gluon_fa3_forward.py", help="Path to your 4-partition Triton script")
    parser.add_argument("--head-dim", type=int, default=128, help="Head dimension to evaluate")
    parser.add_argument("--tune", action="store_true", default=True, help="Enable Triton autotuner during benchmark")
    parser.add_argument("--rep", type=int, default=1000, help="Number of benchmark repetitions")
    
    args = parser.parse_args()

    # Dynamic Module Imports
    def load_module(path, name):
        if not os.path.exists(path):
            print(f"[ERROR] Could not find kernel file at '{path}'")
            sys.exit(1)
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
        return mod

    try:
        fa3_3part_module = load_module(args.module_3part, "fa3_3part_module")
        fa3_4part_module = load_module(args.module_4part, "fa3_4part_module")
    except Exception as e:
        print(f"[ERROR] Failed to load kernel module: {e}")
        sys.exit(1)

    # Sequence lengths to evaluate
    seq_lengths = [512, 1024, 2048, 4096, 8192, 16384]
    data_log = []

    print(f"\n{'='*70}")
    print(f"   STARTING FLASHATTENTION-3 BENCHMARK (HEAD_DIM={args.head_dim})")
    print(f"{'='*70}\n")

    for idx, seq_len in enumerate(seq_lengths):
        shape_str = f"SEQ_LEN={seq_len}-HEAD_DIM={args.head_dim}"
        print(f"start {shape_str} ({idx+1}/{len(seq_lengths)})", flush=True)

        metrics = benchmark_fa3_kernel(
            seq_len=seq_len, 
            head_dim=args.head_dim, 
            fa3_3part_module=fa3_3part_module,
            fa3_4part_module=fa3_4part_module, 
            tune=args.tune, 
            rep=args.rep
        )

        data_log.append({
            "SEQ_LEN": seq_len,
            "HEAD_DIM": args.head_dim,
            **metrics
        })

        # 1. Extract autotune configurations directly
        best_3part = get_best_config(fa3_3part_module)
        # best_4part = get_best_config(fa3_4part_module)

        # 2. Print metrics and configs (GEMM Benchmark Style)
        print(
            f"finish {shape_str} -> SDPA: {metrics['PyTorch_SDPA_TFLOPS']:.2f} TFLOPS, "
            f"3-Part: {metrics['Triton_3Part_TFLOPS']:.2f} TFLOPS, "
            # f"4-Part: {metrics['Triton_4Part_TFLOPS']:.2f} TFLOPS"
        )

        if isinstance(best_3part, str):
            print(f"  3-Part best config: {best_3part}")
        else:
            print(f"  3-Part best config: {best_3part.kwargs}, num_warps={getattr(best_3part, 'num_warps', 'N/A')}")

        # if isinstance(best_4part, str):
        #     print(f"  4-Part best config: {best_4part}", flush=True)
        # else:
        #     print(f"  4-Part best config: {best_4part.kwargs}, num_warps={getattr(best_4part, 'num_warps', 'N/A')}", flush=True)

    df_raw = pd.DataFrame(data_log)
    
    # Summary Table Output
    print(f"\n{'='*70}")
    print("                    SUMMARY BENCHMARK RESULTS")
    print(f"{'='*70}")
    print(df_raw[["SEQ_LEN", 
                  "PyTorch_SDPA_TFLOPS", 
                  "Triton_3Part_TFLOPS", 
                #   "Triton_4Part_TFLOPS"
                  ]].to_string(index=False))
    
    # Plot results
    plot_benchmark_results(df_raw, head_dim=args.head_dim)