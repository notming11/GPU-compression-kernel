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


def benchmark_fa3_kernel(seq_len: int, head_dim: int, fa3_module, tune: bool = True, rep: int = 1000):
    NUM_HEADS = 16
    BATCH_SIZE = max(1, 16384 // seq_len)
    
    # Allocate inputs
    Q_4d = torch.randn((BATCH_SIZE, NUM_HEADS, seq_len, head_dim), device="cuda", dtype=torch.float16)
    K_4d = torch.randn((BATCH_SIZE, NUM_HEADS, seq_len, head_dim), device="cuda", dtype=torch.float16)
    V_4d = torch.randn((BATCH_SIZE, NUM_HEADS, seq_len, head_dim), device="cuda", dtype=torch.float16)

    try:
        # Benchmark PyTorch SDPA using CUDA graphs
        ms_torch = triton.testing.do_bench_cudagraph(
            lambda: torch.nn.functional.scaled_dot_product_attention(Q_4d, K_4d, V_4d),
            rep=rep
        )
        tflops_torch = to_attention_tflops(ms_torch, seq_len, head_dim, BATCH_SIZE, NUM_HEADS)
    except Exception as e:
        print(f"PyTorch SDPA benchmark failed at SEQ_LEN={seq_len}: {e}")
        tflops_torch = None
        torch.cuda.synchronize()

    # 2. Custom Triton FA3 Kernel
    # CUDA Graph captures all B * H 2D launches into a single graph with zero CPU overhead
    def run_triton_multi_head():
        fa3_module.run_fa3_kernel(
            Q_4d, K_4d, V_4d
        )

    try:
        ms_triton = triton.testing.do_bench_cudagraph(run_triton_multi_head, rep=rep)
        tflops_triton = to_attention_tflops(
            ms_triton, seq_len, head_dim, batch=BATCH_SIZE, num_heads=NUM_HEADS
        )
    except Exception as e:
        print(f"Triton FA3 benchmark failed at SEQ_LEN={seq_len}: {e}")
        ms_triton, tflops_triton = None, None

    return {
        "PyTorch_SDPA_TFLOPS": tflops_torch,
        "Triton_FA3_TFLOPS": tflops_triton,
        "PyTorch_ms": ms_torch if tflops_torch else None,
        "Triton_ms": ms_triton if tflops_triton else None,
    }


def plot_benchmark_results(df_peak: pd.DataFrame, head_dim: int, output_dir: str = "Benchmark"):
    if df_peak.empty:
        print("No valid data points to plot.")
        return

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 9), sharex=True)
    x = np.arange(len(df_peak["SEQ_LEN"]))

    # 1. Throughput Plot (TFLOPS)
    ax1.plot(x, df_peak["PyTorch_SDPA_TFLOPS"], marker="o", linewidth=2.5, label="PyTorch SDPA (Native FA2)", color="#2b5c8f")
    ax1.plot(x, df_peak["Triton_FA3_TFLOPS"], marker="s", linewidth=2.5, label="Custom Triton Gluon FA3", color="#d95f02")

    ax1.set_ylabel("Throughput (TFLOPS)", fontsize=11, fontweight="bold")
    ax1.set_title(f"FlashAttention Throughput vs Sequence Length (HEAD_DIM={head_dim}, FP16)", fontsize=13, fontweight="bold", pad=12)
    ax1.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="none")
    ax1.grid(True, linestyle="--", alpha=0.5)

    # 2. Relative Speedup Ratio Plot
    df_peak["Speedup"] = df_peak["Triton_FA3_TFLOPS"] / df_peak["PyTorch_SDPA_TFLOPS"]
    
    width = 0.4
    bars = ax2.bar(x, df_peak["Speedup"], width, color="#729ece", edgecolor="#2b5c8f", alpha=0.85)
    ax2.axhline(1.0, color="#d62728", linestyle="--", linewidth=1.5, label="Parity (1.0x)")

    # Add text labels on top of bars
    for bar in bars:
        height = bar.get_height()
        if not np.isnan(height) and height > 0:
            ax2.annotate(f"{height:.2f}x",
                         xy=(bar.get_x() + bar.get_width() / 2, height),
                         xytext=(0, 3), textcoords="offset points",
                         ha="center", va="bottom", fontsize=9, fontweight="bold")

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

# ---------------------------------------------------------------------------
# MAIN EXECUTION
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark Triton Gluon FA3 Kernel")
    parser.add_argument("--module", type=str, default="./gluon_attention_forward.py", help="Path to your FA3 Triton script")
    parser.add_argument("--head-dim", type=int, default=128, help="Head dimension to evaluate")
    parser.add_argument("--tune", action="store_true", default=True, help="Enable Triton autotuner during benchmark")
    parser.add_argument("--rep", type=int, default=1000, help="Number of benchmark repetitions")
    
    args = parser.parse_args()

    # Dynamic Module Import
    if not os.path.exists(args.module):
        print(f"[ERROR] Could not find FA3 kernel file at '{args.module}'")
        sys.exit(1)

    try:
        spec = importlib.util.spec_from_file_location("fa3_module", args.module)
        fa3_module = importlib.util.module_from_spec(spec)
        sys.modules["fa3_module"] = fa3_module
        spec.loader.exec_module(fa3_module)
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
        print(f"[{idx+1}/{len(seq_lengths)}] Benchmarking SEQ_LEN={seq_len}, HEAD_DIM={args.head_dim}...", end="", flush=True)
        
        metrics = benchmark_fa3_kernel(
            seq_len=seq_len, 
            head_dim=args.head_dim, 
            fa3_module=fa3_module, 
            tune=args.tune, 
            rep=args.rep
        )

        data_log.append({
            "SEQ_LEN": seq_len,
            "HEAD_DIM": args.head_dim,
            **metrics
        })

        torch_tflops = f"{metrics['PyTorch_SDPA_TFLOPS']:.2f}" if metrics['PyTorch_SDPA_TFLOPS'] else "N/A"
        triton_tflops = f"{metrics['Triton_FA3_TFLOPS']:.2f}" if metrics['Triton_FA3_TFLOPS'] else "N/A"
        
        print(f" Done!\n    └─ PyTorch SDPA: {torch_tflops} TFLOPS | Triton FA3: {triton_tflops} TFLOPS")

    df_raw = pd.DataFrame(data_log)
    
    # Summary Table Output
    print(f"\n{'='*70}")
    print("                     SUMMARY BENCHMARK RESULTS")
    print(f"{'='*70}")
    print(df_raw[["SEQ_LEN", "PyTorch_SDPA_TFLOPS", "Triton_FA3_TFLOPS"]].to_string(index=False))
    
    # Plot results
    plot_benchmark_results(df_raw, head_dim=args.head_dim)