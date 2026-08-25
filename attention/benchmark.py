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
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

from triton.experimental.gluon.nvidia.hopper import TensorDescriptor
from triton.language.core import _aggregate as aggregate

# ---------------------------------------------------------------------------
# WORKSPACE & ENVIRONMENT OVERRIDES
# ---------------------------------------------------------------------------
SCRATCH_WORKSPACE = "sbatch_compiler_scratch"
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
    """Standard FlashAttention FLOP calculation: Total FLOPs = 4 * B * H * S^2 * D"""
    if not ms or ms <= 0:
        return 0.0
    flops = 4.0 * batch * num_heads * (seq_len ** 2) * head_dim
    return flops / (ms * 1e-3 * 1e12)

def get_best_config(module, head_dim: int = None):
    """Extracts best_config from module autotuner cache or direct attributes."""
    cache = getattr(module, "_autotune_cache", {})
    if head_dim is not None:
        if head_dim in cache and getattr(cache[head_dim], "best_config", None) is not None:
            return cache[head_dim].best_config
    else:
        for autotuner in cache.values():
            if getattr(autotuner, "best_config", None) is not None:
                return autotuner.best_config

    for name in ["fa3_autotune_kernel", "sparse_ws_kernel_autotune", "fa3_warp_specialized_kernel"]:
        obj = getattr(module, name, None)
        if obj and getattr(obj, "best_config", None) is not None:
            return obj.best_config

    return "Kernel Failed / Not Set"

def prepare_kernel_runner(module, Q, K, V, tune=True, manual_config=None):
    """
    Pre-allocates host TMA descriptors and output memory once, returning a pure 
    GPU launch closure compatible with CUDA Graphs, along with O and best_config.
    """
    BATCH, NUM_HEADS, SEQ_LEN, HEAD_DIM = Q.shape
    O = torch.empty_like(Q)
    
    # Warmup and autotune/run once to determine best config and verify shapes
    O_ref, config = module.run_fa3_kernel(Q, K, V, tune=tune, manual_config=manual_config)
    torch.cuda.synchronize()

    # Extract kernel object and configuration kwargs
    if tune:
        kernel_jit = module.get_autotuned_kernel(HEAD_DIM)
        best_cfg = kernel_jit.best_config
        cfg_kwargs = best_cfg.kwargs
        num_warps = best_cfg.num_warps
    else:
        kernel_jit = module.fa3_warp_specialized_kernel
        best_cfg = manual_config
        cfg_kwargs = manual_config
        num_warps = manual_config["warps"]

    bm = cfg_kwargs["BLOCK_SIZE_M"]
    bn = cfg_kwargs["BLOCK_SIZE_N"]
    bk = cfg_kwargs["BLOCK_SIZE_K"]
    sf = cfg_kwargs["SUBTILE_FACTOR"]
    stages = cfg_kwargs["num_stages"]

    # Flatten tensors for TMA Descriptor allocation
    Q_flat = Q.reshape(-1, HEAD_DIM)
    K_flat = K.reshape(-1, HEAD_DIM)
    V_flat = V.reshape(-1, HEAD_DIM)
    O_flat = O.reshape(-1, HEAD_DIM)

    dummy_block = [1, 1]
    dummy_layout = gluon.language.NVMMASharedLayout.get_default_for(dummy_block, gluon.language.float16)

    # Detect 4-partition (split Q0/Q1, O0/O1) vs 3-partition (single Q, O) by signature
    is_4_partition = "q0_desc" in getattr(module.fa3_warp_specialized_kernel, "arg_names", module.fa3_warp_specialized_kernel.fn.__code__.co_varnames)

    if is_4_partition:
        q0_desc = TensorDescriptor.from_tensor(Q_flat, dummy_block, dummy_layout)
        q1_desc = TensorDescriptor.from_tensor(Q_flat, dummy_block, dummy_layout)
        k_desc = TensorDescriptor.from_tensor(K_flat, dummy_block, dummy_layout)
        v_desc = TensorDescriptor.from_tensor(V_flat, dummy_block, dummy_layout)
        o0_desc = TensorDescriptor.from_tensor(O_flat, dummy_block, dummy_layout)
        o1_desc = TensorDescriptor.from_tensor(O_flat, dummy_block, dummy_layout)

        hook_args = {
            "BLOCK_SIZE_M": bm, "BLOCK_SIZE_N": bn, "BLOCK_SIZE_K": bk, "SUBTILE_FACTOR": sf,
            "q0_desc": q0_desc, "q1_desc": q1_desc, "k_desc": k_desc, "v_desc": v_desc,
            "o0_desc": o0_desc, "o1_desc": o1_desc
        }
        descriptors = (q0_desc, q1_desc, k_desc, v_desc, o0_desc, o1_desc)
    else:
        q_desc = TensorDescriptor.from_tensor(Q_flat, dummy_block, dummy_layout)
        k_desc = TensorDescriptor.from_tensor(K_flat, dummy_block, dummy_layout)
        v_desc = TensorDescriptor.from_tensor(V_flat, dummy_block, dummy_layout)
        o_desc = TensorDescriptor.from_tensor(O_flat, dummy_block, dummy_layout)

        hook_args = {
            "BLOCK_SIZE_M": bm, "BLOCK_SIZE_N": bn, "BLOCK_SIZE_K": bk, "SUBTILE_FACTOR": sf,
            "q_desc": q_desc, "k_desc": k_desc, "v_desc": v_desc, "o_desc": o_desc
        }
        descriptors = (q_desc, k_desc, v_desc, o_desc)

    # Apply TMA layout and block shape hook
    module.fa3_tma_set_block_size_hook(hook_args)

    # Calculate Grid Dimensions
    num_sms = torch.cuda.get_device_properties("cuda").multi_processor_count
    num_pid = triton.cdiv(SEQ_LEN, bm)
    total_tiles = num_pid * BATCH * NUM_HEADS
    grid = (min(num_sms, total_tiles),)

    scheduler = module.GroupedPersistentTileScheduler(8)

    # Construct zero-host-overhead GPU launch closure
    if is_4_partition:
        def launch_fn():
            module.fa3_warp_specialized_kernel[grid](
                *descriptors,
                scheduler,
                SEQ_LEN, HEAD_DIM, NUM_HEADS,
                BLOCK_SIZE_M=bm, BLOCK_SIZE_N=bn, BLOCK_SIZE_K=bk,
                num_stages=stages, SUBTILE_FACTOR=sf, num_warps=num_warps,
            )
    else:
        def launch_fn():
            module.fa3_warp_specialized_kernel[grid](
                *descriptors,
                scheduler,
                SEQ_LEN, HEAD_DIM, NUM_HEADS,
                BLOCK_SIZE_M=bm, BLOCK_SIZE_N=bn, BLOCK_SIZE_K=bk,
                num_stages=stages, SUBTILE_FACTOR=sf, num_warps=num_warps
            )

    return launch_fn, O_ref, best_cfg

def benchmark_fa3_kernel(seq_len: int, head_dim: int, active_modules: dict, tune: bool = True, rep: int = 100):
    NUM_HEADS = 16
    BATCH_SIZE = max(1, 16384 // seq_len)
    
    Q = torch.randn((BATCH_SIZE, NUM_HEADS, seq_len, head_dim), device="cuda", dtype=torch.float16)
    K = torch.randn((BATCH_SIZE, NUM_HEADS, seq_len, head_dim), device="cuda", dtype=torch.float16)
    V = torch.randn((BATCH_SIZE, NUM_HEADS, seq_len, head_dim), device="cuda", dtype=torch.float16)

    results = {}

    # PyTorch Baseline
    try:
        ms_torch = triton.testing.do_bench_cudagraph(
            lambda: torch.nn.functional.scaled_dot_product_attention(Q, K, V),
            rep=rep
        )
        tflops_torch = to_attention_tflops(ms_torch, seq_len, head_dim, BATCH_SIZE, NUM_HEADS)
    except Exception as e:
        print(f"PyTorch SDPA failed at SEQ_LEN={seq_len}, HEAD_DIM={head_dim}: {e}")
        tflops_torch, ms_torch = None, None

    results["PyTorch SDPA"] = {"tflops": tflops_torch, "ms": ms_torch}

    # Evaluate registered modules (both 3-Partition and 4-Partition)
    for name, module in active_modules.items():
        try:
            # 1. Pre-allocate descriptors and build isolated execution closure
            launch_fn, O_triton, best_config = prepare_kernel_runner(module, Q, K, V, tune=tune)
            
            # 2. Correctness check against PyTorch reference
            O_torch = torch.nn.functional.scaled_dot_product_attention(Q, K, V)
            torch.testing.assert_close(O_torch, O_triton, rtol=1e-2, atol=1e-2)

            # 3. Benchmark pure GPU time with CUDA Graphs
            ms = triton.testing.do_bench_cudagraph(launch_fn, rep=rep)
            tflops = to_attention_tflops(ms, seq_len, head_dim, batch=BATCH_SIZE, num_heads=NUM_HEADS)
        except Exception as e:
            print(f"[{name}] benchmark failed at SEQ_LEN={seq_len}, HEAD_DIM={head_dim}: {e}")
            ms, tflops, best_config = None, None, "Kernel Failed / Not Set"

        results[name] = {"tflops": tflops, "ms": ms, "config": best_config}

    return results


def plot_benchmark_results(df: pd.DataFrame, head_dim: int, active_kernel_names: list, output_dir: str = "Benchmark"):
    if df.empty:
        print("No valid data points to plot.")
        return

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 9), sharex=True)
    x = np.arange(len(df["SEQ_LEN"]))

    palette = ["#d95f02", "#7570b3", "#1b9e77", "#e7298a", "#e6ab02"]
    
    # 1. Throughput Plot (TFLOPS)
    ax1.plot(x, df["PyTorch SDPA_TFLOPS"], marker="o", linewidth=2.5, label="PyTorch SDPA (Baseline)", color="#2b5c8f")
    
    for idx, name in enumerate(active_kernel_names):
        col_name = f"{name}_TFLOPS"
        if col_name in df.columns:
            ax1.plot(
                x, df[col_name], 
                marker="s" if idx % 2 == 0 else "^", 
                linewidth=2.5, 
                label=f"Custom Triton ({name})", 
                color=palette[idx % len(palette)]
            )

    ax1.set_ylabel("Throughput (TFLOPS)", fontsize=11, fontweight="bold")
    ax1.set_title(f"FlashAttention Throughput vs Sequence Length (HEAD_DIM={head_dim}, FP16)", fontsize=13, fontweight="bold", pad=12)
    ax1.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="none")
    ax1.grid(True, linestyle="--", alpha=0.5)

    # 2. Relative Speedup Ratio Plot vs PyTorch SDPA
    num_kernels = len(active_kernel_names)
    bar_width = 0.75 / max(1, num_kernels)

    for idx, name in enumerate(active_kernel_names):
        tflops_col = f"{name}_TFLOPS"
        if tflops_col in df.columns:
            speedup = df[tflops_col] / df["PyTorch SDPA_TFLOPS"]
            offset = (idx - (num_kernels - 1) / 2.0) * bar_width
            
            bars = ax2.bar(
                x + offset, speedup, bar_width, 
                label=f"{name} vs SDPA", 
                color=palette[idx % len(palette)], 
                alpha=0.85
            )

            for bar in bars:
                height = bar.get_height()
                if not np.isnan(height) and height > 0:
                    ax2.annotate(
                        f"{height:.2f}x",
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=8, fontweight="bold"
                    )

    ax2.axhline(1.0, color="#d62728", linestyle="--", linewidth=1.5, label="Parity (1.0x)")
    ax2.set_ylabel("Speedup vs PyTorch SDPA", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Sequence Length (SEQ_LEN)", fontsize=11, fontweight="bold", labelpad=8)
    ax2.set_xticks(x)
    ax2.set_xticklabels(df["SEQ_LEN"].tolist(), fontsize=10)
    ax2.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="none")
    ax2.grid(True, linestyle="--", alpha=0.3)

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    output_image = os.path.join(output_dir, f"FA3_Benchmark_HEAD_DIM_{head_dim}.png")
    plt.savefig(output_image, dpi=300, bbox_inches="tight")
    print(f"\n[INFO] Benchmark chart saved successfully to '{output_image}'")
    plt.close(fig)

# ---------------------------------------------------------------------------
# MAIN EXECUTION
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dynamic Triton FA3 Kernel Benchmarking Suite")
    
    # Kernel File Paths
    parser.add_argument("--module-3part", type=str, default="/home/notming/links/scratch/attention/kernels/gluon_attention_forward.py", help="3-Partition non-pingpong script")
    parser.add_argument("--module-4part", type=str, default="/home/notming/links/scratch/attention/kernels/gluon_attention_pingpong_overlap.py", help="4-Partition script")
    
    # Execution Flags
    parser.add_argument("--skip-3part", action="store_true", help="Skip 3-Partition non-pingpong kernel")
    parser.add_argument("--skip-3part-pingpong", action="store_true", help="Skip 3-Partition pingpong kernel")
    parser.add_argument("--skip-4part", action="store_true", help="Skip 4-Partition kernel")
    
    # Benchmark Parameters
    parser.add_argument("--head-dims", type=int, nargs="+", default=[64, 128, 256], help="Head dimensions to evaluate in one shot")
    parser.add_argument("--tune", action="store_true", default=True, help="Enable Triton autotuner")
    parser.add_argument("--rep", type=int, default=1000, help="Benchmark repetitions")
    parser.add_argument("--output-dir", type=str, default="/home/notming/links/scratch/attention/results/plots", help="Directory to save plot results")
    
    args = parser.parse_args()

    def load_module(path, name):
        if not os.path.exists(path):
            return None
        try:
            spec = importlib.util.spec_from_file_location(name, path)
            mod = importlib.util.module_from_spec(spec)
            sys.modules[name] = mod
            spec.loader.exec_module(mod)
            return mod
        except Exception as e:
            print(f"[WARN] Failed loading '{path}': {e}")
            return None

    # Load targets dynamically
    candidate_modules = {
        "3-Part (Standard)": (args.module_3part, args.skip_3part),
        "4-Part": (args.module_4part, args.skip_4part),
    }

    active_modules = {}
    for name, (path, skip) in candidate_modules.items():
        if skip:
            print(f"[INFO] Skipping '{name}' via CLI flag.")
            continue
        mod = load_module(path, name)
        if mod is not None:
            active_modules[name] = mod
        else:
            print(f"[INFO] File not found for '{name}' at '{path}'. Skipping.")

    if not active_modules:
        print("[ERROR] No valid kernel modules loaded for benchmarking. Exiting.")
        sys.exit(1)

    seq_lengths = [512, 1024, 2048, 4096, 8192, 16384]

    # Iterate through each head dimension in a single run
    for head_dim in args.head_dims:
        data_log = []

        print(f"\n{'='*70}")
        print(f"   STARTING FLASHATTENTION-3 BENCHMARK (HEAD_DIM={head_dim})")
        print(f"   Active Kernels: {list(active_modules.keys())}")
        print(f"{'='*70}\n")

        for idx, seq_len in enumerate(seq_lengths):
            shape_str = f"SEQ_LEN={seq_len}-HEAD_DIM={head_dim}"
            print(f"Start {shape_str} ({idx+1}/{len(seq_lengths)})", flush=True)

            metrics = benchmark_fa3_kernel(
                seq_len=seq_len, 
                head_dim=head_dim, 
                active_modules=active_modules,
                tune=args.tune, 
                rep=args.rep
            )

            row = {"SEQ_LEN": seq_len}
            summary_str = []
            
            for name, data in metrics.items():
                row[f"{name}_TFLOPS"] = data["tflops"]
                row[f"{name}_ms"] = data["ms"]
                if data["tflops"] is not None:
                    summary_str.append(f"{name}: {data['tflops']:.2f} TFLOPS")

            data_log.append(row)

            print(f"Finish {shape_str} -> " + ", ".join(summary_str))

            # Output autotuned configurations per kernel
            for name, module in active_modules.items():
                cfg = metrics.get(name, {}).get("config")
                if cfg is None or (isinstance(cfg, str) and cfg == "Kernel Failed / Not Set"):
                    cfg = get_best_config(module, head_dim)
                if isinstance(cfg, str):
                    print(f"  [{name}] best config: {cfg}")
                elif cfg is not None and hasattr(cfg, "kwargs"):
                    print(f"  [{name}] best config: {cfg.kwargs}, num_warps={getattr(cfg, 'num_warps', 'N/A')}")
                elif isinstance(cfg, dict):
                    print(f"  [{name}] best config: {cfg}, num_warps={cfg.get('warps', cfg.get('num_warps', 'N/A'))}")
                else:
                    print(f"  [{name}] best config: {cfg}")

        df_dim = pd.DataFrame(data_log)
        
        # Dedicated Summary Table per HEAD_DIM
        print(f"\n{'='*70}")
        print(f"             BENCHMARK RESULTS TABLE (HEAD_DIM={head_dim})")
        print(f"{'='*70}")
        display_cols = ["SEQ_LEN"] + [f"{k}_TFLOPS" for k in ["PyTorch SDPA"] + list(active_modules.keys())]
        print(df_dim[display_cols].to_string(index=False))
        print(f"{'='*70}\n")

        # Plotting per Head Dimension
        plot_benchmark_results(
            df_dim, 
            head_dim=head_dim, 
            active_kernel_names=list(active_modules.keys()), 
            output_dir=args.output_dir
        )