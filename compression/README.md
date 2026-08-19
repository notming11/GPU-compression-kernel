# Sparse MatMul Kernels (2:4 Structured Sparsity)

Triton/Gluon kernels exploring runtime 2:4 pruning and compression fused with warp-specialized matrix multiplication on H100.

## Version Progression

The numbered files track the development history. Each major version explores a different approach:

| Version | Approach | Key Insight |
|---------|----------|-------------|
| **1–3** | Single-tile compression | Layout transforms, PTX-level register shuffles for 2:4 metadata |
| **4–5** | Compression loop | Multi-tile tiled compression without persistence |
| **6** | Persistent compression | CTA-persistent tile scheduling |
| **7.0–7.5** | Pipelined compression | TMA async copy + software pipelining (double/triple buffering) |
| **7.6** | Warp-specialized (WS) | Split load/compute into separate warp groups |
| **7.7** | WS + separate warp buffers | 4-buffer design with independent warp staging |
| **7.8** | **WS + input pruning** | Prune+compress input tiles before matmul (**negative result — slower than dense**) |
| **8** | Benchmarking scripts | Systematic measurement across 169 shapes |
| **9** | Shape search / profiling | Find optimal tile shapes and sparsity ratios |
| **10** | **WS + output pruning** | Fuse pruning into accumulator writeback (**near-zero overhead**) |
| **11** | **Two-kernel baseline** | Separate prune+compress kernel → sparse matmul (**1.5× dense e2e**) |

## Landmark Kernels (`kernels/`)

### `11.1_2_kernel_baseline.py`
Two-kernel design: a standalone prune+compress kernel writes compressed sparse format, then the existing sparse WS matmul consumes it. Achieves **~990 TFLOPS** (1.5× dense, 3× TorchAO prune standalone).

### `10.1_prune_acc.py`
Single-kernel: fuses 2:4 pruning into the matmul's accumulator-to-output writeback path. Achieves **~1060 TFLOPS** with near-zero overhead vs. pre-computed sparse.

### `7.8.1_prune_ws.py`
Single-kernel: prunes + compresses each input tile inline during the load→compute pipeline. **Negative result**: compression latency on the critical path drops throughput to ~565 TFLOPS (worse than dense).

## Shared Files

| File | Purpose |
|------|---------|
| `common.py` | WGMMA instruction selection, tile scheduler, layout helpers |
| `gluon_ws_sparse.py` | Warp-specialized sparse matmul (the "matmul" half of v11's two-kernel design) |
| `gluon_ws_dense.py` | Warp-specialized dense matmul baseline |
| `prune.py` | Reference 2:4 pruning (top-2 of every 4 elements) |
| `compress_2_4.py` | Dense → compressed sparse format conversion |

## Directory Structure

```
compression/
├── kernels/           # Landmark kernels (with symlinks to shared files)
├── dev/               # All intermediate experiments (v1–v9 + v7.8.2, benchmarks)
├── results/           # Benchmark result
│   ├── logs           # Benchmark .out logs
│   └── plots/         # Benchmark visualization PNGs
├── common.py          # Shared helpers
├── gluon_ws_sparse.py # Sparse matmul kernel
├── gluon_ws_dense.py  # Dense matmul kernel
├── prune.py           # 2:4 pruning
├── compress_2_4.py    # Compression logic
├── MLIR_DUMP/         # LLVM/MLIR lowering
├── sbatch_sh/         # Slurm job scripts
└── Profiling/         # NSight profiles
```
