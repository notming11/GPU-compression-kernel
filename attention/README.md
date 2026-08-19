# Attention Kernels (WIP)

FlashAttention-3 style forward pass kernels with warp specialization for H100, written in Triton/Gluon.

## Active Kernels (`kernels/`)

### `gluon_attention_pingpong_overlap.py`
Main active kernel. Implements ping-pong overlapping of Q·K^T and softmax·V WGMMA stages with warp-specialized load/compute partitions. Uses persistent tile scheduling over (batch, head, seq) dimensions.

### `gluon_attention_forward.py`
Simpler forward-pass variant without ping-pong overlap. Used as the performance baseline.

## Development Files (`dev/`)

| File | Description |
|------|-------------|
| `gluon_3_partition_pingpong.py` | 3-partition ping-pong experiment |
| `fused-attention-ws-device-tma-hopper-or-blackwell.py` | Reference WS attention from Meta's Triton version |
| `gluon_fa3_forward.py` | 4-partition experiment without ping-pong |

## Baseline Benchmarks

FA3 baseline results and throughput plots are in [`results/`](results/).

## Shared Files

| File | Purpose |
|------|---------|
| `common.py` | WGMMA instruction selection and layout helpers (hardlinked to `compression/common.py`) |
| `benchmark.py` | Benchmark harness |
| `pytorch_sdpa.py` | PyTorch SDPA reference for benchmarking |

## Directory Structure

```
attention/
├── kernels/           # Active attention kernels (with symlinks to shared files)
├── dev/               # Experimental variants
├── results/           # Benchmark .out logs + plots
├── common.py          # → hardlink to compression/common.py
├── benchmark.py       # Benchmark harness
├── MLIR_DUMP          # MLIR/LLVM lowerings
├── sbatch_sh/         # Slurm job scripts
└── Profiling/         # NSight profiles
```
