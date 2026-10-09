# Attention Kernels (WIP)

FlashAttention-3 style forward pass kernels with warp specialization (TMA async loads, WGMMA compute, softmax/store partitions) for NVIDIA Hopper (H100), written in Triton/Gluon.

## Active Kernels (`kernels/`)

### `gluon_attention_pingpong_overlap.py`
Main active dense FlashAttention-3 kernel. Implements a 4-partition warp-specialized design (`fa3_producer_partition`, `fa3_consumer_wg0`, `fa3_consumer_wg1`, `fa3_store_partition`) with ping-pong overlapping of GEMM-I ($Q K^T$) and GEMM-II ($P V$) across two compute warp groups, alongside persistent tile scheduling over $(B \times H, \text{seq})$ dimensions.

### `gluon_attention_qk_sparse.py`
Sparse FlashAttention kernel exploring 2:4 structured sparsity on the first GEMM ($Q K^T$). Uses `sparsifier.py` to prune and pack query tiles ($Q$) into compressed 2:4 sparse format with metadata, followed by sparse WGMMA against dense $K^T$ and dense attention with $V$.

### `gluon_attention_qkv_sparse.py`
Full 2:4 sparse FlashAttention kernel incorporating sparse $Q K^T$ alongside dynamic online 2:4 pruning and compression of intermediate attention scores ($S / P$) for sparse $P V$ WGMMA.

### `sparsifier.py`
Persistent warp-specialized TMA kernel for online 2:4 pruning, metadata encoding, and sparse packing of activation tensors.

## Development Files (`dev/`)

| File | Description |
|------|-------------|
| `gluon_attention_forward.py` | Simpler forward-pass baseline without ping-pong overlap |
| `gluon_3_partition_pingpong.py` | 3-partition ping-pong experiment |
| `gluon_attention_alu_xu_pipeline.py` | Pipelined ALU / compute overlap experiment |
| `gluon_no_store_partition.py` | Variant evaluating performance without a dedicated store partition |
| `gluon_fa3_forward.py` | 4-partition forward pass without ping-pong overlap |
| `tensor_bound_proof.py` | Layout and tensor boundary verification tests |

## Benchmarks & Results

- **Dense Baseline Benchmarks**: Log in [`results/logs/FA3_baseline.txt`](results/logs/FA3_baseline.txt), comparison plots in [`results/plots/`](results/plots/).
- **Sparse Attention Benchmarks**:
  - QK sparse logs: [`results/logs/FA3_qk_sparse.txt`](results/logs/FA3_qk_sparse.txt)
  - QKV sparse logs: [`results/logs/FA3_qkv_sparse.txt`](results/logs/FA3_qkv_sparse.txt)
  - Sparse throughput comparison plots: [`results/plots/FA3_Benchmark_sparse_HEAD_DIM_*.png`](results/plots/)
- **Harness Scripts**:
  - [`benchmark.py`](benchmark.py): Evaluates dense FA3 vs. PyTorch SDPA across sequence lengths and head dimensions.
  - [`benchmark_sparse.py`](benchmark_sparse.py): Benchmarks dense FA3 against 2:4 sparse attention kernels.

## Shared Files

| File | Purpose |
|------|---------|
| `common.py` | WGMMA instruction selection and layout helpers (hardlinked to `compression/common.py`) |
| `pytorch_sdpa.py` | PyTorch SDPA reference implementation for verification and benchmarking |
| `sparsifier.py` | Standalone and autotuned 2:4 compression routines |

## Directory Structure

```
attention/
├── kernels/           # Active kernels (dense ping-pong, qk_sparse, qkv_sparse, sparsifier)
├── dev/               # Experimental variants (3-partition, no-store, alu-pipeline)
├── results/           # Benchmark logs (.txt) and comparison curves (.png)
│   ├── logs/          # FA3_baseline.txt, FA3_qk_sparse.txt, FA3_qkv_sparse.txt
│   └── plots/         # Dense and sparse benchmark comparison curves
├── sbatch_sh/         # Slurm batch scripts for automated execution
├── Profiling/         # NCU profile reports (dense & sparse)
│   ├── dense/         # Profiles for dense FA3 & PyTorch SDPA
│   └── sparse/        # Profiles for sparse variants (Q, S, P)
├── MLIR_DUMP/         # Triton MLIR / LLVM lowering dumps
├── benchmark.py       # Dense FA3 benchmark harness vs. PyTorch SDPA
├── benchmark_sparse.py# Sparse FA3 benchmark harness
├── common.py          # WGMMA helpers & layout generators
└── pytorch_sdpa.py    # Reference PyTorch SDPA benchmark
```
