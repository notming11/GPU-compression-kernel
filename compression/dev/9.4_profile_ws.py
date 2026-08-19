import os
import torch
import torch.cuda.nvtx as nvtx
import importlib
import sys

from prune import prune_2_4
from compress_2_4 import compress_dense_to_sparse

# Set environments before importing Triton
os.environ["MLIR_ENABLE_DUMP"] = "1"
os.environ["MLIR_DUMP_PATH"] = "./MLIR_DUMP/7.6"
os.environ["TRITON_CACHE_DIR"] = "./compiler_scratch/.triton_cache"

module_name = "compression_ws"
file_path = "gluon_ws_sparse.py"

# Load the file dynamically
spec = importlib.util.spec_from_file_location(module_name, file_path)
compression_module = importlib.util.module_from_spec(spec)
sys.modules[module_name] = compression_module
spec.loader.exec_module(compression_module)

# Extract the functions you need into the local namespace
run_sparse_ws_matmul = compression_module.run_sparse_ws_matmul
# prune_2_4 = compression_module.prune_2_4

if __name__ == "__main__":
    M, N, K = 49152, 4096, 49152
    print(f"Profiling M={M}, N={N}, K={K}...", flush=True)

    A = torch.randn(M, K, device="cuda", dtype=torch.float16)
    B = torch.randn((K, N), device="cuda", dtype=torch.float16)
    A_pruned = prune_2_4(A)
    A, E = compress_dense_to_sparse(A_pruned)
    E = E.view(M // 16, K)

    # 1. Warmup: Triton autotunes here. 
    # Ncu will see this if you don't use the --nvtx-include flag, 
    # but we will filter it out in the bash command.
    print("Warming up autotuner...", flush=True)
    for _ in range(3):
        _ = run_sparse_ws_matmul(A_pruned, E, B)
    torch.cuda.synchronize()

    # 2. Profile Target
    print("Running targeted profile...", flush=True)
    nvtx.range_push("isolated_wgmma_kernel")
    
    C = run_sparse_ws_matmul(A_pruned, E, B)
    torch.cuda.synchronize()
    
    nvtx.range_pop()
    print("Done.")