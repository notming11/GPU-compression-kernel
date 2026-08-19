import os
import sys
import importlib.util
import torch
import triton
from prune import prune_2_4

# Disable forced recompilation to isolate true kernel execution speed
os.environ["TRITON_ALWAYS_COMPILE"] = "0"

def load_module_from_path(module_name, file_path):
    if not os.path.exists(file_path):
        raise ImportError(f"Could not find the physical file '{file_path}' in the current directory.")
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

def run_wgmma_benchmark():
    print("Loading modules dynamically to handle alphanumeric file limits...")
    try:
        mod_3h = load_module_from_path("mod_3h", "./3H_single_tile_no_gather_better_layout.py")
    except ImportError as e:
        print(e)
        return

    try:
        mod_3c = load_module_from_path("mod_3c", "./3C_test_wgmma_opt_layout.py")
    except ImportError as e:
        print(e)
        return

    try: 
        mod_3d = load_module_from_path("mod_3d", "./3D_single_tile_no_gather.py")
    except ImportError as e:
        print(e)
        return

    M, N, K = 64, 16, 256
    INSTR_SHAPE_N = 16

    print("\n" + "="*150)
    print(f"{'Warp Count':<12} | {'3H no gather better layout (us)':<25} | {'3D no gather (us)':<25}")
    print("="*150)

    for num_warps in [4]:
        # Generate clean experimental inputs
        A = torch.randn(M, K, device="cuda", dtype=torch.float16)
        B = torch.randn(K, N, device="cuda", dtype=torch.float16)
        C = torch.zeros((M, N), device="cuda", dtype=torch.float32)
        
        A_pruned = prune_2_4(A)
        
        # Allocate unique output tensors for safe validation
        D_3h = torch.empty((M, N), device="cuda", dtype=torch.float16)
        D_3d = torch.empty((M, N), device="cuda", dtype=torch.float16)

        # Warmup and functional parity assertion check
        try:
            mod_3h.small_mma(A_pruned, B, C, D_3h, INSTR_SHAPE_N, num_warps)
            mod_3d.small_mma(A_pruned, B, C, D_3d, INSTR_SHAPE_N, num_warps)
            torch.testing.assert_close(D_3h, D_3d, rtol=1e-3, atol=1e-1)
        except Exception as e:
            print(f"Skipping num_warps={num_warps} due to an execution error: {e}")
            # continue

        # Wrap targets in lambda functions for profiling loops
        fn_3h = lambda: mod_3h.small_mma(A_pruned, B, C, D_3h, INSTR_SHAPE_N, num_warps)
        fn_3d = lambda: mod_3d.small_mma(A_pruned, B, C, D_3d, INSTR_SHAPE_N, num_warps)

        # Profile raw device latencies
        ms_3h = triton.testing.do_bench(fn_3h)
        ms_3d = triton.testing.do_bench(fn_3d)

        # Convert milliseconds to microseconds for high-precision visibility
        us_3h = ms_3h * 1000.0
        us_3d = ms_3d * 1000.0
        # us_3t = ms_3t * 1000.0

        print(f"{num_warps:<12} | {us_3h:>25.2f} | {us_3d:>25.2f}")
    
    print("="*150)

if __name__ == "__main__":
    run_wgmma_benchmark()