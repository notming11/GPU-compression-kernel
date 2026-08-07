import argparse
import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.nvidia.hopper import TensorDescriptor
from triton.language.core import _aggregate as aggregate
from triton.experimental.gluon.language.nvidia.hopper import (
    tma,
    mbarrier,
    fence_async_shared,
)
from common import (
    WGMMA,
    GroupedPersistentTileScheduler
)
from gluon_ws_sparse import (
    sparse_ws_kernel_autotune_trimmed,
    sparse_matmul_warp_specialized_kernel,
    sparse_matmul_tma_set_block_size_hook,
)
from triton.language.core import _aggregate as aggregate

@aggregate
class CompressPartitionArgs:
    a_desc: tma.tensor_descriptor
    a_compressed_desc: tma.tensor_descriptor
    e_desc: tma.tensor_descriptor
    a_smem: gl.shared_memory_descriptor
    a_comp_smem: gl.shared_memory_descriptor
    e_smem: gl.shared_memory_descriptor
    load_ready_bar: gl.shared_memory_descriptor
    compute_ready_bar: gl.shared_memory_descriptor
    BLOCK_SIZE_M: gl.constexpr
    BLOCK_SIZE_K: gl.constexpr
    num_warps: gl.constexpr

    @gluon.constexpr_function
    def __init__(self, a_desc, a_compressed_desc, e_desc,
                 a_smem, a_comp_smem, e_smem,
                 load_ready_bar, compute_ready_bar,
                 BLOCK_SIZE_M, BLOCK_SIZE_K, num_warps):
        self.a_desc = a_desc
        self.a_compressed_desc = a_compressed_desc
        self.e_desc = e_desc
        self.a_smem = a_smem
        self.a_comp_smem = a_comp_smem
        self.e_smem = e_smem
        self.load_ready_bar = load_ready_bar
        self.compute_ready_bar = compute_ready_bar
        self.BLOCK_SIZE_M = gl.constexpr(BLOCK_SIZE_M)
        self.BLOCK_SIZE_K = gl.constexpr(BLOCK_SIZE_K)
        self.num_warps = gl.constexpr(num_warps)

@gluon.jit
def ws_compress_load_partition(p):
    pid_m = gl.program_id(0)
    pid_k = gl.program_id(1)
    off_m = pid_m * p.BLOCK_SIZE_M
    off_k = pid_k * p.BLOCK_SIZE_K

    # Async TMA Load (Global -> SMEM)
    mbarrier.expect(p.load_ready_bar, p.a_desc.block_type.nbytes)
    tma.async_copy_global_to_shared(p.a_desc, [off_m, off_k], p.load_ready_bar, p.a_smem)

@gluon.jit
def ws_compress_compute_partition(p):
    # 1. Wait for TMA load to finish
    mbarrier.wait(p.load_ready_bar, 0)

    # 2. Define Register Layout
    if p.num_warps == 4:
        warp_bases: gl.constexpr = [[16, 0], [32, 0]]
    elif p.num_warps == 8:
        warp_bases: gl.constexpr = [[16, 0], [32, 0], [64, 0]]
    else:
        warp_bases: gl.constexpr = [[16, 0], [32, 0], [64, 0], [128, 0]]
        
    a_reg_layout: gl.constexpr = gl.DistributedLinearLayout(
        reg_bases=[[0, 1], [0, 2], [8, 0], [0, 4], [0, 8]],
        lane_bases=[[0, 16], [0, 32], [1, 0], [2, 0], [4, 0]],
        warp_bases=warp_bases,
        block_bases=[],
        shape=[16 * p.num_warps, 64],
    )
    
    a_dense = p.a_smem.load(a_reg_layout)
    a_grouped = a_dense.reshape(p.BLOCK_SIZE_M, p.BLOCK_SIZE_K // 4, 2, 2)
    a_even, a_odd = a_grouped.split()

    a0, a2 = a_even.split()
    a1, a3 = a_odd.split()

    # 3. Prune 2:4 (select top 2 values algebraically)
    c01 = a0 > a1
    c02 = a0 > a2
    c03 = a0 > a3
    c12 = a1 > a2
    c13 = a1 > a3
    c23 = a2 > a3
 
    c10 = ~c01
    c20 = ~c02
    c21 = ~c12

    b0_bool = (c01 & (c02 | c03)) | (c02 & c03)
    b1_bool = (c10 & (c12 | c13)) | (c12 & c13)
    b2_bool = (c20 & (c21 | c23)) | (c21 & c23)

    nz0 = gl.where(b0_bool, a0, gl.where(b1_bool, a1, a2))
    nz1 = gl.where(b0_bool & b1_bool, a1, gl.where(b2_bool & (b0_bool | b1_bool), a2, a3))

    a_compressed = gl.join(nz0, nz1).reshape(p.BLOCK_SIZE_M, p.BLOCK_SIZE_K // 2)

    meta_4 = gl.where(b0_bool,
         gl.where(b1_bool, 4, gl.where(b2_bool, 8, 12)),
         gl.where(b1_bool, gl.where(b2_bool, 9, 13), 14))

    # 4. Pack metadata
    meta_4_reshaped = meta_4.reshape(p.BLOCK_SIZE_M // 16, 2, 8, p.BLOCK_SIZE_K // 64, 4, 2, 2)
    meta_4_permuted = meta_4_reshaped.permute(0, 3, 2, 4, 1, 5, 6)
    meta_4_ready = meta_4_permuted.reshape(p.BLOCK_SIZE_M // 16, p.BLOCK_SIZE_K, 2, 2)

    meta_even, meta_odd = meta_4_ready.split()
    mn0, mn2 = meta_even.split()
    mn1, mn3 = meta_odd.split()

    meta_reordered = gl.inline_asm_elementwise(
        asm="""
        {
        .reg .b32 t1, t2, t3;
        shl.b32 t1, $2, 4;
        shl.b32 t2, $3, 8;
        shl.b32 t3, $4, 12;
        or.b32 $0, $1, t1;
        or.b32 $0, $0, t2;
        or.b32 $0, $0, t3;
        }
        """,
        constraints="=r,r,r,r,r",
        args=[mn0, mn1, mn2, mn3],
        dtype=gl.int16,
        is_pure=True,
        pack=1,
    )
    
    # 5. Store Registers -> SMEM
    p.a_comp_smem.store(a_compressed)
    p.e_smem.store(meta_reordered)
    fence_async_shared()
    
    # Signal the store partition
    mbarrier.arrive(p.compute_ready_bar, count=1)

@gluon.jit
def ws_compress_store_partition(p):
    pid_m = gl.program_id(0)
    pid_k = gl.program_id(1)
    off_m = pid_m * p.BLOCK_SIZE_M
    off_k = pid_k * p.BLOCK_SIZE_K

    # Wait for compute to finish writing to SMEM
    mbarrier.wait(p.compute_ready_bar, 0)

    # Async TMA Store (SMEM -> Global)
    tma.async_copy_shared_to_global(p.a_compressed_desc, [off_m, off_k // 2], p.a_comp_smem)
    tma.async_copy_shared_to_global(p.e_desc, [off_m // 16, off_k], p.e_smem)
    
    tma.store_wait(0)

@gluon.jit
def ws_tma_compress_2_4_kernel(
    a_desc, a_compressed_desc, e_desc,
    M, K,
    BLOCK_SIZE_M: gl.constexpr, BLOCK_SIZE_K: gl.constexpr,
    num_warps: gl.constexpr,
):  
    # 1. Allocate SMEM
    a_smem = gl.allocate_shared_memory(gl.float16, [BLOCK_SIZE_M, BLOCK_SIZE_K], a_desc.layout)
    a_comp_smem = gl.allocate_shared_memory(gl.float16, [BLOCK_SIZE_M, BLOCK_SIZE_K // 2], a_compressed_desc.layout)
    e_smem = gl.allocate_shared_memory(gl.int16, [BLOCK_SIZE_M // 16, BLOCK_SIZE_K], e_desc.layout)
    
    # 2. Setup MBarriers
    load_ready_bar = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
    compute_ready_bar = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
    
    mbarrier.init(load_ready_bar, count=1)
    mbarrier.init(compute_ready_bar, count=1)
    
    # 3. Create Shared State
    p = CompressPartitionArgs(
        a_desc, a_compressed_desc, e_desc,
        a_smem, a_comp_smem, e_smem,
        load_ready_bar, compute_ready_bar,
        BLOCK_SIZE_M, BLOCK_SIZE_K, num_warps
    )

    # 4. Launch Warp-Specialized Partitions
    gl.warp_specialize([
        (ws_compress_compute_partition, (p,)),
        (ws_compress_load_partition, (p,)),
        (ws_compress_store_partition, (p,))
    ], [1, 1], [24, 24])
    
def compress_get_configs(pre_hook=None):
    def valid(BM, BK, warps):
        # Calculate SMEM bytes for unbuffered single-stage
        # a_smem (dense) + a_comp_smem (compressed) + e_smem (metadata)
        smem_bytes = 2 * (
            (BM * BK) +
            (BM * (BK // 2)) +
            ((BM // 16) * BK)
        ) + 64  # 32 bytes each for load and compute mbarriers
        
        if smem_bytes > 232448:
            return False
        return True

    return [
        triton.Config(
            {
                "BLOCK_SIZE_M" : BM,
                "BLOCK_SIZE_K" : BK,
            },
            num_warps=warps,
            pre_hook=pre_hook
        )
        for BM in (64, 128, 256,)
        for BK in (64, 128, 256,)
        for warps in (4, 8)
        if valid(BM, BK, warps)
    ]
    
def compress_tma_set_block_size_hook(nargs):
    block_m = nargs["BLOCK_SIZE_M"]
    block_k = nargs["BLOCK_SIZE_K"]
    # Update TMA block shapes for the current config
    nargs["a_desc"].block_shape = [block_m, block_k]
    nargs["a_compressed_desc"].block_shape = [block_m, block_k // 2]
    nargs["e_desc"].block_shape = [block_m // 16, block_k]
    # Update TMA layouts for the current config
    nargs["a_desc"].layout = gl.NVMMASharedLayout.get_default_for(nargs["a_desc"].block_shape, gl.float16)
    nargs["a_compressed_desc"].layout = gl.NVMMASharedLayout.get_default_for(nargs["a_compressed_desc"].block_shape, gl.float16)
    nargs["e_desc"].layout = gl.NVMMASharedLayout.get_default_for(nargs["e_desc"].block_shape, gl.int16)

compress_2_4_autotune = triton.autotune(
    configs=compress_get_configs(
        pre_hook=compress_tma_set_block_size_hook
    ),
    key=["M", "K"],
    do_bench = lambda kernel_call, quantiles: triton.testing.do_bench_cudagraph(
        kernel_call, rep=100, quantiles=quantiles
    ),
)(ws_tma_compress_2_4_kernel)

def run_2_kernel_ws_matmul(A, B, tune=True, manual_config=None):
    M, N, K = A.shape[0], B.shape[1], B.shape[0]
    a_compressed = torch.empty((M, K // 2), device=A.device, dtype=torch.float16)
    e = torch.empty((M // 16, K), device=A.device, dtype=torch.int16)
    c = torch.empty((M, N), device=A.device, dtype=torch.float16)
    
    dummy_block = [1, 1]
    dummy_layout_f16 = gl.NVMMASharedLayout.get_default_for(dummy_block, gl.float16)
    dummy_layout_i16 = gl.NVMMASharedLayout.get_default_for(dummy_block, gl.int16)
    a_desc = TensorDescriptor.from_tensor(A, dummy_block, dummy_layout_f16)
    a_compressed_desc = TensorDescriptor.from_tensor(a_compressed, dummy_block, dummy_layout_f16)
    e_desc = TensorDescriptor.from_tensor(e, dummy_block, dummy_layout_i16)
    b_desc = TensorDescriptor.from_tensor(B, dummy_block, dummy_layout_f16)
    c_desc = TensorDescriptor.from_tensor(c, dummy_block, dummy_layout_f16)

    if tune:
        # Standard 2D grid for the unbuffered, non-persistent compression kernel
        def grid_prune(meta):
            return (triton.cdiv(M, meta["BLOCK_SIZE_M"]), triton.cdiv(K, meta["BLOCK_SIZE_K"]))
            
        compress_2_4_autotune[grid_prune](
            a_desc, a_compressed_desc, e_desc,
            M, K
        )
        
        cache_size_bytes = 256 * 1024 * 1024 
    
        # Allocate and write zeros to force eviction of existing L2 data
        dummy = torch.empty(cache_size_bytes, dtype=torch.int8, device="cuda")
        dummy.zero_()
        
        # 1D SM-bound grid for the persistent matmul kernel
        def grid_matmul(meta):
            num_sms = torch.cuda.get_device_properties("cuda").multi_processor_count
            num_pid = triton.cdiv(M, meta["BLOCK_SIZE_M"]) * triton.cdiv(N, meta["BLOCK_SIZE_N"])
            return (min(num_sms, num_pid), )
        
        sparse_ws_kernel_autotune_trimmed[grid_matmul](
            a_compressed_desc, e_desc, b_desc, c_desc, GroupedPersistentTileScheduler(8),
            M, N, K
        )
    else:
        # 1. Prepare kwargs for the TMA hook for compression (no num_buffers)
        compress_hook_kwargs = {
            "BLOCK_SIZE_M": manual_config["BM"],
            "BLOCK_SIZE_K": manual_config["BK"],
            "a_desc": a_desc, "a_compressed_desc": a_compressed_desc, "e_desc": e_desc
        }
        
        # 2. Mutate the descriptors manually for compression
        compress_tma_set_block_size_hook(compress_hook_kwargs)
        
        # 3. Calculate standard 2D grid for unbuffered compression
        grid_compress = (triton.cdiv(M, manual_config["BM"]), triton.cdiv(K, manual_config["BK"]))
        
        # 4. Launch the base compression kernel directly
        ws_tma_compress_2_4_kernel[grid_compress](
            a_desc, a_compressed_desc, e_desc,
            M, K,
            BLOCK_SIZE_M=manual_config["BM"],
            BLOCK_SIZE_K=manual_config["BK"],
            num_warps=manual_config["warps"]
        )
        
        # 5. Prepare kwargs for the TMA hook for sparse matmul (kept identical)
        hook_kwargs = {
            "BLOCK_SIZE_M": manual_config["BM"],
            "BLOCK_SIZE_N": manual_config["BN"],
            "BLOCK_SIZE_K": manual_config["BK"],
            "SUBTILE_FACTOR": manual_config["SF"],
            "a_desc": a_compressed_desc, "e_desc": e_desc, "b_desc": b_desc, "c_desc": c_desc
        }
        
        # 6. Mutate the descriptors manually for sparse matmul
        sparse_matmul_tma_set_block_size_hook(hook_kwargs)
        
        # 7. Calculate 1D grid using manual config for persistent sparse matmul
        num_sms = torch.cuda.get_device_properties("cuda").multi_processor_count
        num_pid_matmul = triton.cdiv(M, manual_config["BM"]) * triton.cdiv(N, manual_config["BN"])
        grid_matmul = (min(num_sms, num_pid_matmul), )
        
        # 8. Launch the base matmul kernel directly
        sparse_matmul_warp_specialized_kernel[grid_matmul](
            a_compressed_desc, e_desc, b_desc, c_desc, GroupedPersistentTileScheduler(8),
            M, N, K,
            BLOCK_SIZE_M=manual_config["BM"], 
            BLOCK_SIZE_N=manual_config["BN"], 
            BLOCK_SIZE_K=manual_config["BK"],
            num_buffers=manual_config["buffers"], 
            SUBTILE_FACTOR=manual_config["SF"], 
            num_warps=manual_config["warps"]
        )
        
    return c

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Sparse Warp-Specialized Matmul")
    parser.add_argument("--tune", action="store_true", help="Enable Triton autotuning")
    
    # Manual config arguments (ignored if --tune is passed)
    parser.add_argument("--bm", type=int, default=128, help="BLOCK_SIZE_M")
    parser.add_argument("--bn", type=int, default=256, help="BLOCK_SIZE_N")
    parser.add_argument("--bk", type=int, default=64, help="BLOCK_SIZE_K")
    parser.add_argument("--warps", type=int, default=8, help="Number of warps")
    parser.add_argument("--buffers", type=int, default=5, help="Number of buffers")
    parser.add_argument("--sf", type=int, default=8, help="SUBTILE_FACTOR")
    
    args = parser.parse_args()
    manual_config = {
        "BM": args.bm,
        "BN": args.bn,
        "BK": args.bk,
        "warps": args.warps,
        "buffers": args.buffers,
        "SF": args.sf
    }
    if args.tune:
        print("Running sparse matmul. Autotuning enabled.")
    else:
        print(f"Running sparse matmul with manual config: {manual_config}")
    sizes = [
        (49152, 8192, 49152)
    ]
    from compress_2_4 import *
    from prune import *
    for M, N, K in sizes:
        A = torch.randn(M, K, device="cuda", dtype=torch.float16)
        B = torch.randn((K, N), device="cuda", dtype=torch.float16)
        C = torch.empty(M, N, device="cuda", dtype=torch.float16)
        A_pruned = prune_2_4(A)
        A_compressed, E = compress_dense_to_sparse(A_pruned)
        E = E.view(M // 16, K)
        
        D = run_2_kernel_ws_matmul(A, B, tune=args.tune, manual_config=manual_config)
        # a_comp, e_comp = run_2_kernel_ws_matmul(A, B, tune=args.tune, manual_config=manual_config)
        
        torch.testing.assert_close(A_pruned @ B, D, rtol=1e-3, atol=1e-1)
        # torch.testing.assert_close(e_comp, E, rtol=1e-3, atol=1e-1)
        # torch.testing.assert_close(a_comp, A_compressed, rtol=1e-3, atol=1e-1)
    
    print("Done sparse.")