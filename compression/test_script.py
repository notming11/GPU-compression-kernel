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
class Counter:
    index: gl.tensor
    phase: gl.tensor
    num_barriers: gl.constexpr
    @gluon.constexpr_function
    def __init__(self, index, phase, num_barriers):
        self.index = index
        self.phase = phase
        self.num_barriers = gl.constexpr(num_barriers)
    @gluon.jit
    def create(phase, num_barriers: gl.constexpr):
        return Counter(gl.to_tensor(0), gl.to_tensor(phase), num_barriers)
    @gluon.must_use_result
    @gluon.jit
    def next(self, pred=True):
        incr = self.index + gl.where(pred, 1, 0)
        rollover = incr == self.num_barriers
        index = gl.where(rollover, 0, incr)
        phase = gl.where(rollover, self.phase ^ 1, self.phase)
        return Counter(index, phase, self.num_barriers)
@aggregate
class CompressPartitionArgs:
    a_desc: tma.tensor_descriptor
    a_compressed_desc: tma.tensor_descriptor
    e_desc: tma.tensor_descriptor
    a_bufs: gl.shared_memory_descriptor
    a_comp_bufs: gl.shared_memory_descriptor
    e_bufs: gl.shared_memory_descriptor
    load_empty_bars: gl.shared_memory_descriptor
    load_ready_bars: gl.shared_memory_descriptor
    store_empty_bars: gl.shared_memory_descriptor
    store_ready_bars: gl.shared_memory_descriptor
    BLOCK_SIZE_M: gl.constexpr
    BLOCK_SIZE_K: gl.constexpr
    num_warps: gl.constexpr
    @gluon.constexpr_function
    def __init__(self, a_desc, a_compressed_desc, e_desc,
                 a_bufs, a_comp_bufs, e_bufs,
                 load_empty_bars, load_ready_bars,
                 store_empty_bars, store_ready_bars,
                 BLOCK_SIZE_M, BLOCK_SIZE_K, num_warps):
        self.a_desc = a_desc
        self.a_compressed_desc = a_compressed_desc
        self.e_desc = e_desc
        self.a_bufs = a_bufs
        self.a_comp_bufs = a_comp_bufs
        self.e_bufs = e_bufs
        self.load_empty_bars = load_empty_bars
        self.load_ready_bars = load_ready_bars
        self.store_empty_bars = store_empty_bars
        self.store_ready_bars = store_ready_bars
        self.BLOCK_SIZE_M = gl.constexpr(BLOCK_SIZE_M)
        self.BLOCK_SIZE_K = gl.constexpr(BLOCK_SIZE_K)
        self.num_warps = gl.constexpr(num_warps)
        
def ws_compress_load_partition(p, SchedulerImpl: gl.constexpr):
    BLOCK_SIZE_M: gl.constexpr = p.a_desc.block_type.shape[0]
    BLOCK_SIZE_K: gl.constexpr = p.a_desc.block_type.shape[1]
    state = Counter.create(1, p.load_empty_bars.shape[0])
    scheduler = SchedulerImpl.initialize(p.a_desc.shape[0], p.a_desc.shape[1], BLOCK_SIZE_M, BLOCK_SIZE_K)
    for idx in range(scheduler.get_num_tiles()):
        pid_m, pid_k = scheduler.get_tile(idx)
        off_m = pid_m * BLOCK_SIZE_M
        off_k = pid_k * BLOCK_SIZE_K
        mbarrier.wait(p.load_empty_bars.index(state.index), state.phase)
        bar = p.load_ready_bars.index(state.index)
        mbarrier.expect(bar, p.a_desc.block_type.nbytes)
        tma.async_copy_global_to_shared(p.a_desc, [off_m, off_k], bar, p.a_bufs.index(state.index))
        state = state.next()

def ws_compress_compute_partition(p, SchedulerImpl: gl.constexpr):
    load_state = Counter.create(0, p.load_empty_bars.shape[0])
    store_state = Counter.create(1, p.store_empty_bars.shape[0])
    # 1. Define Register Layout
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
    
    for _ in range(scheduler.get_num_tiles()):
        mbarrier.wait(p.load_ready_bars.index(load_state.index), load_state.phase)
        
        a_dense = p.a_bufs.index(load_state.index).load(a_reg_layout)
        mbarrier.arrive(p.load_empty_bars.index(load_state.index), count=1)
        load_state = load_state.next()
    
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
        
        mbarrier.wait(p.store_empty_bars.index(store_state.index), store_state.phase)
        # 5. Store Registers -> SMEM
        p.a_comp_bufs.index(store_state.index).store(a_compressed)
        p.e_bufs.index(store_state.index).store(meta_reordered)
        fence_async_shared()
        
        # Signal the store partition
        mbarrier.arrive(p.store_ready_bars.index(store_state.index), count=1)
        store_state = store_state.next()

def ws_compress_store_partition(p, SchedulerImpl: gl.constexpr):
    BLOCK_SIZE_M: gl.constexpr = p.a_desc.block_type.shape[0]
    BLOCK_SIZE_K: gl.constexpr = p.a_desc.block_type.shape[1]

    state = Counter.create(0, p.store_empty_bars.shape[0])
    scheduler = SchedulerImpl.initialize(p.a_desc.shape[0], p.a_desc.shape[1], BLOCK_SIZE_M, BLOCK_SIZE_K)
    
    num_buffers: gl.constexpr = p.store_empty_bars.shape[0]
    outstanding_stores: gl.constexpr = 2 * (num_buffers - 1)
    store_iter = 0
    for idx in range(scheduler.get_num_tiles()):
        pid_m, pid_k = scheduler.get_tile(idx)
        off_m = pid_m * BLOCK_SIZE_M
        off_k = pid_k * BLOCK_SIZE_K
        # Wait for compute to finish writing to SMEM
        mbarrier.wait(p.store_ready_bars.index(state.index), state.phase)
        # Async TMA Store (SMEM -> Global)
        tma.async_copy_shared_to_global(p.a_compressed_desc, [off_m, off_k // 2], p.a_comp_bufs.index(state.index))
        tma.async_copy_shared_to_global(p.e_desc, [off_m // 16, off_k], p.e_bufs.index(state.index))
        
        if store_iter >= num_buffers - 1:
            tma.store_wait(outstanding_stores)
            empty_idx = (store_iter - (num_buffers - 1)) % num_buffers
            mbarrier.arrive(p.store_empty_bars.index(empty_idx), count=1)
        state = state.next()
        store_iter += 1
    tma.store_wait(0)
@gluon.jit
def ws_tma_compress_2_4_kernel(
    a_desc, a_compressed_desc, e_desc,
    SchedulerImpl: gl.constexpr,
    M, K,
    BLOCK_SIZE_M: gl.constexpr, BLOCK_SIZE_K: gl.constexpr,
    num_buffers: gl.constexpr,
    num_warps: gl.constexpr,
):  
    # 1. Allocate SMEM
    a_bufs = gl.allocate_shared_memory(gl.float16, [num_buffers, BLOCK_SIZE_M, BLOCK_SIZE_K], a_desc.layout)
    a_comp_bufs = gl.allocate_shared_memory(gl.float16, [num_buffers, BLOCK_SIZE_M, BLOCK_SIZE_K // 2], a_compressed_desc.layout)
    e_bufs = gl.allocate_shared_memory(gl.int16, [num_buffers, BLOCK_SIZE_M // 16, BLOCK_SIZE_K], e_desc.layout)
    
    # 2. Setup MBarriers
    load_empty_bars = gl.allocate_shared_memory(gl.int64, [num_buffers, 1], mbarrier.MBarrierLayout())
    load_ready_bars = gl.allocate_shared_memory(gl.int64, [num_buffers, 1], mbarrier.MBarrierLayout())
    store_empty_bars = gl.allocate_shared_memory(gl.int64, [num_buffers, 1], mbarrier.MBarrierLayout())
    store_ready_bars = gl.allocate_shared_memory(gl.int64, [num_buffers, 1], mbarrier.MBarrierLayout())
    
    for i in gl.static_range(num_buffers):
        mbarrier.init(load_empty_bars.index(i), count=1)
        mbarrier.init(load_ready_bars.index(i), count=1)
        mbarrier.init(store_empty_bars.index(i), count=1)
        mbarrier.init(store_ready_bars.index(i), count=1)
    
    # 3. Create Shared State
    p = CompressPartitionArgs(
        a_desc, a_compressed_desc, e_desc,
        a_bufs, a_comp_bufs, e_bufs,
        load_empty_bars, load_ready_bars,
        store_empty_bars, store_ready_bars,
        BLOCK_SIZE_M, BLOCK_SIZE_K, num_warps
    )
    # 4. Launch Warp-Specialized Partitions
    # Allocates the primary warps to compute, 1 warp to load, and 1 warp to store
    gl.warp_specialize([
        (ws_compress_compute_partition, (p, SchedulerImpl)),
        (ws_compress_load_partition, (p, SchedulerImpl)),
        (ws_compress_store_partition, (p, SchedulerImpl))
    ], [1, 1], [24, 24])
    
def compress_get_configs(pre_hook=None):
    def valid(BM, BK, warps, buffers):
        smem_bytes = 2 * (
            (buffers * BM * BK) +
            (buffers * BM * (BK // 2)) +
            (buffers * (BM // 16) * BK)
        ) + (32 * buffers) + 32
        if smem_bytes > 232448:
            return False
        return True
    return [
        triton.Config(
            {
                "BLOCK_SIZE_M" : BM,
                "BLOCK_SIZE_K" : BK,
                "num_buffers": buffers,
            },
            num_warps=warps,
            pre_hook=pre_hook
        )
        for BM in (64, 128, 256,)
        for BK in (64, 128, 256,)
        for warps in (4, 8, 16)
        for buffers in (3, 4, 5, 6, 7)
        if valid(BM, BK, warps, buffers)
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
    a_compressed = torch.empty((M,K//2), device=A.device, dtype=torch.float16)
    e = torch.empty((M//16,K), device=A.device, dtype=torch.int16)
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
        def grid_prune(meta):
            num_sms = torch.cuda.get_device_properties("cuda").multi_processor_count
            num_pid = triton.cdiv(M, meta["BLOCK_SIZE_M"]) * triton.cdiv(K, meta["BLOCK_SIZE_K"])
            return (min(num_sms, num_pid), )
        compress_2_4_autotune[grid_prune](
            a_desc, a_compressed_desc, e_desc,
            GroupedPersistentTileScheduler(8),
            M, K
        )
        
        # print("Done pruning")
        # return a_compressed, e
        
        cache_size_bytes = 256 * 1024 * 1024 
    
        # Allocate and write zeros to force eviction of existing L2 data
        dummy = torch.empty(cache_size_bytes, dtype=torch.int8, device="cuda")
        dummy.zero_()
        
        # Let the autotuner handle everything
        def grid_matmul(meta):
            num_sms = torch.cuda.get_device_properties("cuda").multi_processor_count
            num_pid = triton.cdiv(M, meta["BLOCK_SIZE_M"]) * triton.cdiv(N, meta["BLOCK_SIZE_N"])
            return (min(num_sms, num_pid), )
        
        sparse_ws_kernel_autotune_trimmed[grid_matmul](
            a_compressed_desc, e_desc, b_desc, c_desc, GroupedPersistentTileScheduler(8),
            M, N, K
        )
    else:
        # 1. Prepare kwargs for the TMA hook for compression
        compress_hook_kwargs = {
            "BLOCK_SIZE_M": manual_config["BM"],
            "BLOCK_SIZE_K": manual_config["BK"],
            "a_desc": a_desc, "a_compressed_desc": a_compressed_desc, "e_desc": e_desc
        }
        
        # 2. Mutate the descriptors manually for compression
        compress_tma_set_block_size_hook(compress_hook_kwargs)
        
        # 3. Calculate grid using manual config for compression
        num_sms = torch.cuda.get_device_properties("cuda").multi_processor_count
        num_pid_compress = triton.cdiv(M, manual_config["BM"]) * triton.cdiv(K, manual_config["BK"])
        grid_compress = (min(num_sms, num_pid_compress), )
        
        # 4. Launch the base compression kernel directly
        ws_tma_compress_2_4_kernel[grid_compress](
            a_desc, a_compressed_desc, e_desc,
            GroupedPersistentTileScheduler(8),
            M, K,
            BLOCK_SIZE_M=manual_config["BM"],
            BLOCK_SIZE_K=manual_config["BK"],
            num_buffers=manual_config["buffers"],
            num_warps=manual_config["warps"]
        )
        # 5. Prepare kwargs for the TMA hook for sparse matmul
        hook_kwargs = {
            "BLOCK_SIZE_M": manual_config["BM"],
            "BLOCK_SIZE_N": manual_config["BN"],
            "BLOCK_SIZE_K": manual_config["BK"],
            "SUBTILE_FACTOR": manual_config["SF"],
            "a_desc": a_compressed_desc, "e_desc": e_desc, "b_desc": b_desc, "c_desc": c_desc
        }
        
        # 6. Mutate the descriptors manually for sparse matmul
        sparse_matmul_tma_set_block_size_hook(hook_kwargs)
        
        # 7. Calculate grid using manual config for sparse matmul
        num_sms = torch.cuda.get_device_properties("cuda").multi_processor_count
        num_pid = triton.cdiv(M, manual_config["BM"]) * triton.cdiv(N, manual_config["BN"])
        grid = (min(num_sms, num_pid), )
        num_pid_matmul = triton.cdiv(M, manual_config["BM"]) * triton.cdiv(N, manual_config["BN"])
        grid_matmul = (min(num_sms, num_pid_matmul), )
        
        # 8. Launch the base matmul kernel directly (bypassing autotune)
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