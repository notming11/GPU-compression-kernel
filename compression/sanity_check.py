import torch
import triton
from typing import Union
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.language.core import _aggregate as aggregate
import os

from prune import prune_2_4

from triton.experimental.gluon.nvidia.hopper import TensorDescriptor
from triton.experimental.gluon.language.nvidia.hopper import (
    tma,
    mbarrier,
    fence_async_shared,
)


@aggregate
class PersistentTileScheduler:
    pid_start: gl.tensor
    pid_end: gl.tensor
    num_pid_m: gl.tensor

    @gluon.constexpr_function
    def __init__(self, pid_start, pid_end, num_pid_m):
        self.pid_start = pid_start
        self.pid_end = pid_end
        self.num_pid_m = num_pid_m

    @gluon.jit
    def initialize(M, N, BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr):
        kernel_id = gl.program_id(axis=0)
        num_kernels = gl.num_programs(axis=0)
        num_pid_m = gl.cdiv(M, BLOCK_M)
        num_pid_n = gl.cdiv(N, BLOCK_N)
        num_pid = num_pid_m * num_pid_n
        pid_per_kernel = gl.cdiv(num_pid, num_kernels)
        pid_start = kernel_id * pid_per_kernel
        pid_end = min(pid_start + pid_per_kernel, num_pid)
        return PersistentTileScheduler(pid_start, pid_end, num_pid_m)

    @gluon.jit
    def get_num_tiles(self):
        return self.pid_end - self.pid_start

    @gluon.jit
    def get_tile(self, idx):
        pid = self.pid_start + idx
        pid_m = pid % self.num_pid_m
        pid_n = pid // self.num_pid_m
        return pid_m, pid_n


@gluon.jit
def issue_sparse_loads_stealb(
    producer,
    a_pruned_desc,
    b_desc,
    off_m,
    off_n,
    k,
    bars,
    a_pruned_bufs,
    b_bufs,
    stealb: gl.constexpr,
    num_buffers: gl.constexpr,
    pred=True,
):
    index = producer % num_buffers
    b_index = producer % (num_buffers + stealb)
    producer += 1
    bar = bars.index(index)
    mbarrier.expect(
        bar,
        a_pruned_desc.block_type.nbytes + b_desc.block_type.nbytes,
        pred,
    )
    tma.async_copy_global_to_shared(
        a_pruned_desc, [off_m, k], bar, a_pruned_bufs.index(index), pred
    )
    tma.async_copy_global_to_shared(
        b_desc, [k, off_n], bar, b_bufs.index(b_index), pred
    )
    return producer


@gluon.jit
def consume_loads_only(
    consumer,
    bars,
    a_pruned_bufs,
    a_pruned_reg_layout: gl.constexpr,
    a_intermediate_layout: gl.constexpr,
    num_buffers: gl.constexpr,
):
    index = consumer % num_buffers
    phase = consumer // num_buffers & 1
    consumer += 1
    
    # Wait for TMA to finish loading into SMEM
    mbarrier.wait(bars.index(index), phase)
    
    # 1. Load from Shared Memory to Registers using your layout
    a_reg = a_pruned_bufs.index(index).load(a_pruned_reg_layout)
    
    # 2. Simple arithmetic operation in registers
    a_reg = a_reg + 1.0
    
    # 3. Convert to intermediate layout (simulating your grouping/reshaping)
    # a_reg = gl.convert_layout(a_reg, a_intermediate_layout)
    
    # 4. Store from Registers back to Shared Memory
    a_pruned_bufs.index(index).store(a_reg)
    
    return consumer


@gluon.jit
def run_sparse_runtime_matmul(
    a_pruned_desc,
    b_desc,
    c_desc,
    SchedulerImpl: gl.constexpr,
    M, N, K,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    BLOCK_K: gl.constexpr,
    num_buffers: gl.constexpr,
    STEALB: gl.constexpr,
    num_warps: gl.constexpr,
):
    dtype: gl.constexpr = a_pruned_desc.dtype
    K = a_pruned_desc.shape[1]

    gl.static_assert(num_buffers >= 3, "expected at least 3 buffers")
    a_pruned_bufs = gl.allocate_shared_memory(
        dtype, [num_buffers] + a_pruned_desc.block_type.shape, a_pruned_desc.layout
    )
    b_bufs = gl.allocate_shared_memory(
        dtype, [num_buffers + STEALB] + b_desc.block_type.shape, b_desc.layout
    )
    
    if not STEALB:
        c_smem = gl.allocate_shared_memory(
            dtype, c_desc.block_type.shape, c_desc.layout
        )
    else:
        gl.static_assert(
            2 * BLOCK_N * BLOCK_K >= BLOCK_M * BLOCK_N,
            "B tile not large enough to steal",
        )
        
    bars = gl.allocate_shared_memory(
        gl.int64, [num_buffers, 1], mbarrier.MBarrierLayout()
    )
    
    for i in gl.static_range(num_buffers):
        mbarrier.init(bars.index(i), count=1)
        
    producer = 0
    consumer = 0

    scheduler = SchedulerImpl.initialize(
        c_desc.shape[0], c_desc.shape[1], BLOCK_M, BLOCK_N
    )
    num_tiles = scheduler.get_num_tiles()

    if num_warps == 4:
        a_warp_bases: gl.constexpr = [[16, 0], [32, 0]]
    elif num_warps == 8:
        a_warp_bases: gl.constexpr = [[16, 0], [32, 0], [64, 0]]
    elif num_warps == 16:
        a_warp_bases: gl.constexpr = [[16, 0], [32, 0], [64, 0], [128, 0]]
    
    a_pruned_reg_layout: gl.constexpr = gl.DistributedLinearLayout(
        reg_bases=[[0, 1], [0, 2], [8, 0], [0, 4], [0, 8]],
        lane_bases=[[0, 16], [0, 32], [1, 0], [2, 0], [4, 0]],
        warp_bases=a_warp_bases,
        block_bases=[],
        shape=[16 * num_warps, 64],
    )

    a_intermediate_layout: gl.constexpr = gl.DistributedLinearLayout(
        reg_bases=[[0, 1], [0, 2], [0, 4], [8, 0], [0, 32]],
        lane_bases=[[0, 8], [0, 16], [1, 0], [2, 0], [4, 0]],
        warp_bases=a_warp_bases,
        block_bases=[],
        shape=[16 * num_warps, 64],
    )

    idx = 0
    pid_m, pid_n = scheduler.get_tile(idx)
    off_m = pid_m * BLOCK_M
    off_n = pid_n * BLOCK_N
    
    # 1. Initial Prologue
    for ki in gl.static_range(0, BLOCK_K * (num_buffers - 2), BLOCK_K):
        producer = issue_sparse_loads_stealb(
            producer, a_pruned_desc, b_desc, off_m, off_n, ki, bars, a_pruned_bufs, b_bufs, STEALB, num_buffers,
        )
    k = BLOCK_K * (num_buffers - 2)
    producer = issue_sparse_loads_stealb(
        producer, a_pruned_desc, b_desc, off_m, off_n, k, bars, a_pruned_bufs, b_bufs, STEALB, num_buffers,
    )
    
    for _ in range(num_tiles):
        consumer = consume_loads_only(
            consumer, bars, a_pruned_bufs, a_pruned_reg_layout, a_intermediate_layout, num_buffers
        )

        if STEALB:
            tma.store_wait(pendings=0)
            
        # 2. Main inner loop
        for k in range(BLOCK_K * (num_buffers - 1), K, BLOCK_K):
            producer = issue_sparse_loads_stealb(
                producer, a_pruned_desc, b_desc, off_m, off_n, k, bars, a_pruned_bufs, b_bufs, STEALB, num_buffers,
            )
            consumer = consume_loads_only(
                consumer, bars, a_pruned_bufs, a_pruned_reg_layout, a_intermediate_layout, num_buffers
            )

        epilogue_off_m = off_m
        epilogue_off_n = off_n

        idx += 1
        pid_m, pid_n = scheduler.get_tile(idx)
        off_m = pid_m * BLOCK_M
        off_n = pid_n * BLOCK_N
        pred = idx < num_tiles
        
        # 3. Peeled next prologue (with consumer properly restored)
        for ki in gl.static_range(0, BLOCK_K * (num_buffers - 2), BLOCK_K):
            producer = issue_sparse_loads_stealb(
                producer, a_pruned_desc, b_desc, off_m, off_n, ki, bars, a_pruned_bufs, b_bufs, STEALB, num_buffers, pred,
            )
            consumer = consume_loads_only(
                consumer, bars, a_pruned_bufs, a_pruned_reg_layout, a_intermediate_layout, num_buffers
            )
            
        k = BLOCK_K * (num_buffers - 2)
        producer = issue_sparse_loads_stealb(
            producer, a_pruned_desc, b_desc, off_m, off_n, k, bars, a_pruned_bufs, b_bufs, STEALB, num_buffers, pred,
        )

        if not STEALB:
            c_buf = c_smem
            tma.store_wait(pendings=0)
        else:
            c_buf = b_bufs.index(producer % (num_buffers + STEALB))._reinterpret(
                dtype, c_desc.block_type.shape, c_desc.layout
            )
            
        fence_async_shared()
        tma.async_copy_shared_to_global(c_desc, [epilogue_off_m, epilogue_off_n], c_buf)
        
    tma.store_wait(pendings=0)

# ... [matmul_get_configs and sparse_runtime_matmul_tma_set_block_size_hook remain identical] ...

def matmul_get_configs(pre_hook=None):
    def valid(BM, BN, BK, warps, buffers, SB):
        smem_bytes = 2 * (
                (buffers * BM * BK) +
                ((buffers + SB) * BK * BN) +
                ((1 - SB) * BM * BN)
        ) + (8 * buffers)

        if smem_bytes > 232448: return False
        if SB and 2 * BN * BK < BM * BN: return False
        if SB and BM > BK: return False
        if (BM * BN) >= 65536 and warps < 12: return False
        if (BM * BN) <= 4096 and warps > 8: return False

        elements_per_thread = (BM * BN) / (warps * 32)
        if elements_per_thread > 256: return False
        return True

    return [
        triton.Config(
            {"BLOCK_M": BM, "BLOCK_N": BN, "BLOCK_K": BK, "num_buffers": buffers, "STEALB": SB},
            num_warps=warps,
            pre_hook=pre_hook,
        )
        for BM in (64, 128, 256)
        for BN in (64, 128, 256)
        for BK in (64, 128, 256)
        for warps in (4, 8, 16)
        for buffers in (3, 4, 5, 6, 7)
        for SB in (True, False)
        if valid(BM, BN, BK, warps, buffers, SB)
    ]

def sparse_runtime_matmul_tma_set_block_size_hook(nargs):
    block_m = nargs["BLOCK_M"]
    block_n = nargs["BLOCK_N"]
    block_k = nargs["BLOCK_K"]

    nargs["a_pruned_desc"].block_shape = [block_m, block_k]
    nargs["b_desc"].block_shape = [block_k, block_n]
    nargs["c_desc"].block_shape = [block_m, block_n]

    nargs["a_pruned_desc"].layout = gl.NVMMASharedLayout.get_default_for(nargs["a_pruned_desc"].block_shape, gl.float16)
    nargs["b_desc"].layout = gl.NVMMASharedLayout.get_default_for(nargs["b_desc"].block_shape, gl.float16)
    nargs["c_desc"].layout = gl.NVMMASharedLayout.get_default_for(nargs["c_desc"].block_shape, gl.float16)

# Wrap the memory-only kernel in the autotuner
sparse_runtime_kernel = triton.autotune(
    configs=matmul_get_configs(pre_hook=sparse_runtime_matmul_tma_set_block_size_hook),
    key=["M", "N", "K"],
    do_bench = lambda kernel_call, quantiles: triton.testing.do_bench_cudagraph(
        kernel_call, quantiles=quantiles),
)(run_sparse_runtime_matmul)

# The host-side wrapper expected by the benchmarking script
def run_sparse_runtime_matmul(A_pruned, B, C=None):
    M, N, K = A_pruned.shape[0], B.shape[1], B.shape[0]
    
    if C is None:
        c = torch.empty((M, N), device=A_pruned.device, dtype=torch.float16)
    else:
        c = C

    # Set up dummy blocks for descriptors (the pre_hook will resize these)
    dummy_block = [1, 1]
    dummy_layout_f16 = gl.NVMMASharedLayout.get_default_for(dummy_block, gl.float16)
    
    a_desc = TensorDescriptor.from_tensor(A_pruned, dummy_block, dummy_layout_f16)
    b_desc = TensorDescriptor.from_tensor(B, dummy_block, dummy_layout_f16)
    c_desc = TensorDescriptor.from_tensor(c, dummy_block, dummy_layout_f16)

    def grid(meta):
        num_sms = torch.cuda.get_device_properties("cuda").multi_processor_count
        num_pid = triton.cdiv(M, meta["BLOCK_M"]) * triton.cdiv(N, meta["BLOCK_N"])
        return (min(num_sms, num_pid), )

    # Launch the kernel. Note: We do not pass SparseWGMMA here because 
    # the memory-only kernel has MMA operations stripped out.
    sparse_runtime_kernel[grid](
        a_desc, 
        b_desc, 
        c_desc, 
        PersistentTileScheduler, 
        M, N, K
    )

    return c

if __name__ == "__main__":
    import time
    
    os.environ["TRITON_ALWAYS_COMPILE"]="1"
    
    for M, N, K in [(49152, 16, 49152)]:
        # Note: I'm keeping the shape 64 for BLOCK_M/BLOCK_K to match your [16*num_warps, 64] layout
        # If your local environment was using 128, just make sure `shape=[BLOCK_M, BLOCK_K]` inside layout matches it!
        for BLOCK_M, BLOCK_N, BLOCK_K in [(64, 64, 64)]:
            for num_warps, num_buffers, SB in [(4, 4, False)]:

                A = torch.randn(M, K, device="cuda", dtype=torch.float16)
                B = torch.randn((K, N), device="cuda", dtype=torch.float16)
                C = torch.empty(M, N, device="cuda", dtype=torch.float16)

                A_pruned = prune_2_4(A)

                a_pruned_layout = gl.NVMMASharedLayout.get_default_for([BLOCK_M, BLOCK_K], gl.float16)
                b_layout = gl.NVMMASharedLayout.get_default_for([BLOCK_K, BLOCK_N], gl.float16)
                c_layout = gl.NVMMASharedLayout.get_default_for([BLOCK_M, BLOCK_N], gl.float16)

                a_pruned_desc = TensorDescriptor.from_tensor(A_pruned, [BLOCK_M, BLOCK_K], a_pruned_layout)
                b_desc = TensorDescriptor.from_tensor(B, [BLOCK_K, BLOCK_N], b_layout)
                c_desc = TensorDescriptor.from_tensor(C, [BLOCK_M, BLOCK_N], c_layout)

                num_sms = torch.cuda.get_device_properties("cuda").multi_processor_count
                num_pid = triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N)
                grid = (min(num_sms, num_pid),)
                
                # Warmup
                for _ in range(5):
                    run_sparse_runtime_matmul[grid](
                        a_pruned_desc, b_desc, c_desc, PersistentTileScheduler,
                        M, N, K, BLOCK_M, BLOCK_N, BLOCK_K, num_buffers, STEALB=False, num_warps=num_warps,
                    )
                
                torch.cuda.synchronize()
                start = time.time()
                
                iters = 10
                for _ in range(iters):
                    run_sparse_runtime_matmul[grid](
                        a_pruned_desc, b_desc, c_desc, PersistentTileScheduler,
                        M, N, K, BLOCK_M, BLOCK_N, BLOCK_K, num_buffers, STEALB=False, num_warps=num_warps,
                    )
                    
                torch.cuda.synchronize()
                end = time.time()
                
                # print(f"Mem+Register kernel ran in {(end - start) / iters * 1000:.3f} ms on average.")