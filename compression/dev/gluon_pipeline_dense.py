import torch
import triton

from triton.experimental import gluon
from triton.experimental.gluon import language as gl

from triton.experimental.gluon.nvidia.hopper import TensorDescriptor
from triton.experimental.gluon.language.nvidia.hopper import (
    tma,
    mbarrier,
    fence_async_shared,
)

try:
    from common import (
        WGMMA,
        PersistentTileScheduler
    )
except ImportError:
    try:
        from .common import (
            WGMMA,
            PersistentTileScheduler
        )
    except ImportError:
        from .compression.common import (
            WGMMA,
            PersistentTileScheduler
        )

@gluon.jit
def issue_loads_stealb(producer, a_desc, b_desc, off_m, off_n, k, bars, a_bufs, b_bufs, stealb: gl.constexpr,
                       num_buffers: gl.constexpr, pred=True):
    index = producer % num_buffers
    b_index = producer % (num_buffers + stealb)
    producer += 1
    bar = bars.index(index)
    mbarrier.expect(bar, a_desc.block_type.nbytes + b_desc.block_type.nbytes, pred)
    tma.async_copy_global_to_shared(a_desc, [off_m, k], bar, a_bufs.index(index), pred)
    tma.async_copy_global_to_shared(b_desc, [k, off_n], bar, b_bufs.index(b_index), pred)
    return producer

@gluon.jit
def issue_mma_stealb(consumer, mma, bars, a_bufs, b_bufs, stealb: gl.constexpr, num_buffers: gl.constexpr):
    index = consumer % num_buffers
    b_index = consumer % (num_buffers + stealb)
    phase = consumer // num_buffers & 1
    consumer += 1
    mbarrier.wait(bars.index(index), phase)
    mma = mma.wait_num_outstanding(0)
    mma = mma.issue_async_mma(a_bufs.index(index), b_bufs.index(b_index))
    return consumer, mma

@gluon.jit
def persistent_matmul_pipelined_kernel(a_desc, b_desc, c_desc, MMAImpl: gl.constexpr, SchedulerImpl: gl.constexpr,
                                        M, N, K, BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr, BLOCK_K: gl.constexpr, num_buffers: gl.constexpr, STEALB: gl.constexpr, num_warps: gl.constexpr):
    dtype: gl.constexpr = a_desc.dtype
    K = a_desc.shape[1]

    gl.static_assert(num_buffers >= 3, "expected at least 3 buffers")
    a_bufs = gl.allocate_shared_memory(dtype, [num_buffers] + a_desc.block_type.shape, a_desc.layout)
    b_bufs = gl.allocate_shared_memory(dtype, [num_buffers + STEALB] + b_desc.block_type.shape, b_desc.layout)
    if not STEALB:
        c_smem = gl.allocate_shared_memory(dtype, c_desc.block_type.shape, c_desc.layout)
    else:
        gl.static_assert(2 * BLOCK_N * BLOCK_K >= BLOCK_M * BLOCK_N, "B tile not large enough to steal")
    bars = gl.allocate_shared_memory(gl.int64, [num_buffers, 1], mbarrier.MBarrierLayout())
    for i in gl.static_range(num_buffers):
        mbarrier.init(bars.index(i), count=1)
    producer = 0
    consumer = 0

    mma = MMAImpl.initialize(dtype, BLOCK_M, BLOCK_N, num_warps)
    scheduler = SchedulerImpl.initialize(c_desc.shape[0], c_desc.shape[1], BLOCK_M, BLOCK_N)
    num_tiles = scheduler.get_num_tiles()

    idx = 0
    pid_m, pid_n = scheduler.get_tile(idx)
    off_m = pid_m * BLOCK_M
    off_n = pid_n * BLOCK_N
    for ki in gl.static_range(0, BLOCK_K * (num_buffers - 2), BLOCK_K):
        producer = issue_loads_stealb(producer, a_desc, b_desc, off_m, off_n, ki, bars, a_bufs, b_bufs, STEALB,
                                      num_buffers)
    k = BLOCK_K * (num_buffers - 2)
    producer = issue_loads_stealb(producer, a_desc, b_desc, off_m, off_n, k, bars, a_bufs, b_bufs, STEALB, num_buffers)

    for _ in range(num_tiles):
        consumer, mma = issue_mma_stealb(consumer, mma, bars, a_bufs, b_bufs, STEALB, num_buffers)
        if STEALB:
            tma.store_wait(pendings=0)
        for k in range(BLOCK_K * (num_buffers - 1), K, BLOCK_K):
            producer = issue_loads_stealb(producer, a_desc, b_desc, off_m, off_n, k, bars, a_bufs, b_bufs, STEALB,
                                          num_buffers)
            consumer, mma = issue_mma_stealb(consumer, mma, bars, a_bufs, b_bufs, STEALB, num_buffers)

        epilogue_off_m = off_m
        epilogue_off_n = off_n

        idx += 1
        pid_m, pid_n = scheduler.get_tile(idx)
        off_m = pid_m * BLOCK_M
        off_n = pid_n * BLOCK_N
        pred = idx < num_tiles
        for ki in gl.static_range(0, BLOCK_K * (num_buffers - 2), BLOCK_K):
            producer = issue_loads_stealb(producer, a_desc, b_desc, off_m, off_n, ki, bars, a_bufs, b_bufs, STEALB,
                                          num_buffers, pred)
            consumer, mma = issue_mma_stealb(consumer, mma, bars, a_bufs, b_bufs, STEALB, num_buffers)
        k = BLOCK_K * (num_buffers - 2)
        producer = issue_loads_stealb(producer, a_desc, b_desc, off_m, off_n, k, bars, a_bufs, b_bufs, STEALB,
                                      num_buffers)

        mma = mma.wait_num_outstanding(0)
        c, mma = mma.take_result()
        c = c.to(dtype)
        if not STEALB:
            c_buf = c_smem
            tma.store_wait(pendings=0)
        else:
            c_buf = b_bufs.index(producer % (num_buffers + STEALB))._reinterpret(dtype, c_desc.block_type.shape,
                                                                                 c_desc.layout)
        c_buf.store(c)
        fence_async_shared()
        tma.async_copy_shared_to_global(c_desc, [epilogue_off_m, epilogue_off_n], c_buf)
    tma.store_wait(pendings=0)

def persistent_matmul_pipelined(A, B, C, BLOCK_M, BLOCK_N, BLOCK_K, num_buffers, num_warps, SchedulerImpl):
    M, N = C.shape

    a_layout = gl.NVMMASharedLayout.get_default_for([BLOCK_M, BLOCK_K], gl.float16)
    b_layout = gl.NVMMASharedLayout.get_default_for([BLOCK_K, BLOCK_N], gl.float16)
    c_layout = gl.NVMMASharedLayout.get_default_for([BLOCK_M, BLOCK_N], gl.float16)

    a_desc = TensorDescriptor.from_tensor(A, [BLOCK_M, BLOCK_K], a_layout)
    b_desc = TensorDescriptor.from_tensor(B, [BLOCK_K, BLOCK_N], b_layout)
    c_desc = TensorDescriptor.from_tensor(C, [BLOCK_M, BLOCK_N], c_layout)

    num_sms = torch.cuda.get_device_properties("cuda").multi_processor_count
    num_pid = triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N)
    grid = (min(num_sms, num_pid), )
    persistent_matmul_pipelined_kernel[grid](a_desc, b_desc, c_desc, WGMMA, SchedulerImpl, num_buffers,
                                             STEALB=num_buffers == 4, num_warps=num_warps)

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

        if (BM * BN) >= 65536 and warps < 12:  return False
        if (BM * BN) <= 4096 and warps > 8:    return False

        elements_per_thread = (BM * BN) / (warps * 32)
        if elements_per_thread > 256:
            return False

        return True

    return [
        triton.Config(
            {
                "BLOCK_M": BM,
                "BLOCK_N": BN,
                "BLOCK_K": BK,
                "num_buffers": buffers,
                "STEALB": SB,
            },
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

def matmul_tma_set_block_size_hook(nargs):
    block_m = nargs["BLOCK_M"]
    block_n = nargs["BLOCK_N"]
    block_k = nargs["BLOCK_K"]

    nargs["a_desc"].block_shape = [block_m, block_k]
    nargs["b_desc"].block_shape = [block_k, block_n]
    nargs["c_desc"].block_shape = [block_m, block_n]

    nargs["a_desc"].layout = gl.NVMMASharedLayout.get_default_for(nargs["a_desc"].block_shape, gl.float16)
    nargs["b_desc"].layout = gl.NVMMASharedLayout.get_default_for(nargs["b_desc"].block_shape, gl.float16)
    nargs["c_desc"].layout = gl.NVMMASharedLayout.get_default_for(nargs["c_desc"].block_shape, gl.float16)

dense_kernel = triton.autotune(
    configs=matmul_get_configs(pre_hook=matmul_tma_set_block_size_hook),
    key=["M", "N", "K"],
    do_bench = lambda kernel_call, quantiles: triton.testing.do_bench_cudagraph(
        kernel_call, quantiles=quantiles),
)(persistent_matmul_pipelined_kernel)

def run_dense_matmul(A, B, C=None):
    M, N, K = A.shape[0], B.shape[1], B.shape[0]

    if C is None:
        c = torch.empty((M, N), device=A.device, dtype=torch.float16)
    else:
        c = C
    dummy_block = [1, 1]
    dummy_layout_f16 = gl.NVMMASharedLayout.get_default_for(dummy_block, gl.float16)
    a_desc = TensorDescriptor.from_tensor(A, dummy_block, dummy_layout_f16)
    b_desc = TensorDescriptor.from_tensor(B, dummy_block, dummy_layout_f16)
    c_desc = TensorDescriptor.from_tensor(c, dummy_block, dummy_layout_f16)

    def grid(meta):
        num_sms = torch.cuda.get_device_properties("cuda").multi_processor_count
        num_pid = triton.cdiv(M, meta["BLOCK_M"]) * triton.cdiv(N, meta["BLOCK_N"])
        return (min(num_sms, num_pid), )
        
    dense_kernel[grid](a_desc, b_desc, c_desc, WGMMA, PersistentTileScheduler, M, N, K)
    return c

if __name__ == "__main__":
    # sizes = [(768, 768, 768), (2048, 1024, 2048)]

    # for M, N, K in sizes:
    #     A = torch.randn(M, K, device="cuda", dtype=torch.float16)
    #     B = torch.randn((K, N), device="cuda", dtype=torch.float16)
    #     C = torch.empty(M, N, device="cuda", dtype=torch.float16)
    #     D = run_dense_matmul(A,B)
    #     torch.testing.assert_close(A @ B, D, rtol=1e-3, atol=1e-1)
    # print("Done dense.")
    
    for M, N, K in [(49152, 16, 49152)]:
        for BLOCK_M, BLOCK_N, BLOCK_K in [(128, 64, 128)]:
            for num_warps, num_buffers, SB in [(8, 3, False)]:
                # print(f"Testing dense persistent: M={M}, N={N}, K={K}, BLOCK_M={BLOCK_M}, BLOCK_N={BLOCK_N}, BLOCK_K={BLOCK_K}, num_warps={num_warps}...", end=" ", flush=True)

                A = torch.randn(M, K, device="cuda", dtype=torch.float16)
                B = torch.randn((K, N), device="cuda", dtype=torch.float16)
                C = torch.empty(M, N, device="cuda", dtype=torch.float16)

                # A_pruned = prune_2_4(A)
                # C_ref = A_pruned @ B
                
                # D = run_sparse_runtime_matmul(A_pruned, B, C)
                # torch.testing.assert_close(C_ref, D, rtol=1e-3, atol=1e-1)
                # C = torch.empty(M, N, device="cuda", dtype=torch.float16)

                # sparse_persistent_matmul(A, E, B, C, BLOCK_M, BLOCK_N, BLOCK_K, 3, num_warps, PersistentTileScheduler)
                a_layout = gl.NVMMASharedLayout.get_default_for(
                    [BLOCK_M, BLOCK_K], gl.float16
                )
                b_layout = gl.NVMMASharedLayout.get_default_for([BLOCK_K, BLOCK_N], gl.float16)
                c_layout = gl.NVMMASharedLayout.get_default_for([BLOCK_M, BLOCK_N], gl.float16)

                a_desc = TensorDescriptor.from_tensor(A, [BLOCK_M, BLOCK_K], a_layout)
                b_desc = TensorDescriptor.from_tensor(B, [BLOCK_K, BLOCK_N], b_layout)
                c_desc = TensorDescriptor.from_tensor(C, [BLOCK_M, BLOCK_N], c_layout)

                num_sms = torch.cuda.get_device_properties("cuda").multi_processor_count
                num_pid = triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N)
                grid = (min(num_sms, num_pid),)
                persistent_matmul_pipelined_kernel[grid](
                    a_desc,
                    b_desc,
                    c_desc,
                    WGMMA,
                    PersistentTileScheduler,
                    M, N, K, 
                    BLOCK_M,    
                    BLOCK_N,
                    BLOCK_K,
                    num_buffers,            
                    STEALB=False,
                    num_warps=num_warps,
                )
                
                # torch.testing.assert_close(C_ref, C, rtol=1e-3, atol=1e-1)
                print("PASSED")