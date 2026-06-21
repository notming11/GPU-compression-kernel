import torch
import triton
import itertools
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

from triton.experimental.gluon.nvidia.hopper import TensorDescriptor
from triton.experimental.gluon.language.nvidia.hopper import (
    tma,
    mbarrier,
    fence_async_shared,
    warpgroup_mma_init,
    warpgroup_mma,
    warpgroup_mma_wait,
)

# configuration for autotune
def find_configs(is_pipelined=False):
    """Generate a list of valid configurations, optionally including num_buf."""
    valid_configs = []
    
    # Search space
    M_N_configs = [64, 128, 256]
    K_configs = [64, 128]
    warps_configs = [4, 8] 
    
    # If pipelined, explore double, triple, and quad buffering
    buf_configs = [2, 3, 4] if is_pipelined else [1]
    
    for BLOCK_M, BLOCK_N, BLOCK_K, num_warps, num_buf in itertools.product(M_N_configs, M_N_configs, K_configs, warps_configs, buf_configs):
        
        # 1. Prune by Shared Memory Limit (Hopper limit is ~227KB)
        # We multiply the memory needed for A and B by num_buf!
        smem_bytes = (num_buf * BLOCK_M * BLOCK_K * 2) + \
                     (num_buf * BLOCK_K * BLOCK_N * 2) + \
                     (BLOCK_M * BLOCK_N * 4) 
                    
        if smem_bytes > 227 * 1024: 
            continue # Skip, exceeds Hopper SM limits

        # 2. Check if warps can be cleanly distributed
        mReps = (BLOCK_M + 63) // 64
        nReps = num_warps // mReps
        if nReps == 0:
            continue
            
        # 3. Check for a valid INSTR_SHAPE_N
        maxN = max(BLOCK_N // nReps, 8)
        n = 256
        while n > maxN or BLOCK_N % n != 0:
            n -= 8
            if n <= 0: break
            
        if n > 0:
            valid_configs.append({
                "BLOCK_M": BLOCK_M, "BLOCK_N": BLOCK_N, "BLOCK_K": BLOCK_K, 
                "num_warps": num_warps, "instr_shape_n": n, "num_buf": num_buf
            })
            
    return valid_configs

# helper function
@gluon.constexpr_function
def get_instr_shape_n(BLOCK_M, BLOCK_N, num_warps):
    m = 16
    mReps = triton.cdiv(BLOCK_M, m)
    nReps = triton.cdiv(num_warps, mReps)
    maxN = max(BLOCK_N // nReps, 8)

    # get INSTR_SHAPE_N, it has to be a factor of BLOCK_N and a multiple of 8
    n = 256
    while n > maxN or BLOCK_N % n != 0:
        n -= 8
    return n

@gluon.constexpr_function
def get_warps_per_cta(BLOCK_M, BLOCK_N, num_warps):
    m = 16
    warps_per_cta = [4, 1]
    while warps_per_cta[0] * warps_per_cta[1] != num_warps:
        if BLOCK_M > m * warps_per_cta[0]:
            warps_per_cta[0] *= 2
        else:
            warps_per_cta[1] *= 2
    return warps_per_cta

# WGMMA pipelined with asynchronous load
@gluon.jit
def pipleine_wgmma_async_load_kernel(
    a_desc, b_desc, c_desc,
    NUM_WARPS: gl.constexpr, num_buf: gl.constexpr
):

    # tensor constant
    BLOCK_M: gl.constexpr = a_desc.block_type.shape[0]
    BLOCK_K: gl.constexpr = b_desc.block_type.shape[0]
    BLOCK_N: gl.constexpr = b_desc.block_type.shape[1]
    K = a_desc.shape[1]
    dtype: gl.constexpr = a_desc.dtype

    pid_M = gl.program_id(0)
    pid_N = gl.program_id(1)

    start_M = pid_M * BLOCK_M
    start_N = pid_N * BLOCK_N

    # initialize shared memory
    a_smem = gl.allocate_shared_memory(dtype, [num_buf, BLOCK_M, BLOCK_K], a_desc.layout)
    b_smem = gl.allocate_shared_memory(dtype, [num_buf, BLOCK_K, BLOCK_N], b_desc.layout)

    # set mma layout
    m: gl.constexpr = 16
    n: gl.constexpr = get_instr_shape_n(BLOCK_M, BLOCK_N, NUM_WARPS)
    k: gl.constexpr = 256 // dtype.primitive_bitwidth
    warps_per_cta: gl.constexpr = get_warps_per_cta(BLOCK_M, BLOCK_N, NUM_WARPS)
    mma_layout: gl.constexpr = gl.NVMMADistributedLayout(
        version =[3, 0],
        warps_per_cta = warps_per_cta,
        instr_shape = [m, n, k]
    )

    # initialize accumulator
    acc = gl.zeros((BLOCK_M, BLOCK_N), dtype = gl.float32, layout = mma_layout)
    acc = warpgroup_mma_init(acc)

    # set up barriers for tma
    bars = gl.allocate_shared_memory(gl.int64, [num_buf, 1], mbarrier.MBarrierLayout())
    for i in range(0, num_buf):
        mbarrier.init(bars.index(i), count = 1)

    fetch_index = 0
    fetch_k = 0

    # prefetch data to register with tma
    for _ in gl.static_range(num_buf - 1):
        bar = bars.index(fetch_index)
        a = a_smem.index(fetch_index)
        b = b_smem.index(fetch_index)

        mbarrier.expect(bar, a_desc.block_type.nbytes + b_desc.block_type.nbytes)
        tma.async_copy_global_to_shared(a_desc, [start_M, fetch_k], bar, a)
        tma.async_copy_global_to_shared(b_desc, [fetch_k, start_N], bar, b)
        
        fetch_index += 1
        fetch_k += BLOCK_K

    # main loop to perform mma
    compute_index = 0
    for k in range(0, K, BLOCK_K):
        # if there's blocks to fetch, fetch it
        if fetch_k < K:
            a = a_smem.index(fetch_index)
            b = b_smem.index(fetch_index)
            bar = bars.index(fetch_index)
            mbarrier.expect(bar, a_desc.block_type.nbytes + b_desc.block_type.nbytes)
            tma.async_copy_global_to_shared(a_desc, [start_M, fetch_k], bar, a)
            tma.async_copy_global_to_shared(b_desc, [fetch_k, start_N], bar, b)

            fetch_index += 1
            if fetch_index == num_buf:
                fetch_index = 0
            fetch_k += BLOCK_K

        # perform mma
        # wait for tma to finish fetching
        compute_block = k // BLOCK_K
        compute_phase = (compute_block // num_buf) & 1
        bar = bars.index(compute_index)
        mbarrier.wait(bar, phase = compute_phase)

        a = a_smem.index(compute_index)
        b = b_smem.index(compute_index)

        # wgmma wait is skipped for k = 0
        acc = warpgroup_mma_wait(num_outstanding = num_buf - 1, deps = (acc, ))
        acc = warpgroup_mma(a, b, acc, is_async = True)

        compute_index += 1
        if compute_index == num_buf:
            compute_index = 0
    
    # flush the buffer
    acc = warpgroup_mma_wait(num_outstanding = 0, deps = (acc, ))
    
    for i in gl.static_range(num_buf):
        mbarrier.invalidate(bars.index(i))

    c_smem = gl.allocate_shared_memory(c_desc.dtype, [BLOCK_M, BLOCK_N], c_desc.layout)
    c_smem.store(acc)
    fence_async_shared()
    tma.async_copy_shared_to_global(c_desc, [start_M, start_N], c_smem)
    tma.store_wait(pendings = 0)

def pipleine_wgmma_async_load(A, B, C, BLOCK_M, BLOCK_N, BLOCK_K, num_warps, num_buf):
    M, N = C.shape

    a_layout = gl.NVMMASharedLayout.get_default_for([BLOCK_M, BLOCK_K], gl.float16)
    a_desc = TensorDescriptor.from_tensor(A, [BLOCK_M, BLOCK_K], a_layout)

    b_layout = gl.NVMMASharedLayout.get_default_for([BLOCK_K, BLOCK_N], gl.float16)
    b_desc = TensorDescriptor.from_tensor(B, [BLOCK_K, BLOCK_N], b_layout)

    c_layout = gl.NVMMASharedLayout.get_default_for([BLOCK_M, BLOCK_N], gl.float32)
    c_desc = TensorDescriptor.from_tensor(C, [BLOCK_M, BLOCK_N], c_layout)

    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    pipleine_wgmma_async_load_kernel[grid](a_desc, b_desc, c_desc, num_warps, num_buf, num_warps = num_warps)

# persistent WGMMA pipelined with async load
@gluon.jit
def persistent_pipeline_wgmma_async_load_kernel(
    a_desc, b_desc, c_desc,
    NUM_WARPS: gl.constexpr, num_buf: gl.constexpr
):

    pid = gl.program_id(0)
    num_programs = gl.num_programs(0)

    BLOCK_M: gl.constexpr = a_desc.block_type.shape[0]
    BLOCK_K: gl.constexpr = b_desc.block_type.shape[0]
    BLOCK_N: gl.constexpr = b_desc.block_type.shape[1]
    M = c_desc.shape[0]
    N = c_desc.shape[1]
    K = a_desc.shape[1]
    dtype: gl.constexpr = a_desc.dtype

    # initialize shared memory
    a_smem = gl.allocate_shared_memory(dtype, [num_buf, BLOCK_M, BLOCK_K], a_desc.layout)
    b_smem = gl.allocate_shared_memory(dtype, [num_buf, BLOCK_K, BLOCK_N], b_desc.layout)
    bars = gl.allocate_shared_memory(gl.int64, [num_buf, 1], mbarrier.MBarrierLayout())

    # persistent setup
    GROUP_SIZE_M: gl.constexpr = 8
    num_pid_m = gl.cdiv(M, BLOCK_M)
    num_pid_n = gl.cdiv(N, BLOCK_N)
    num_jobs = num_pid_m * num_pid_n
    num_pid_group = GROUP_SIZE_M * num_pid_n

    for job_id in range(pid, num_jobs, num_programs):
        # swizzling
        group_id = job_id // num_pid_group
        first_pid_m = group_id * GROUP_SIZE_M
        curr_group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)

        pid_M = first_pid_m + ((job_id % num_pid_group) % curr_group_size_m)
        pid_N = (job_id % num_pid_group) // curr_group_size_m

        start_M = pid_M * BLOCK_M
        start_N = pid_N * BLOCK_N

        # set mma layout
        m: gl.constexpr = 16
        n: gl.constexpr = get_instr_shape_n(BLOCK_M, BLOCK_N, NUM_WARPS)
        k: gl.constexpr = 256 // dtype.primitive_bitwidth
        warps_per_cta: gl.constexpr = get_warps_per_cta(BLOCK_M, BLOCK_N, NUM_WARPS)
        mma_layout: gl.constexpr = gl.NVMMADistributedLayout(
            version =[3, 0],
            warps_per_cta = warps_per_cta,
            instr_shape = [m, n, k]
        )

        # initialize accumulator
        acc = gl.zeros((BLOCK_M, BLOCK_N), dtype = gl.float32, layout = mma_layout)
        acc = warpgroup_mma_init(acc)

        # set up barriers for tma
        for i in range(0, num_buf):
            mbarrier.init(bars.index(i), count = 1)

        fetch_index = 0
        fetch_k = 0

        # prefetch data to register with tma
        for _ in gl.static_range(num_buf - 1):
            bar = bars.index(fetch_index)
            a = a_smem.index(fetch_index)
            b = b_smem.index(fetch_index)

            mbarrier.expect(bar, a_desc.block_type.nbytes + b_desc.block_type.nbytes)
            tma.async_copy_global_to_shared(a_desc, [start_M, fetch_k], bar, a)
            tma.async_copy_global_to_shared(b_desc, [fetch_k, start_N], bar, b)
        
            fetch_index += 1
            fetch_k += BLOCK_K

        # main loop to perform mma
        compute_index = 0
        for k in range(0, K, BLOCK_K):
            # if there's blocks to fetch, fetch it
            if fetch_k < K:
                a = a_smem.index(fetch_index)
                b = b_smem.index(fetch_index)
                bar = bars.index(fetch_index)
                mbarrier.expect(bar, a_desc.block_type.nbytes + b_desc.block_type.nbytes)
                tma.async_copy_global_to_shared(a_desc, [start_M, fetch_k], bar, a)
                tma.async_copy_global_to_shared(b_desc, [fetch_k, start_N], bar, b)

                fetch_index += 1
                if fetch_index == num_buf:
                    fetch_index = 0
                fetch_k += BLOCK_K

            # perform mma
            # wait for tma to finish fetching
            compute_block = k // BLOCK_K
            compute_phase = (compute_block // num_buf) & 1
            bar = bars.index(compute_index)
            mbarrier.wait(bar, phase = compute_phase)

            a = a_smem.index(compute_index)
            b = b_smem.index(compute_index)

            # wgmma wait is skipped for k = 0
            acc = warpgroup_mma_wait(num_outstanding = num_buf - 1, deps = (acc, ))
            acc = warpgroup_mma(a, b, acc, is_async = True)

            compute_index += 1
            if compute_index == num_buf:
                compute_index = 0

        # flush the buffer
        acc = warpgroup_mma_wait(num_outstanding = 0, deps = (acc, ))

        for i in gl.static_range(num_buf):
            mbarrier.invalidate(bars.index(i))

        c_smem = gl.allocate_shared_memory(c_desc.dtype, [BLOCK_M, BLOCK_N], c_desc.layout)
        c_smem.store(acc)
        fence_async_shared()
        tma.async_copy_shared_to_global(c_desc, [start_M, start_N], c_smem)
        tma.store_wait(pendings = 0)

def persistent_pipeline_wgmma_async_load(A, B, C, BLOCK_M, BLOCK_N, BLOCK_K, num_warps, num_buf):
    M, N = C.shape

    a_layout = gl.NVMMASharedLayout.get_default_for([BLOCK_M, BLOCK_K], gl.float16)
    a_desc = TensorDescriptor.from_tensor(A, [BLOCK_M, BLOCK_K], a_layout)

    b_layout = gl.NVMMASharedLayout.get_default_for([BLOCK_K, BLOCK_N], gl.float16)
    b_desc = TensorDescriptor.from_tensor(B, [BLOCK_K, BLOCK_N], b_layout)

    c_layout = gl.NVMMASharedLayout.get_default_for([BLOCK_M, BLOCK_N], gl.float32)
    c_desc = TensorDescriptor.from_tensor(C, [BLOCK_M, BLOCK_N], c_layout)

    num_sms = torch.cuda.get_device_properties("cuda").multi_processor_count

    grid = (min(num_sms, triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N)), )
    persistent_pipeline_wgmma_async_load_kernel[grid](a_desc, b_desc, c_desc, num_warps, num_buf, num_warps = num_warps)

# benchmarking
def run_benchmark():
    print("Benchmarking pipelined persistent")
    print("=================================")
    print("    K    num_buf         persistent     nonpersistent")

    def tflops(M, N, K, ms):
        return (2 * M * N * K) / (ms * 1e9)

    for num_buffers, K in itertools.product([1, 2, 3, 4], [2**i for i in range(9, 15)]):
        M, N = 8192, 8192
    
        torch.manual_seed(0)
        A = torch.randn((M, K), device="cuda", dtype=torch.float16)
        B = torch.randn((K, N), device="cuda", dtype=torch.float16)
        C = torch.empty((M, N), device="cuda", dtype=torch.float32)

        BLOCK_M = 128
        BLOCK_N = 256
        BLOCK_K = 64
        num_warps = 8

        tflop_pipe = tflops(M, N, K, triton.testing.do_bench(
            lambda: pipleine_wgmma_async_load(A, B, C, BLOCK_M, BLOCK_N, BLOCK_K, num_warps, num_buffers),
            rep=20
        ))

        tflop_persistent = tflops(M, N, K, triton.testing.do_bench(
            lambda: persistent_pipeline_wgmma_async_load(A, B, C, BLOCK_M, BLOCK_N, BLOCK_K, num_warps, num_buffers),
            rep=20
        ))

        print(f"{K:>5} {num_buffers:>10} {tflop_persistent:>17.2f} {tflop_pipe:>17.2f}")



if __name__ == "__main__":
    run_benchmark()