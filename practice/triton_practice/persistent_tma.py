import torch
import triton
import triton.language as tl

DEVICE = "cuda"

def get_autotune_config():
    return [
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
    ]

@triton.jit
def _compute_pid(pid, num_block_m, num_block_n, GROUP_SIZE_M: tl.constexpr):
    # Find the number of blocks in a group  
    num_block_group = num_block_n * GROUP_SIZE_M

    # Find the group number of this block
    group_num = pid // num_block_group

    # Find position of the block in the group
    pid_group = pid % num_block_group

    # Find the block index in the grid
    first_pid_m = group_num * GROUP_SIZE_M
    curr_group_size_m = tl.minimum(num_block_m - first_pid_m, GROUP_SIZE_M)
    pid_m = pid_group % curr_group_size_m + first_pid_m
    pid_n = pid_group // curr_group_size_m

    return pid_m, pid_n

@triton.autotune(
    configs=get_autotune_config(),
    key=['M', 'N', 'K'],
)
@triton.jit
def persistent_matmul_tma_kernel(
    a_ptr, b_ptr, c_ptr, 
    M, N, K, 
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    NUM_SMS: tl.constexpr
):

    pid = tl.program_id(axis = 0)

    # loop over tiles to compute
    num_block_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_block_n = tl.cdiv(N, BLOCK_SIZE_N)
    tile_num = num_block_m * num_block_n
    for tile_id in tl.range(pid, tile_num, NUM_SMS):

        # calculate coordinate of current tile
        pid_m, pid_n = _compute_pid(tile_id, num_block_m, num_block_n, GROUP_SIZE_M)

        # standard matmul with tma

        # Find block pointers to blocks in A and B
        a_block_ptr = tl.make_block_ptr(
            base = a_ptr,
            shape = (M, K),
            strides = (stride_am, stride_ak),
            offsets = (pid_m * BLOCK_SIZE_M, 0),
            block_shape = (BLOCK_SIZE_M, BLOCK_SIZE_K),
            order = (1, 0)
        )

        b_block_ptr = tl.make_block_ptr(
            base = b_ptr,
            shape = (K, N),
            strides = (stride_bk, stride_bn),
            offsets = (0, BLOCK_SIZE_N * pid_n),
            block_shape = (BLOCK_SIZE_K, BLOCK_SIZE_N),
            order = (1, 0)
        )

        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
            # load data from A and B
            a = tl.load(a_block_ptr, boundary_check=(0, 1))
            b = tl.load(b_block_ptr, boundary_check=(0, 1))

            # do matmul
            accumulator = tl.dot(a, b, accumulator)

            # advance A and B to next block in k dimension
            a_block_ptr = tl.advance(a_block_ptr, (0, BLOCK_SIZE_K))
            b_block_ptr = tl.advance(b_block_ptr, (BLOCK_SIZE_K, 0))

        # Find pointers to element in C
        c_block_ptr = tl.make_block_ptr(
            base = c_ptr,
            shape = (M, N),
            strides = (stride_cm, stride_cn),
            offsets = (pid_m * BLOCK_SIZE_M, pid_n * BLOCK_SIZE_N),
            block_shape = (BLOCK_SIZE_M, BLOCK_SIZE_N),
            order = (1, 0)
        )

        # store result to C
        tl.store(c_block_ptr, accumulator, boundary_check = (0, 1))

def persistent_matmul_tma(A: torch.Tensor, B: torch.Tensor):
    # Dimensions
    M, K = A.shape
    _, N = B.shape

    # Allocate memory to C
    C = torch.empty((M, N), device = A.device, dtype = torch.float32)

    # find # of SM 
    num_sms = torch.cuda.get_device_properties(DEVICE).multi_processor_count

    # print(num_sms)

    # define the grid with 1 block per 1 SM
    def grid(meta):
        num_blocks = triton.cdiv(M, meta['BLOCK_SIZE_M']) * triton.cdiv(N, meta['BLOCK_SIZE_N'])
        return (min(num_sms, num_blocks), )

    # call kernel
    persistent_matmul_tma_kernel[grid](
        A, B, C,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        NUM_SMS=num_sms  
    )

    return C


@triton.autotune(
    configs=get_autotune_config(),
    key=['M', 'N', 'K'],
)
@triton.jit
def persistent_matmul_kernel(
    a_ptr, b_ptr, c_ptr, 
    M, N, K, 
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    NUM_SMS: tl.constexpr
):

    pid = tl.program_id(axis = 0)

    # loop over tiles to compute
    num_block_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_block_n = tl.cdiv(N, BLOCK_SIZE_N)
    tile_num = num_block_m * num_block_n
    for tile_id in tl.range(pid, tile_num, NUM_SMS):

        # calculate coordinate of current tile
        pid_m, pid_n = _compute_pid(tile_id, num_block_m, num_block_n, GROUP_SIZE_M)

        # standard matmul

        # Find pointers to element in A and B
        offset_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M                    # create offset to pointers from 
        offset_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N                    # a_ptr and b_ptr
        offset_k = tl.arange(0, BLOCK_SIZE_K)
    
        a_ptrs = a_ptr + (offset_am[:, None] * stride_am + offset_k[None, :] * stride_ak)       # create matrix of pointers 
        b_ptrs = b_ptr + (offset_k[:, None] * stride_bk + offset_bn[None, :] * stride_bn)

        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
            # load data from A and B
            a = tl.load(a_ptrs, mask = offset_k[None, :] < K-k*BLOCK_SIZE_K, other = 0.0)
            b = tl.load(b_ptrs, mask = offset_k[:, None] < K-k*BLOCK_SIZE_K, other = 0.0)

            # do matmul
            accumulator = tl.dot(a, b, accumulator)

            # advance A and B to next block in k dimension
            a_ptrs += BLOCK_SIZE_K * stride_ak
            b_ptrs += BLOCK_SIZE_K * stride_bk

        # Find pointers to element in C
        c_ptrs = c_ptr + (offset_am[:, None] * stride_cm + offset_bn[None, :] * stride_cn)
        c_mask = (offset_am[:, None] < M) & (offset_bn[None, :] < N)

        # store result to C
        tl.store(c_ptrs, accumulator, mask = c_mask)

def persistent_matmul(A: torch.Tensor, B: torch.Tensor):
    # Dimensions
    M, K = A.shape
    _, N = B.shape

    # Allocate memory to C
    C = torch.empty((M, N), device = A.device, dtype = torch.float32)

    # find # of SM 
    num_sms = torch.cuda.get_device_properties(DEVICE).multi_processor_count

    # print(num_sms)

    # define the grid with 1 block per 1 SM
    def grid(meta):
        num_blocks = triton.cdiv(M, meta['BLOCK_SIZE_M']) * triton.cdiv(N, meta['BLOCK_SIZE_N'])
        return (min(num_sms, num_blocks), )

    # call kernel
    persistent_matmul_kernel[grid](
        A, B, C,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        NUM_SMS=num_sms  
    )

    return C

@triton.autotune(
    configs=get_autotune_config(),
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr, 
    M, N, K, 
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr
):

    pid = tl.program_id(axis = 0)

    # Find the number of blocks on the C for both dimension
    num_block_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_block_n = tl.cdiv(N, BLOCK_SIZE_N)

    # Find the number of blocks in a group
    num_block_group = num_block_n * GROUP_SIZE_M

    # Find the group number of this block
    group_num = pid // num_block_group

    # Find position of the block in the group
    pid_group = pid % num_block_group

    # Find the block index in the grid
    first_pid_m = group_num * GROUP_SIZE_M
    curr_group_size_m = min(num_block_m - first_pid_m, GROUP_SIZE_M)
    pid_m = pid_group % curr_group_size_m + first_pid_m
    pid_n = pid_group // curr_group_size_m

    # Find pointers to element in A and B
    offset_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M                    # create offset to pointers from 
    offset_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N                    # a_ptr and b_ptr
    offset_k = tl.arange(0, BLOCK_SIZE_K)
    
    a_ptrs = a_ptr + (offset_am[:, None] * stride_am + offset_k[None, :] * stride_ak)       # create matrix of pointers 
    b_ptrs = b_ptr + (offset_k[:, None] * stride_bk + offset_bn[None, :] * stride_bn)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # load data from A and B
        a = tl.load(a_ptrs, mask = offset_k[None, :] < K-k*BLOCK_SIZE_K, other = 0.0)
        b = tl.load(b_ptrs, mask = offset_k[:, None] < K-k*BLOCK_SIZE_K, other = 0.0)

        # do matmul
        accumulator = tl.dot(a, b, accumulator)

        # advance A and B to next block in k dimension
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    # Find pointers to element in C
    c_ptrs = c_ptr + (offset_am[:, None] * stride_cm + offset_bn[None, :] * stride_cn)
    c_mask = (offset_am[:, None] < M) & (offset_bn[None, :] < N)

    # store result to C
    tl.store(c_ptrs, accumulator, mask = c_mask)


def matmul(A: torch.Tensor, B: torch.Tensor):
    # Dimensions
    M, K = A.shape
    _, N = B.shape

    # Allocate memory to C
    C = torch.empty((M, N), device = A.device, dtype = torch.float32)

    # Define grid
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_SIZE_M']) * triton.cdiv(N, meta['BLOCK_SIZE_N']), )

    # Launch Kernel
    matmul_kernel[grid](
        A, B, C,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
    )

    return C

# TEST & BENCHMARK SETUP
torch.manual_seed(0)

# 1. Scale up for the H100
MATRIX_SIZE = 8192
a = torch.rand((MATRIX_SIZE, MATRIX_SIZE), device=DEVICE, dtype=torch.float16)
b = torch.rand((MATRIX_SIZE, MATRIX_SIZE), device=DEVICE, dtype=torch.float16)

# 2. Print Hardware Info Once
num_sms = torch.cuda.get_device_properties(DEVICE).multi_processor_count
gpu_name = torch.cuda.get_device_name(DEVICE)
print(f"--- BENCHMARK REPORT ---")
print(f"Hardware : {gpu_name}")
print(f"Total SMs: {num_sms}")
print(f"Matrix   : {MATRIX_SIZE} x {MATRIX_SIZE}")
print(f"------------------------")

# 3. Accurate Benchmarking
grouping_ms = triton.testing.do_bench(lambda : matmul(a, b))
persistent_ms = triton.testing.do_bench(lambda: persistent_matmul(a, b))
persistent_tma_ms = triton.testing.do_bench(lambda: persistent_matmul_tma(a, b))

print(f"Grouping Matmul       : {grouping_ms:.4f} ms")
print(f"Persistent Matmul     : {persistent_ms:.4f} ms")
print(f"Persistent TMA Matmul : {persistent_tma_ms:.4f} ms")