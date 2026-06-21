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

@triton.autotune(
    configs=get_autotune_config(),
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_kernel_naive(
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
    grid_n = tl.cdiv(N, BLOCK_SIZE_N)
    pid_m = pid // grid_n
    pid_n = pid % grid_n

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
        C.stride(0), C.stride(1)
    )

    return C

def matmul_naive(A: torch.Tensor, B: torch.Tensor):
    # Dimensions
    M, K = A.shape
    _, N = B.shape

    # Allocate memory to C
    C = torch.empty((M, N), device = A.device, dtype = torch.float32)

    # Define grid
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_SIZE_M']) * triton.cdiv(N, meta['BLOCK_SIZE_N']), )

    # Launch Kernel
    matmul_kernel_naive[grid](
        A, B, C,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1)
    )

    return C

# TEST
torch.manual_seed(0)

# Create two matrices
a = torch.rand((4096, 4096), device=DEVICE, dtype=torch.float32)
b = torch.rand((4096, 4096), device=DEVICE, dtype=torch.float32)

# Run standard PyTorch
torch_output = torch.matmul(a, b)

# Run your Triton kernel
triton_output = matmul_naive(a, b)

# Check correctness
max_diff = torch.max(torch.abs(torch_output - triton_output)).item()
if max_diff > 1e-3:
    print(f"Mismatch: {max_diff}")
else :
    print("Match")

# benchmarking
ms_baseline = triton.testing.do_bench(lambda: torch.matmul(a, b))              # Pytorch baseline benchmark
ms_triton = triton.testing.do_bench(lambda: matmul(a, b))  # triton kernel
ms_naive = triton.testing.do_bench(lambda: matmul_naive(a, b))

print(f"Baseline Time (PyTorch)    : {ms_baseline:.4f} ms")
print(f"Naive Time (Triton) : {ms_naive:.4f} ms")
print(f"Optimization Time (Triton) : {ms_triton:.4f} ms")