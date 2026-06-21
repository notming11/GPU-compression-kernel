import torch
import triton
import triton.language as tl

DEVICE = triton.runtime.driver.active.get_active_torch_device()

@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis = 0)       # find idx of block

    block_start = pid * BLOCK_SIZE
    offset = block_start + tl.arange(0, BLOCK_SIZE)         # array of idx of the vector

    mask = offset < n_elements          # filter element that's out of bound

    x = tl.load(x_ptr + offset, mask = mask)
    y = tl.load(y_ptr + offset, mask = mask)                # load data from global memory to local memory

    output = x + y                      # perform vector addition on block
    tl.store(output_ptr + offset, output, mask = mask)      # store data from local memory to global memory

def vec_add(x : torch.tensor, y : torch.tensor, n_elements : int):
    output = torch.empty_like(x)        # create empty array with length same as x

    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )   # find dimension of the grid

    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)         # start kernel

    return output

# setup
torch.manual_seed(0)
size = 2**26
x = torch.rand(size, device=DEVICE)
y = torch.rand(size, device=DEVICE)

# correctness check
output_torch = x + y
output_triton = vec_add(x, y, size)

print(f'The maximum difference between torch and triton is {torch.max(torch.abs(output_torch - output_triton))}')

# benchmarking
ms_baseline = triton.testing.do_bench(lambda: x + y)              # Pytorch baseline benchmark
ms_triton = triton.testing.do_bench(lambda: vec_add(x, y, size))  # triton kernel

print(f"Baseline Time (PyTorch)    : {ms_baseline:.4f} ms")
print(f"Optimization Time (Triton) : {ms_triton:.4f} ms")
