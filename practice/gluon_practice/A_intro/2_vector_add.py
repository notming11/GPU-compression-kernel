import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

@gluon.jit
def vector_add_kernel(x_ptr, y_ptr, out_ptr, num_elements, BLOCK_SIZE: gl.constexpr):
    pid = gl.program_id(0)

    start = pid*BLOCK_SIZE
    end = gl.minimum(start + BLOCK_SIZE, num_elements)

    # Very slow, just to start with
    for i in range(start, end):
        x = gl.load(x_ptr + i)
        y = gl.load(y_ptr + i)

        gl.store(out_ptr + i, x + y)

def vector_add(x: torch.Tensor, y: torch.Tensor, BLOCK_SIZE: int = 1024):
    output = torch.empty_like(x)
    num_elements = x.numel()

    grid = (triton.cdiv(num_elements, BLOCK_SIZE), )
    vector_add_kernel[grid](x, y, output, num_elements, BLOCK_SIZE, num_warps = 4)

    return output

if __name__ == "__main__":
    torch.manual_seed(0)
    size = 98432
    
    x = torch.rand(size, device='cuda')
    y = torch.rand(size, device='cuda')
    
    # Run our Gluon kernel
    output_gluon = vector_add(x, y)
    
    # Verify against PyTorch's native addition
    torch.testing.assert_close(x + y, output_gluon)
    print("Success! Vector Addition completed in Gluon.")