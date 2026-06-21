import torch
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

# 1. Define the Kernel
@gluon.jit
def copy_scalar_kernel(in_ptr, out_ptr):
    # Load the scalar value from device memory
    value = gl.load(in_ptr)
    # Store the scalar value back to device memory
    gl.store(out_ptr, value)

# 2. Define the Launcher
def copy_scalar(input_tensor, output_tensor):
    # Launch the kernel with a 1D grid of size 1
    copy_scalar_kernel[(1,)](input_tensor, output_tensor)

# 3. Test the "Hello World"
if __name__ == "__main__":
    # Create an input tensor with the meaning of life, the universe, and everything
    input_data = torch.tensor([42.0], device="cuda")
    output_data = torch.empty_like(input_data)

    # Run the kernel
    copy_scalar(input_data, output_data)

    print(f"Input: {input_data.item()}")
    print(f"Output: {output_data.item()}")
    
    # Verify correctness
    torch.testing.assert_close(input_data, output_data, atol=0, rtol=0)
    print("Success! Hello, Gluon!")