# try warp specializtion wit element add in 2d

import torch
import triton
import importlib
from functools import partial
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

from triton.experimental.gluon.nvidia.hopper import TensorDescriptor
from triton.experimental.gluon.language.nvidia.hopper import tma, mbarrier, fence_async_shared

# 2d add with tma
@gluon.jit
def write_smem_tma(write_idx, bars, x_smem, y_smem, x_desc, y_desc, offset_M, BLOCK_N: gl.constexpr, num_buf: gl.constexpr):
    offset_N = write_idx * BLOCK_N
    bar = bars.index(write_idx % num_buf)
    mbarrier.expect(bar, x_desc.block_type.nbytes + y_desc.block_type.nbytes)
    tma.async_copy_global_to_shared(x_desc, [offset_M, offset_N], bar, x_smem.index(write_idx % num_buf))
    tma.async_copy_global_to_shared(y_desc, [offset_M, offset_N], bar, y_smem.index(write_idx % num_buf))

@gluon.jit
def add_smem_tma(add_idx, bars, x_smem, y_smem, z_smem, z_desc, offset_M, BLOCK_N: gl. constexpr, num_buf: gl.constexpr, layout: gl.constexpr):
    offset_N = add_idx * BLOCK_N
    bar = bars.index(add_idx % num_buf)
    add_phase = (add_idx // num_buf) & 1
    mbarrier.wait(bar, phase = add_phase)
    x_val = x_smem.index(add_idx % num_buf).load(layout)
    y_val = y_smem.index(add_idx % num_buf).load(layout)
    z_val = x_val + y_val

    tma.store_wait(pendings=0)
    z_smem.store(z_val)
    fence_async_shared()
    tma.async_copy_shared_to_global(z_desc, [offset_M, offset_N], z_smem)


@gluon.jit
def tma_element_add_kernel(
    x_desc, y_desc, z_desc, M, N, BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr, num_buf: gl.constexpr
):

    pid = gl.program_id(0)
    layout: gl.constexpr = gl.BlockedLayout([1,1], [1, 32], [1, 4], [1, 0])
    dtype: gl.constexpr = x_desc.type.block_type.element_ty
    x_smem = gl.allocate_shared_memory(dtype, [num_buf, BLOCK_M, BLOCK_N], x_desc.layout)
    y_smem = gl.allocate_shared_memory(dtype, [num_buf, BLOCK_M, BLOCK_N], y_desc.layout)
    z_smem = gl.allocate_shared_memory(dtype, [BLOCK_M, BLOCK_N], z_desc.layout)

    offset_M = pid * BLOCK_M

    # setup barrier
    bars = gl.allocate_shared_memory(gl.int64, [num_buf, 1], mbarrier.MBarrierLayout())
    for i in gl.static_range(num_buf):
        mbarrier.init(bars.index(i), count = 1)
    
    fence_async_shared()

    write_idx = 0
    add_idx = 0
    for _ in gl.static_range(num_buf - 1):
        write_smem_tma(write_idx, bars, x_smem, y_smem, x_desc, y_desc, offset_M, BLOCK_N, num_buf)
        write_idx += 1

    for _ in range(gl.cdiv(N, BLOCK_N) - (num_buf - 1)):
        write_smem_tma(write_idx, bars, x_smem, y_smem, x_desc, y_desc, offset_M, BLOCK_N, num_buf)
        write_idx += 1

        add_smem_tma(add_idx, bars, x_smem, y_smem, z_smem, z_desc, offset_M, BLOCK_N, num_buf, layout)
        add_idx += 1
    
    for _ in range(num_buf-1):
        add_smem_tma(add_idx, bars, x_smem, y_smem, z_smem, z_desc, offset_M, BLOCK_N, num_buf, layout)
        add_idx += 1

    for i in range(num_buf):
        mbarrier.invalidate(bars.index(i))

    tma.store_wait(pendings = 0)

def tma_element_add(x, y, z, BLOCK_M, BLOCK_N, num_buf):
    M, N = x.shape
    block_shape = [BLOCK_M, BLOCK_N]
    layout = gl.NVMMASharedLayout.get_default_for(block_shape, gl.float32)

    x_desc = TensorDescriptor.from_tensor(x, block_shape, layout)
    y_desc = TensorDescriptor.from_tensor(y, block_shape, layout)
    z_desc = TensorDescriptor.from_tensor(z, block_shape, layout)

    grid = (triton.cdiv(M, BLOCK_M),)
    tma_element_add_kernel[grid](x_desc, y_desc, z_desc, M, N, BLOCK_M, BLOCK_N, num_buf)

# 2d add with tma and warp specialization
# load warp
@gluon.jit
def load_partition(descs, barriers, buffers, offset_M, numel, BLOCK_N: gl.constexpr):
    # unpack arguments
    a_desc = descs[0]
    b_desc = descs[1]
    
    load_empty_bars = barriers[0]
    load_ready_bars = barriers[1]
    
    a_bufs = buffers[0]
    b_bufs = buffers[1]
    
    N = numel[1]

    num_buf: gl.constexpr = a_bufs.type.shape[0]

    for i in range(gl.cdiv(N, BLOCK_N)):
        # initialize buffers and barriers
        index = i % num_buf
        phase = (i // num_buf) & 1
        a_buf = a_bufs.index(index)
        b_buf = b_bufs.index(index)
        load_empty_bar = load_empty_bars.index(index)
        load_ready_bar = load_ready_bars.index(index)

        # mbarrier start with phase 0 incomplete and phase 1 complete
        mbarrier.wait(load_empty_bar, phase ^ 1)

        # load smem now that buffers are empty
        offset_N = i * BLOCK_N
        mbarrier.expect(load_ready_bar, a_desc.block_type.nbytes + b_desc.block_type.nbytes)
        tma.async_copy_global_to_shared(a_desc, [offset_M, offset_N], load_ready_bar, a_buf)
        tma.async_copy_global_to_shared(b_desc, [offset_M, offset_N], load_ready_bar, b_buf)

# store warp to global mem
@gluon.jit
def store_partition(descs, barriers, buffers, offset_M, numel, BLOCK_N: gl.constexpr):
    # unpack arguments
    c_desc = descs[2]
    
    c_empty_bars = barriers[2]
    c_ready_bars = barriers[3]
    
    c_bufs = buffers[2]
    
    N = numel[1]

    num_buf: gl.constexpr = c_bufs.type.shape[0]

    for i in range(gl.cdiv(N, BLOCK_N)):
        # initialize buffers and barriers
        index = i % num_buf
        phase = (i // num_buf) & 1
        c_buf = c_bufs.index(index)
        c_empty_bar = c_empty_bars.index(index)
        c_ready_bar = c_ready_bars.index(index)

        # wait for c_buffers to fill
        mbarrier.wait(c_ready_bar, phase)

        # now buffer is filled store to global memory
        offset_N = i * BLOCK_N
        tma.async_copy_shared_to_global(c_desc, [offset_M, offset_N], c_buf)
        tma.store_wait(num_buf - 1)
        c_empty_bar = c_empty_bars.index((i - num_buf + 1) % num_buf)

        # tell the compute warp that c_buf is empty
        mbarrier.arrive(c_empty_bar, count = 1, pred = i >= num_buf - 1)

    # wait till all warp finish storing c to global memory
    tma.store_wait(0)

# compute
@gluon.jit
def compute_partition(barriers, buffers, N, BLOCK_N:gl.constexpr, layout: gl.constexpr):
    # unpack arguments
    load_empty_bars = barriers[0]
    load_ready_bars = barriers[1]
    c_empty_bars = barriers[2]
    c_ready_bars = barriers[3]
    
    a_bufs = buffers[0]
    b_bufs = buffers[1]
    c_bufs = buffers[2]

    num_load_buf: gl.constexpr = a_bufs.type.shape[0]
    num_store_buf: gl.constexpr = c_bufs.type.shape[0]

    for i in range(gl.cdiv(N, BLOCK_N)):
        load_index = i % num_load_buf
        load_phase = (i // num_load_buf) & 1
        a_buf = a_bufs.index(load_index)
        b_buf = b_bufs.index(load_index)
        load_empty_bar = load_empty_bars.index(load_index)
        load_ready_bar = load_ready_bars.index(load_index)

        # wait for the load warp
        mbarrier.wait(load_ready_bar, load_phase)
        a_val = a_buf.load(layout)
        b_val = b_buf.load(layout)

        # fence and finish loading data before telling the load warp to load new data
        fence_async_shared()
        mbarrier.arrive(load_empty_bar, count = 1)

        c_val = a_val + b_val

        store_index = i % num_store_buf
        store_phase = (i // num_store_buf) & 1
        c_buf = c_bufs.index(store_index)
        c_empty_bar = c_empty_bars.index(store_index)
        c_ready_bar = c_ready_bars.index(store_index)

        # wait for c_buf to clear and store new c_val
        mbarrier.wait(c_empty_bar, store_phase ^ 1)
        c_buf.store(c_val)

        # fence and finish storing before telling the store warp to store c to global memory
        fence_async_shared()
        mbarrier.arrive(c_ready_bar, count = 1)

# warp specialized kernel for 2d element add
@gluon.jit
def warp_specialize_element_add_kernel(
    a_desc, b_desc, c_desc,
    M, N, BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr,
    num_load_buf: gl.constexpr, num_store_buf: gl.constexpr, num_warps: gl.constexpr
):
    # layout in register
    layout: gl.constexpr = gl.BlockedLayout([1, 1], [1, 32], [1, num_warps], [1, 0])

    # allocate space for smem
    a_bufs = gl.allocate_shared_memory(a_desc.dtype, [num_load_buf, BLOCK_M, BLOCK_N], a_desc.layout)
    b_bufs = gl.allocate_shared_memory(b_desc.dtype, [num_load_buf, BLOCK_M, BLOCK_N], b_desc.layout)
    c_bufs = gl.allocate_shared_memory(c_desc.dtype, [num_store_buf, BLOCK_M, BLOCK_N], c_desc.layout)
    load_empty_bars = gl.allocate_shared_memory(gl.int64, [num_load_buf, 1], mbarrier.MBarrierLayout())
    load_ready_bars = gl.allocate_shared_memory(gl.int64, [num_load_buf, 1], mbarrier.MBarrierLayout())
    c_empty_bars = gl.allocate_shared_memory(gl.int64, [num_store_buf, 1], mbarrier.MBarrierLayout())
    c_ready_bars = gl.allocate_shared_memory(gl.int64, [num_store_buf, 1], mbarrier.MBarrierLayout())

    for i in gl.static_range(num_load_buf):
        mbarrier.init(load_empty_bars.index(i), count = 1)
        mbarrier.init(load_ready_bars.index(i), count = 1)

    for i in gl.static_range(num_store_buf):
        mbarrier.init(c_empty_bars.index(i), count = 1)
        mbarrier.init(c_ready_bars.index(i), count = 1)

    descs = (a_desc, b_desc, c_desc)
    barriers = (load_empty_bars, load_ready_bars, c_empty_bars, c_ready_bars)
    buffers = (a_bufs, b_bufs, c_bufs)
    numel = (M, N)

    pid = gl.program_id(0)
    offset_M = pid * BLOCK_M

    gl.warp_specialize([
        (compute_partition, (barriers, buffers, N, BLOCK_N, layout)),
        (load_partition, (descs, barriers, buffers, offset_M, numel, BLOCK_N)),
        (store_partition, (descs, barriers, buffers, offset_M, numel, BLOCK_N))
    ],
    (1, 1), (24, 24)        # minimum # of warp/registers for a warp partition
    )

def warp_specialize_element_add(a, b, c, BLOCK_M, BLOCK_N, num_load_buf, num_store_buf, num_warps):
    M, N = a.shape
    grid = (triton.cdiv(M, BLOCK_M),)

    block_shape = [BLOCK_M, BLOCK_N]
    layout = gl.NVMMASharedLayout.get_default_for(block_shape, gl.float32)
    a_desc = TensorDescriptor.from_tensor(a, block_shape, layout)
    b_desc = TensorDescriptor.from_tensor(b, block_shape, layout)
    c_desc = TensorDescriptor.from_tensor(c, block_shape, layout)

    warp_specialize_element_add_kernel[grid](
        a_desc, b_desc, c_desc, M, N, 
        BLOCK_M, BLOCK_N, num_load_buf, num_store_buf, num_warps = num_warps
    )

# benchmarking
def benchmark_element_add():
    # Define block sizes and resources
    BLOCK_M = 64
    BLOCK_N = 64
    num_buf = 2         # For standard TMA
    num_load_buf = 4    # For WS load pipeline
    num_store_buf = 2   # For WS store double-buffering
    num_warps = 4
    
    print("Benchmarking 2D Elementwise Add: PyTorch vs TMA vs Warp Specialization")
    print("=" * 75)
    print(f"{'Matrix Size (M x N)':<22} | {'Torch (GB/s)':<15} | {'TMA (GB/s)':<15} | {'Warp Spec (GB/s)':<15}")
    print("-" * 75)

    # We will fix M and scale N to see how it handles increasingly long inner loops
    M = 4096
    
    for i in range(10, 16):
        N = 2 ** i
        
        # Allocate contiguous tensors on the GPU
        a = torch.randn(M, N, device='cuda', dtype=torch.float32)
        b = torch.randn(M, N, device='cuda', dtype=torch.float32)
        c_tma = torch.empty_like(a)
        c_ws = torch.empty_like(a)

        # 1. Benchmark PyTorch (Baseline)
        ms_torch = triton.testing.do_bench(lambda: torch.add(a, b))
        
        # 2. Benchmark standard TMA implementation
        ms_tma = triton.testing.do_bench(
            lambda: tma_element_add(a, b, c_tma, BLOCK_M, BLOCK_N, num_buf)
        )
        
        # 3. Benchmark Warp Specialized implementation
        # (Assuming you renamed your wrapper function to 'warp_specialize_element_add')
        ms_ws = triton.testing.do_bench(
            lambda: warp_specialize_element_add(a, b, c_ws, BLOCK_M, BLOCK_N, num_load_buf, num_store_buf, num_warps)
        )

        # Calculate Memory Bandwidth (GB/s)
        # 2 matrix reads (A, B) + 1 matrix write (C) = 3 total matrices
        # 4 bytes per float32 element
        gb = (3 * M * N * 4) / 1e9  
        
        gbps_torch = gb / (ms_torch / 1000)
        gbps_tma = gb / (ms_tma / 1000)
        gbps_ws = gb / (ms_ws / 1000)

        # Print the formatted results
        size_str = f"{M} x {N}"
        print(f"{size_str:<22} | {gbps_torch:<15.2f} | {gbps_tma:<15.2f} | {gbps_ws:<15.2f}")

if __name__ == "__main__":
    # Ensure you are running on Hopper (sm_90) before running WS
    if torch.cuda.get_device_capability()[0] >= 9:
        benchmark_element_add()
    else:
        print("Warp Specialization requires Hopper (sm_90) or newer GPUs.")