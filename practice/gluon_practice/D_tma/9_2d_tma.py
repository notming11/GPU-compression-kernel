import torch
import triton
import triton.language as tl
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

from triton.experimental.gluon.nvidia.hopper import TensorDescriptor
from triton.experimental.gluon.language.nvidia.hopper import tma, mbarrier, fence_async_shared
from triton.experimental.gluon.language.nvidia.ampere import async_copy as cp

# baseline 2d element add
# helper function
@gluon.jit
def write_smem(
    write_idx,
    smem, ptrs, offsets_M,
    M, N, stride_N,
    BLOCK_N: gl.constexpr, num_buf: gl.constexpr, layout: gl.constexpr
):

    offsets_N = write_idx * BLOCK_N + gl.arange(0, BLOCK_N, layout = gl.SliceLayout(0, layout))
    mask = (offsets_M < M)[:, None] & (offsets_N < N)[None, :]
    cp.async_copy_global_to_shared(smem.index(write_idx % num_buf), ptrs + stride_N * offsets_N[None, :], mask)
    cp.commit_group()

@gluon.jit
def add_smem(
    add_idx,
    x_smem, y_smem, z_ptrs, offsets_M,
    M, N, stride_N,
    BLOCK_N: gl.constexpr, num_buf: gl.constexpr, layout: gl.constexpr
):

    offsets_N = add_idx * BLOCK_N + gl.arange(0, BLOCK_N, layout = gl.SliceLayout(0, layout))
    mask = (offsets_M < M)[:, None] & (offsets_N < N)[None, :]
    x_val = x_smem.index(add_idx % num_buf).load(layout)
    y_val = y_smem.index(add_idx % num_buf).load(layout)
    z_val = x_val + y_val
    gl.store(z_ptrs + stride_N * offsets_N[None, :], z_val, mask)

@gluon.jit
def pipeline_element_add_kernel(
    x_ptr, y_ptr, z_ptr,
    M, N,
    xstride_M, xstride_N,
    ystride_M, ystride_N,
    zstride_M, zstride_N,
    BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr,
    layout: gl.constexpr, smem_layout: gl.constexpr, num_buf: gl.constexpr
):

    pid = gl.program_id(0)
    offsets_M = pid * BLOCK_M + gl.arange(0, BLOCK_M, layout = gl.SliceLayout(1, layout))

    x_ptrs = x_ptr + xstride_M * offsets_M[:, None]
    y_ptrs = y_ptr + ystride_M * offsets_M[:, None]
    z_ptrs = z_ptr + zstride_M * offsets_M[:, None]

    x_smem = gl.allocate_shared_memory(x_ptr.dtype.element_ty, [num_buf, BLOCK_M, BLOCK_N], layout = smem_layout)
    y_smem = gl.allocate_shared_memory(y_ptr.dtype.element_ty, [num_buf, BLOCK_M, BLOCK_N], layout = smem_layout)

    write_idx = 0
    add_idx = 0

    for _ in gl.static_range(num_buf - 1):
        write_smem(write_idx, x_smem, x_ptrs, offsets_M, M, N, xstride_N, BLOCK_N, num_buf, layout)
        write_smem(write_idx, y_smem, y_ptrs, offsets_M, M, N, xstride_N, BLOCK_N, num_buf, layout)
        write_idx += 1

    for _ in range(gl.cdiv(N, BLOCK_N) - (num_buf - 1)):
        write_smem(write_idx, x_smem, x_ptrs, offsets_M, M, N, xstride_N, BLOCK_N, num_buf, layout)
        write_smem(write_idx, y_smem, y_ptrs, offsets_M, M, N, ystride_N, BLOCK_N, num_buf, layout)
        write_idx += 1

        cp.wait_group(num_buf - 1)

        add_smem(add_idx, x_smem, y_smem, z_ptrs, offsets_M, M, N, zstride_N, BLOCK_M, num_buf, layout)
        add_idx += 1

    for i in gl.static_range(num_buf - 1):
        cp.wait_group(num_buf - 2 - i)
        add_smem(add_idx, x_smem, y_smem, z_ptrs, offsets_M, M, N, zstride_N, BLOCK_M, num_buf, layout)
        add_idx += 1

def pipeline_element_add(x, y, z, layout, smem_layout, BLOCK_M, BLOCK_N, num_buf):
    M, N = x.shape
    grid = (triton.cdiv(M, BLOCK_M),)
    pipeline_element_add_kernel[grid](
        x, y, z, 
        M, N, 
        *x.stride(),
        *y.stride(),
        *z.stride(),
        BLOCK_M, BLOCK_N,
        layout, smem_layout, num_buf
    )


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

# benchmark
def run_benchmark():
    BLOCK_M = 64
    BLOCK_N = 64
    num_buf = 4
    layout = gl.BlockedLayout(
        size_per_thread = [1, 1],
        threads_per_warp = [1, 32],
        warps_per_cta = [1, 4],
        order = [1, 0]
    )
    smem_layout = gl.SwizzledSharedLayout(
        vec = 1, per_phase = 1, max_phase = 1, order = [1, 0]
    )

    print(f"{'Matrix Size':<15} | {'Pipeline (GB/s)':<15} | {'TMA (GB/s)':<15}")
    print("-"*60)

    for i in range(10, 16):
        M = 2**i
        N = 2**i
        x = torch.randn(M, N, device='cuda', dtype=torch.float32)
        y = torch.randn(M, N, device='cuda', dtype=torch.float32)
        z = torch.empty_like(x)

        def get_gbps(ms):
            return (3 * x.numel() * x.element_size()) / ms * 1e-6

        ms_pipeline = triton.testing.do_bench(lambda: pipeline_element_add(x, y, z, layout, smem_layout, BLOCK_M, BLOCK_N, num_buf))
        gbps_pipeline = get_gbps(ms_pipeline)

        if not torch.equal(x + y, z):
            print("FAILED")

        ms_tma = triton.testing.do_bench(lambda: tma_element_add(x, y, z, BLOCK_M, BLOCK_N, num_buf))
        gbps_tma = get_gbps(ms_tma)

        if not torch.equal(x + y, z):
            print("FAILED")

        size_str = f"{M}x{N}"
        print(f"{size_str:<15} | {gbps_pipeline:<15.2f} | {gbps_tma:<15.2f}")

if __name__ == "__main__":
    run_benchmark()
