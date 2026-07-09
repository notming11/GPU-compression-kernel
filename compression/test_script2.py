import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

a_warp_bases: gl.constexpr = [[16, 0], [32, 0]]
a_shape: gl.constexpr = [64, 64]
a_pruned_reg_layout: gl.constexpr = gl.DistributedLinearLayout(
    reg_bases=[[0, 1], [0, 2], [8, 0], [0, 16], [0, 32]], 
    lane_bases=[[0, 4], [0, 8], [1, 0], [2, 0], [4, 0]], 
    warp_bases=a_warp_bases, 
    block_bases=[], 
    shape=a_shape
)

meta_local_layout: gl.constexpr = gl.DistributedLinearLayout(
    reg_bases=[[0, 1], [0, 2], [8, 0], [0, 4], [0, 8]], 
    lane_bases=[[0, 16], [0, 32], [1, 0], [2, 0], [4, 0]], 
    warp_bases=a_warp_bases, 
    block_bases=[], 
    shape=[64, 16]
)

@gluon.constexpr_function
def get_warps_per_cta(BLOCK_M, BLOCK_N, num_warps):
    warps_per_cta = [4, 1]
    m = 16
    while warps_per_cta[0] * warps_per_cta[1] != num_warps:
        if BLOCK_M > m * warps_per_cta[0]:
            warps_per_cta[0] *= 2
        else:
            warps_per_cta[1] *= 2
    return warps_per_cta

@gluon.constexpr_function
def get_instr_shape_n(BLOCK_M, BLOCK_N, num_warps):
    m = 16
    mReps = triton.cdiv(BLOCK_M, m)
    nReps = triton.cdiv(num_warps, mReps)
    maxN = max(BLOCK_N // nReps, 8)
    n = 256
    while n > maxN or BLOCK_N % n != 0:
        n -= 8
    return n

@gluon.constexpr_function
def pick_sparse_wgmma_layout(dtype, BLOCK_M, BLOCK_N, num_warps):
    m = 16
    k = 32
    n = get_instr_shape_n(BLOCK_M, BLOCK_N, num_warps)
    warps_per_cta = get_warps_per_cta(BLOCK_M, BLOCK_N, num_warps)
    return gl.NVMMADistributedLayout(
        version=[3, 0],
        warps_per_cta=warps_per_cta,
        instr_shape=[m, n, k],
    )

@gluon.jit
def create_metadata(meta_1, meta_2):
    return meta_1 | (meta_2 << 4)

@gluon.jit
def create_metadata_8(meta_1, meta_2):
    return meta_1 | (meta_2 << 8)

@gluon.jit
def test_kernel(a_ptr, BLOCK_M: gl.constexpr, BLOCK_K: gl.constexpr):
    a_pruned = gl.zeros((BLOCK_M, BLOCK_K), dtype=gl.int8, layout=a_pruned_reg_layout)
    
    a_grouped = a_pruned.reshape(BLOCK_M, BLOCK_K // 4, 2, 2)
    a_even, a_odd = a_grouped.split()

    a0, a2 = a_even.split()
    a1, a3 = a_odd.split()

    m0 = a0 != 0
    m1 = a1 != 0
    m3 = a3 != 0

    bit0 = ~m0 & m1
    bit1 = ~m0 & ~m1
    bit2 = (m0 & m1) | (~m0 & ~m1) | m3
    bit3 = (~m0 & m1) | ~m1

    idx0 = bit0 | (bit1.to(gl.int16) << 1)
    idx1 = bit2 | (bit3.to(gl.int16) << 1)

    meta_4 = idx0 | (idx1 << 2)

    # Convert layout to make the reduction local!
    meta_4_local = gl.convert_layout(meta_4, meta_local_layout)

    meta_grouped = meta_4_local.reshape(BLOCK_M, BLOCK_K // 16, 2, 2)
    meta = gl.reduce(gl.reduce(meta_grouped, 3, create_metadata), 2, create_metadata_8)
    meta_reshaped = meta.reshape(BLOCK_M // 16, 2, 8, BLOCK_K // 64, 4)
    meta_reordered = meta_reshaped.permute(0, 3, 2, 4, 1).reshape(BLOCK_M // 16, BLOCK_K)

    gl.static_print(meta_reordered.type.layout.format_tensor_view((BLOCK_M // 16, BLOCK_K)))
    
test_kernel[(1,)](None, 128, 128)
print("done")
