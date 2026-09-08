import triton
from typing import Union
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.language.core import _aggregate as aggregate

from triton.experimental.gluon.language.nvidia.hopper import (
    warpgroup_mma,
    warpgroup_mma_wait,
    warpgroup_mma_accumulator,
)

# Instruction Selection

@gluon.constexpr_function
def get_warps_per_cta(BLOCK_M, BLOCK_N, num_warps):
    warps_per_cta = [4, 1]
    m = 16
    # Tile the atom until we have enough warps.
    while warps_per_cta[0] * warps_per_cta[1] != num_warps:
        # Tile along M only if it would not cause broadcasting.
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
    assert n >= 8, "expected to find a valid n"
    return n

@gluon.constexpr_function
def pick_wgmma_layout(dtype, BLOCK_M, BLOCK_N, num_warps, sparse=False):
    m = 16
    k = (512 if sparse else 256) // dtype.primitive_bitwidth
    n = get_instr_shape_n(BLOCK_M, BLOCK_N, num_warps)
    warps_per_cta = get_warps_per_cta(BLOCK_M, BLOCK_N, num_warps)
    return gl.NVMMADistributedLayout(
        version=[3, 0],
        warps_per_cta=warps_per_cta,
        instr_shape=[m, n, k],
    )

# Instructions

@aggregate
class WGMMA:
    acc: Union[warpgroup_mma_accumulator, gl.tensor]
    use_acc: gl.tensor
    layout: gl.constexpr
    sparse: gl.constexpr
    BLOCK_M: gl.constexpr
    BLOCK_N: gl.constexpr

    @gluon.constexpr_function
    def __init__(self, acc, use_acc, layout, BLOCK_M, BLOCK_N, sparse=False):
        self.acc = acc
        self.use_acc = (use_acc)
        self.layout = gl.constexpr(layout)
        self.sparse = gl.constexpr(sparse)
        self.BLOCK_M = gl.constexpr(BLOCK_M)
        self.BLOCK_N = gl.constexpr(BLOCK_N)

    @gluon.jit
    def initialize(dtype: gl.constexpr, BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr, num_warps: gl.constexpr, sparse: gl.constexpr=False):
        mma_layout: gl.constexpr = pick_wgmma_layout(dtype, BLOCK_M, BLOCK_N, num_warps, sparse=sparse)
        acc = gl.zeros((BLOCK_M, BLOCK_N), dtype=gl.float32, layout=mma_layout)
        return WGMMA(acc, gl.to_tensor(False), mma_layout, BLOCK_M, BLOCK_N, sparse=sparse)

    @gluon.jit
    def initialize_from(dtype: gl.constexpr, mma, num_warps: gl.constexpr, sparse: gl.constexpr=False):
        acc, _ = mma.take_result()
        mma_layout: gl.constexpr = pick_wgmma_layout(dtype, mma.BLOCK_M, mma.BLOCK_N, num_warps, sparse=sparse)
        acc = gl.convert_layout(acc, mma_layout, assert_trivial=True)
        return WGMMA(acc, mma.use_acc, mma_layout, mma.BLOCK_M, mma.BLOCK_N, sparse=sparse)

    @gluon.jit
    def issue_async_mma(self, a, b):
        gl.static_assert(not self.sparse, "Instruction shape set for sparse.")
        acc = warpgroup_mma(a, b, self.acc, is_async=True, use_acc=self.use_acc)
        # Note that aggregates don't support in-place mutation, so we need to
        # return a new instance and re-assign it at the callsite.
        return WGMMA(acc, gl.to_tensor(True), self.layout, self.BLOCK_M, self.BLOCK_N, sparse=self.sparse)

    @gluon.jit
    def issue_metadata_load(self, e):
        return e.load(gl.DotOperandLayout(
            operand_index=0,
            parent=self.layout,
            k_width=32 // e.dtype.primitive_bitwidth,
            meta=1,
        ))

    @gluon.jit
    def issue_async_sparse_mma(self, a, e_reg, b):
        gl.static_assert(self.sparse, "Instruction shape set for dense.")
        acc = warpgroup_mma(a, b, self.acc, e=e_reg, is_async=True, use_acc=self.use_acc)
        # Note that aggregates don't support in-place mutation, so we need to
        # return a new instance and re-assign it at the callsite.
        return WGMMA(acc, gl.to_tensor(True), self.layout, self.BLOCK_M, self.BLOCK_N, sparse=self.sparse)

    @gluon.jit
    def wait_num_outstanding(self, num_outstanding: gl.constexpr):
        acc = warpgroup_mma_wait(num_outstanding, (self.acc, ))
        return WGMMA(acc, self.use_acc, self.layout, self.BLOCK_M, self.BLOCK_N, sparse=self.sparse)

    # Take the result and reset the accumulator.
    @gluon.jit
    def take_result(self):
        return self.acc, WGMMA(self.acc, gl.to_tensor(False), self.layout, self.BLOCK_M, self.BLOCK_N, sparse=self.sparse)
    
    @gluon.jit
    def dense_to_2_4_sparse(self, a_dense, num_warps, BLOCK_SIZE_M: gl.constexpr, BLOCK_SIZE_K: gl.constexpr, in_smem=False):
        # 2. Define Register Layout
        if num_warps == 4:
            warp_bases: gl.constexpr = [[16, 0], [32, 0]]
        elif num_warps == 8:
            warp_bases: gl.constexpr = [[16, 0], [32, 0], [64, 0]]
        else:
            warp_bases: gl.constexpr = [[16, 0], [32, 0], [64, 0], [128, 0]]

        a_reg_layout: gl.constexpr = gl.DistributedLinearLayout(
            reg_bases=[[0, 1], [0, 2], [8, 0], [0, 4], [0, 8]],
            lane_bases=[[0, 16], [0, 32], [1, 0], [2, 0], [4, 0]],
            warp_bases=warp_bases,
            block_bases=[],
            shape=[16 * num_warps, 64],
        )   

        if in_smem:
            a_dense = a_dense.load(a_reg_layout)
        else:
            a_dense = gl.convert_layout(a_dense, a_reg_layout)
        
        a_grouped = a_dense.reshape(BLOCK_SIZE_M, BLOCK_SIZE_K // 4, 2, 2)
        a_even, a_odd = a_grouped.split()

        a0, a2 = a_even.split()
        a1, a3 = a_odd.split()

        # 3. Prune 2:4 (select top 2 values algebraically)
        c01 = a0 > a1
        c02 = a0 > a2
        c03 = a0 > a3
        c12 = a1 > a2
        c13 = a1 > a3
        c23 = a2 > a3
    
        c10 = ~c01
        c20 = ~c02
        c21 = ~c12

        b0_bool = (c01 & (c02 | c03)) | (c02 & c03)
        b1_bool = (c10 & (c12 | c13)) | (c12 & c13)
        b2_bool = (c20 & (c21 | c23)) | (c21 & c23)

        nz0 = gl.where(b0_bool, a0, gl.where(b1_bool, a1, a2))
        nz1 = gl.where(b0_bool & b1_bool, a1, gl.where(b2_bool & (b0_bool | b1_bool), a2, a3))

        a_compressed = gl.join(nz0, nz1).reshape(BLOCK_SIZE_M, BLOCK_SIZE_K // 2)

        meta_4 = gl.where(b0_bool,
            gl.where(b1_bool, 4, gl.where(b2_bool, 8, 12)),
            gl.where(b1_bool, gl.where(b2_bool, 9, 13), 14))

        # 4. Pack metadata
        meta_4_reshaped = meta_4.reshape(BLOCK_SIZE_M // 16, 2, 8, BLOCK_SIZE_K // 64, 4, 2, 2)
        meta_4_permuted = meta_4_reshaped.permute(0, 3, 2, 4, 1, 5, 6)
        meta_4_ready = meta_4_permuted.reshape(BLOCK_SIZE_M // 16, BLOCK_SIZE_K, 2, 2)

        meta_even, meta_odd = meta_4_ready.split()
        mn0, mn2 = meta_even.split()
        mn1, mn3 = meta_odd.split()

        meta_reordered = gl.inline_asm_elementwise(
            asm="""
            {
            .reg .b32 t1, t2, t3;
            shl.b32 t1, $2, 4;
            shl.b32 t2, $3, 8;
            shl.b32 t3, $4, 12;
            or.b32 $0, $1, t1;
            or.b32 $0, $0, t2;
            or.b32 $0, $0, t3;
            }
            """,
            constraints="=r,r,r,r,r",
            args=[mn0, mn1, mn2, mn3],
            dtype=gl.int16,
            is_pure=True,
            pack=1,
        )

        a_compressed = gl.convert_layout(
            a_compressed,
            gl.DotOperandLayout(
                operand_index=0,
                parent=self.layout,
                k_width=32 // a_compressed.dtype.primitive_bitwidth,
                meta=0
            )
        )

        meta_reordered = gl.convert_layout(
            meta_reordered,
            gl.DotOperandLayout(
                operand_index=0,
                parent=self.layout,
                k_width=32 // gl.int16.primitive_bitwidth,
                meta=1
            )
        )

        return a_compressed, meta_reordered

# Schedulers

def GroupedPersistentTileScheduler(GROUP_SIZE_M):
    # Bind this as a constexpr so it can be captured.
    GROUP_SIZE_M = gl.constexpr(GROUP_SIZE_M)

    # Like C++ templates!
    @aggregate
    class GroupedPersistentTileSchedulerImpl:
        start_pid: gl.tensor
        num_pid_m: gl.tensor
        num_pid_in_group: gl.tensor
        num_pid: gl.tensor

        @gluon.constexpr_function
        def __init__(self, start_pid, num_pid_m, num_pid_in_group, num_pid):
            self.start_pid = start_pid
            self.num_pid_m = num_pid_m
            self.num_pid_in_group = num_pid_in_group
            self.num_pid = num_pid

        @gluon.jit
        def initialize(M, N, BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr):
            start_pid = gl.program_id(axis=0)
            num_pid_m = gl.cdiv(M, BLOCK_M)
            num_pid_n = gl.cdiv(N, BLOCK_N)
            num_pid_in_group = GROUP_SIZE_M * num_pid_n
            num_pid = num_pid_m * num_pid_n
            return GroupedPersistentTileSchedulerImpl(start_pid, num_pid_m, num_pid_in_group, num_pid)

        @gluon.jit
        def get_num_tiles(self):
            return gl.cdiv(self.num_pid - self.start_pid, gl.num_programs(axis=0))

        @gluon.jit
        def get_tile(self, idx):
            tile_id = self.start_pid + idx * gl.num_programs(axis=0)
            group_id = tile_id // self.num_pid_in_group
            first_pid_m = group_id * GROUP_SIZE_M
            group_size_m = min(self.num_pid_m - first_pid_m, GROUP_SIZE_M)
            pid_m = first_pid_m + (tile_id % group_size_m)
            pid_n = (tile_id % self.num_pid_in_group) // group_size_m
            return pid_m, pid_n

    GroupedPersistentTileSchedulerImpl.__name__ = f"GroupedPersistentTileScheduler({GROUP_SIZE_M.value})"
    return GroupedPersistentTileSchedulerImpl

@aggregate
class PersistentTileScheduler:
    pid_start: gl.tensor
    pid_end: gl.tensor
    num_pid_m: gl.tensor

    @gluon.constexpr_function
    def __init__(self, pid_start, pid_end, num_pid_m):
        self.pid_start = pid_start
        self.pid_end = pid_end
        self.num_pid_m = num_pid_m

    @gluon.jit
    def initialize(M, N, BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr):
        kernel_id = gl.program_id(axis=0)
        num_kernels = gl.num_programs(axis=0)
        num_pid_m = gl.cdiv(M, BLOCK_M)
        num_pid_n = gl.cdiv(N, BLOCK_N)
        num_pid = num_pid_m * num_pid_n
        pid_per_kernel = gl.cdiv(num_pid, num_kernels)
        pid_start = kernel_id * pid_per_kernel
        pid_end = min(pid_start + pid_per_kernel, num_pid)
        return PersistentTileScheduler(pid_start, pid_end, num_pid_m)

    @gluon.jit
    def get_num_tiles(self):
        return self.pid_end - self.pid_start

    @gluon.jit
    def get_tile(self, idx):
        # Delinearize the tile ID along M.
        pid = self.pid_start + idx
        pid_m = pid % self.num_pid_m
        pid_n = pid // self.num_pid_m
        return pid_m, pid_n

@aggregate
class Counter:
    index: gl.tensor
    phase: gl.tensor
    num_barriers: gl.constexpr

    @gluon.constexpr_function
    def __init__(self, index, phase, num_barriers):
        self.index = index
        self.phase = phase
        self.num_barriers = gl.constexpr(num_barriers)

    @gluon.jit
    def create(phase, num_barriers: gl.constexpr):
        return Counter(gl.to_tensor(0), gl.to_tensor(phase), num_barriers)

    @gluon.must_use_result
    @gluon.jit
    def next(self, pred=True):
        incr = self.index + gl.where(pred, 1, 0)
        rollover = incr == self.num_barriers
        index = gl.where(rollover, 0, incr)
        phase = gl.where(rollover, self.phase ^ 1, self.phase)
        return Counter(index, phase, self.num_barriers)