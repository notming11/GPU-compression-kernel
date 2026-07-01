import torch
import triton

from triton.experimental import gluon
from triton.experimental.gluon import language as gl

from triton.experimental.gluon.nvidia.hopper import TensorDescriptor
from triton.language.core import _aggregate as aggregate

from triton.experimental.gluon.language.nvidia.hopper import (
    tma,
    mbarrier,
    fence_async_shared,
)

from common import (
    WGMMA,
    GroupedPersistentTileScheduler
)
import os

from prune import prune_2_4
from compress_2_4 import compress_dense_to_sparse

@aggregate
class SparsePartitionArgs:
    a_pruned_desc: tma.tensor_descriptor
    b_desc: tma.tensor_descriptor
    c_desc: tma.tensor_descriptor
    
    a_pruned_bufs: gl.shared_memory_descriptor
    a_comp_bufs: gl.shared_memory_descriptor
    e_bufs: gl.shared_memory_descriptor
    b_bufs: gl.shared_memory_descriptor
    
    a_pruned_empty_bars: gl.shared_memory_descriptor
    a_pruned_ready_bars: gl.shared_memory_descriptor
    
    a_comp_empty_bars: gl.shared_memory_descriptor
    a_comp_ready_bars: gl.shared_memory_descriptor

    b_empty_bars: gl.shared_memory_descriptor
    b_ready_bars: gl.shared_memory_descriptor
    
    acc_bufs: gl.shared_memory_descriptor
    acc_empty_bars: gl.shared_memory_descriptor
    acc_ready_bars: gl.shared_memory_descriptor
    
    SUBTILE_FACTOR: gl.constexpr
    num_warps_compute: gl.constexpr
    num_warps_compress: gl.constexpr

    @gluon.constexpr_function
    def __init__(self, a_pruned_desc, b_desc, c_desc,
                 a_pruned_bufs, a_comp_bufs, e_bufs, b_bufs,
                 a_pruned_empty_bars, a_pruned_ready_bars,
                 a_comp_empty_bars, a_comp_ready_bars,
                 b_empty_bars, b_ready_bars,
                 acc_bufs, acc_empty_bars, acc_ready_bars, 
                 SUBTILE_FACTOR, num_warps_compute, num_warps_compress):
        self.a_pruned_desc = a_pruned_desc
        self.b_desc = b_desc
        self.c_desc = c_desc
        self.a_pruned_bufs = a_pruned_bufs
        self.a_comp_bufs = a_comp_bufs
        self.e_bufs = e_bufs
        self.b_bufs = b_bufs
        self.a_pruned_empty_bars = a_pruned_empty_bars
        self.a_pruned_ready_bars = a_pruned_ready_bars
        self.a_comp_empty_bars = a_comp_empty_bars
        self.a_comp_ready_bars = a_comp_ready_bars
        self.b_empty_bars = b_empty_bars
        self.b_ready_bars = b_ready_bars
        self.acc_bufs = acc_bufs
        self.acc_empty_bars = acc_empty_bars
        self.acc_ready_bars = acc_ready_bars
        self.SUBTILE_FACTOR = gl.constexpr(SUBTILE_FACTOR)
        self.num_warps_compute = gl.constexpr(num_warps_compute)
        self.num_warps_compress = gl.constexpr(num_warps_compress)

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

@gluon.jit
def _split_n(x, SUBTILE_FACTOR: gl.constexpr):
    split_count: gl.constexpr = SUBTILE_FACTOR.bit_length() - 1  # log2
    xs = (x, )
    for _ in gl.static_range(split_count):
        next_xs = ()
        for j in gl.static_range(len(xs)):
            x = xs[j]
            next_xs += x.reshape(x.shape[0], 2, x.shape[1] // 2).permute(0, 2, 1).split()
        xs = next_xs
    return xs

@gluon.jit
def sparse_matmul_load_partition(p, SchedulerImpl: gl.constexpr):
    BLOCK_M: gl.constexpr = p.a_pruned_desc.block_type.shape[0]
    BLOCK_N: gl.constexpr = p.b_desc.block_type.shape[1]
    BLOCK_K: gl.constexpr = p.b_desc.block_type.shape[0]
    K = p.b_desc.shape[0]

    state_a = Counter.create(1, p.a_pruned_empty_bars.shape[0])
    state_b = Counter.create(1, p.b_empty_bars.shape[0])
    scheduler = SchedulerImpl.initialize(p.c_desc.shape[0], p.c_desc.shape[1], BLOCK_M, BLOCK_N)

    for idx in range(scheduler.get_num_tiles()):
        pid_m, pid_n = scheduler.get_tile(idx)
        off_m = pid_m * BLOCK_M
        off_n = pid_n * BLOCK_N

        for k in range(0, K, BLOCK_K):
            bar_a = p.a_pruned_ready_bars.index(state_a.index)
            bar_b = p.b_ready_bars.index(state_b.index)
            mbarrier.wait(p.a_pruned_empty_bars.index(state_a.index), state_a.phase)
            mbarrier.wait(p.b_empty_bars.index(state_b.index), state_b.phase)

            mbarrier.expect(bar_a, p.a_pruned_desc.block_type.nbytes)
            mbarrier.expect(bar_b, p.b_desc.block_type.nbytes)
            
            tma.async_copy_global_to_shared(p.a_pruned_desc, [off_m, k], bar_a, p.a_pruned_bufs.index(state_a.index))
            tma.async_copy_global_to_shared(p.b_desc, [k, off_n], bar_b, p.b_bufs.index(state_b.index))
            
            state_a = state_a.next()
            state_b = state_b.next()

@gluon.jit
def sparse_matmul_compress_partition(p, SchedulerImpl: gl.constexpr):
    BLOCK_M: gl.constexpr = p.a_pruned_desc.block_type.shape[0]
    BLOCK_N: gl.constexpr = p.b_desc.block_type.shape[1]
    BLOCK_K: gl.constexpr = p.b_desc.block_type.shape[0]
    K = p.b_desc.shape[0]

    num_warps = p.num_warps_compress

    state_a = Counter.create(0, p.a_pruned_empty_bars.shape[0])
    state_comp = Counter.create(1, p.a_comp_empty_bars.shape[0])
    
    scheduler = SchedulerImpl.initialize(p.c_desc.shape[0], p.c_desc.shape[1], BLOCK_M, BLOCK_N)

    a_warp_bases: gl.constexpr = [[16, 0], [32, 0]] if num_warps == 4 else ([[16, 0], [32, 0], [0, 0]] if num_warps == 8 else [[16, 0], [32, 0], [0, 0], [0, 0]])
    a_shape: gl.constexpr = [64, 64]
    a_pruned_reg_layout: gl.constexpr = gl.DistributedLinearLayout(
        reg_bases=[[0, 1], [0, 2], [0, 4], [0, 8], [8, 0]], 
        lane_bases=[[0, 16], [0, 32], [1, 0], [2, 0], [4, 0]], 
        warp_bases=a_warp_bases, 
        block_bases=[], 
        shape=a_shape
    )

    for _ in range(scheduler.get_num_tiles()):
        for _ in range(0, K, BLOCK_K):
            mbarrier.wait(p.a_pruned_ready_bars.index(state_a.index), state_a.phase)
            mbarrier.wait(p.a_comp_empty_bars.index(state_comp.index), state_comp.phase)

            a_pruned_smem = p.a_pruned_bufs.index(state_a.index)
            a_pruned = a_pruned_smem.load(a_pruned_reg_layout)

            a_grouped = a_pruned.reshape(BLOCK_M, BLOCK_K // 4, 2, 2)
            a_even, a_odd = a_grouped.split()

            a0, a2 = a_even.split()
            a1, a3 = a_odd.split()

            idx0 = (~(a0 != 0) & (a1 != 0)) | ((~(a0 != 0) & ~(a1 != 0)) << 1)
            idx1 = (((a0 != 0) & (a1 != 0)) | (~(a0 != 0) & ~(a1 != 0)) | (a3 != 0)) | (((~(a0 != 0) & (a1 != 0)) | ~(a1 != 0)) << 1)

            nz0 = gl.where(idx0 == 0, a0, gl.where(idx0 == 1, a1, gl.where(idx0 == 2, a2, a3)))
            nz1 = gl.where(idx1 == 0, a0, gl.where(idx1 == 1, a1, gl.where(idx1 == 2, a2, a3)))

            a_compressed = gl.join(nz0, nz1)
            a_compressed = a_compressed.reshape(BLOCK_M, BLOCK_K // 2)

            meta_4 = idx0 | (idx1 << 2)

            meta_grouped = meta_4.reshape(BLOCK_M, BLOCK_K // 16, 2, 2)
            meta_even, meta_odd = meta_grouped.split()

            mn0, mn2 = meta_even.split()
            mn1, mn3 = meta_odd.split()

            mn0 = mn0.to(gl.int16)
            mn1 = mn1.to(gl.int16)
            mn2 = mn2.to(gl.int16)
            mn3 = mn3.to(gl.int16)

            meta = mn0 | (mn1 << 4) | (mn2 << 8) | (mn3 << 12)
            meta_reshaped = meta.reshape(BLOCK_M // 16, 2, 8, BLOCK_K // 64, 4)
            meta_reordered = meta_reshaped.permute(0, 3, 2, 4, 1).reshape(BLOCK_M // 16, BLOCK_K)

            a_comp_smem = p.a_comp_bufs.index(state_comp.index)
            e_smem = p.e_bufs.index(state_comp.index)

            a_comp_smem.store(a_compressed)
            e_smem.store(meta_reordered)

            fence_async_shared()

            mbarrier.arrive(p.a_pruned_empty_bars.index(state_a.index), count=1)
            mbarrier.arrive(p.a_comp_ready_bars.index(state_comp.index), count=1)

            state_a = state_a.next()
            state_comp = state_comp.next()

@gluon.jit
def store_acc_to_smem_subtile(p, mma, acc_state):
    mma = mma.wait_num_outstanding(0)
    acc, mma = mma.take_result()
    accs = _split_n(acc, p.SUBTILE_FACTOR)

    for i in gl.static_range(p.SUBTILE_FACTOR):
        mbarrier.wait(p.acc_empty_bars.index(acc_state.index), acc_state.phase)
        c_buf = p.acc_bufs.index(acc_state.index)

        c_buf.store(accs[i].to(p.c_desc.dtype))
        fence_async_shared()
        mbarrier.arrive(p.acc_ready_bars.index(acc_state.index), count=1)
        acc_state = acc_state.next()

    return acc_state

@gluon.jit
def sparse_matmul_compute_partition(p, SchedulerImpl: gl.constexpr):
    BLOCK_M: gl.constexpr = p.a_pruned_desc.block_type.shape[0]
    BLOCK_N: gl.constexpr = p.b_desc.block_type.shape[1]
    BLOCK_K: gl.constexpr = p.b_desc.block_type.shape[0]
    K = p.b_desc.shape[0]
    dtype: gl.constexpr = p.a_pruned_desc.dtype

    state_comp = Counter.create(0, p.a_comp_empty_bars.shape[0])
    state_b = Counter.create(0, p.b_empty_bars.shape[0])
    acc_state = Counter.create(1, p.acc_empty_bars.shape[0])

    release_comp = Counter.create(0, p.a_comp_empty_bars.shape[0])
    release_b = Counter.create(0, p.b_empty_bars.shape[0])

    scheduler = SchedulerImpl.initialize(p.c_desc.shape[0], p.c_desc.shape[1], BLOCK_M, BLOCK_N)

    outstanding_mmas: gl.constexpr = 0
    global_k_iter = 0

    for _ in range(scheduler.get_num_tiles()):
        mma = WGMMA.initialize(dtype, BLOCK_M, BLOCK_N, p.num_warps_compute, sparse=True)

        for _ in range(0, K, BLOCK_K):
            mbarrier.wait(p.a_comp_ready_bars.index(state_comp.index), state_comp.phase)
            mbarrier.wait(p.b_ready_bars.index(state_b.index), state_b.phase)

            mma = mma.wait_num_outstanding(outstanding_mmas)
            mma = mma.issue_async_sparse_mma(p.a_comp_bufs.index(state_comp.index), p.e_bufs.index(state_comp.index), p.b_bufs.index(state_b.index))

            if global_k_iter >= outstanding_mmas + 1:
                mbarrier.arrive(p.a_comp_empty_bars.index(release_comp.index), count=1)
                mbarrier.arrive(p.b_empty_bars.index(release_b.index), count=1)
                release_comp = release_comp.next()
                release_b = release_b.next()

            state_comp = state_comp.next()
            state_b = state_b.next()
            global_k_iter += 1

        acc_state = store_acc_to_smem_subtile(p, mma, acc_state)

@gluon.jit
def sparse_matmul_store_partition(p, SchedulerImpl: gl.constexpr):
    BLOCK_M: gl.constexpr = p.c_desc.block_type.shape[0]
    SPLIT_N: gl.constexpr = p.c_desc.block_type.shape[1]
    BLOCK_N: gl.constexpr = SPLIT_N * p.SUBTILE_FACTOR

    state = Counter.create(0, p.acc_empty_bars.shape[0])
    scheduler = SchedulerImpl.initialize(p.c_desc.shape[0], p.c_desc.shape[1], BLOCK_M, BLOCK_N)

    num_buffers: gl.constexpr = 2
    outstanding_stores: gl.constexpr = 1
    store_iter = 0

    for idx in range(scheduler.get_num_tiles()):
        pid_m, pid_n = scheduler.get_tile(idx)
        off_m, off_n = pid_m * BLOCK_M, pid_n * BLOCK_N

        for i in gl.static_range(p.SUBTILE_FACTOR):
            mbarrier.wait(p.acc_ready_bars.index(state.index), state.phase)
            c_buf = p.acc_bufs.index(state.index)

            tma.async_copy_shared_to_global(p.c_desc, [off_m, off_n + i * SPLIT_N], c_buf)

            if store_iter >= outstanding_stores:
                tma.store_wait(outstanding_stores)
                empty_idx = (store_iter - outstanding_stores) % num_buffers
                mbarrier.arrive(p.acc_empty_bars.index(empty_idx), count=1)

            state = state.next()
            store_iter += 1

    tma.store_wait(0)

@gluon.jit
def idle_partition(p, SchedulerImpl: gl.constexpr):
    # Dummy partition to absorb unused warps
    pass

@gluon.jit
def sparse_matmul_warp_specialized_kernel(
    a_pruned_desc, b_desc, c_desc, SchedulerImpl: gl.constexpr,
    M, N, K, BLOCK_SIZE_M: gl.constexpr, BLOCK_SIZE_N: gl.constexpr, BLOCK_SIZE_K: gl.constexpr,
    num_buffers: gl.constexpr, SUBTILE_FACTOR: gl.constexpr,
    num_warps_compute: gl.constexpr, num_warps_compress: gl.constexpr,
    a_comp_layout: gl.constexpr, e_layout: gl.constexpr,
    a_comp_shape_0: gl.constexpr, a_comp_shape_1: gl.constexpr,
    e_shape_0: gl.constexpr, e_shape_1: gl.constexpr):
    
    dtype: gl.constexpr = a_pruned_desc.dtype

    a_pruned_bufs = gl.allocate_shared_memory(dtype, [num_buffers] + a_pruned_desc.block_type.shape, a_pruned_desc.layout)
    
    a_comp_bufs = gl.allocate_shared_memory(dtype, [num_buffers, a_comp_shape_0, a_comp_shape_1], a_comp_layout)
    e_bufs = gl.allocate_shared_memory(gl.int16, [num_buffers, e_shape_0, e_shape_1], e_layout)
    
    b_bufs = gl.allocate_shared_memory(dtype, [num_buffers] + b_desc.block_type.shape, b_desc.layout)

    a_pruned_empty_bars = gl.allocate_shared_memory(gl.int64, [num_buffers, 1], mbarrier.MBarrierLayout())
    a_pruned_ready_bars = gl.allocate_shared_memory(gl.int64, [num_buffers, 1], mbarrier.MBarrierLayout())
    
    a_comp_empty_bars = gl.allocate_shared_memory(gl.int64, [num_buffers, 1], mbarrier.MBarrierLayout())
    a_comp_ready_bars = gl.allocate_shared_memory(gl.int64, [num_buffers, 1], mbarrier.MBarrierLayout())
    
    b_empty_bars = gl.allocate_shared_memory(gl.int64, [num_buffers, 1], mbarrier.MBarrierLayout())
    b_ready_bars = gl.allocate_shared_memory(gl.int64, [num_buffers, 1], mbarrier.MBarrierLayout())
    
    for i in gl.static_range(num_buffers):
        mbarrier.init(a_pruned_empty_bars.index(i), count=1)
        mbarrier.init(a_pruned_ready_bars.index(i), count=1)
        mbarrier.init(a_comp_empty_bars.index(i), count=1)
        mbarrier.init(a_comp_ready_bars.index(i), count=1)
        mbarrier.init(b_empty_bars.index(i), count=1)
        mbarrier.init(b_ready_bars.index(i), count=1)

    acc_bufs = gl.allocate_shared_memory(dtype, [2] + c_desc.block_type.shape, c_desc.layout)
    acc_empty_bars = gl.allocate_shared_memory(gl.int64, [2, 1], mbarrier.MBarrierLayout())
    acc_ready_bars = gl.allocate_shared_memory(gl.int64, [2, 1], mbarrier.MBarrierLayout())

    for i in gl.static_range(2):
        mbarrier.init(acc_empty_bars.index(i), count=1)
        mbarrier.init(acc_ready_bars.index(i), count=1)

    p = SparsePartitionArgs(a_pruned_desc, b_desc, c_desc,
                            a_pruned_bufs, a_comp_bufs, e_bufs, b_bufs,
                            a_pruned_empty_bars, a_pruned_ready_bars,
                            a_comp_empty_bars, a_comp_ready_bars,
                            b_empty_bars, b_ready_bars,
                            acc_bufs, acc_empty_bars, acc_ready_bars,
                            SUBTILE_FACTOR, num_warps_compute, num_warps_compress)

    num_warps_total = gl.num_programs(axis=1) # Triton models warps in axis 1 or num_warps builtin?
    # Actually we just pass fixed list
    idle_warps: gl.constexpr = 16 - num_warps_compute - num_warps_compress - 1 - 1
    gl.warp_specialize([
        (sparse_matmul_compute_partition, (p, SchedulerImpl)),
        (sparse_matmul_compress_partition, (p, SchedulerImpl)),
        (sparse_matmul_load_partition, (p, SchedulerImpl)),
        (sparse_matmul_store_partition, (p, SchedulerImpl)),
        (idle_partition, (p, SchedulerImpl)),
    ], [num_warps_compress, 1, 1, idle_warps], [64, 64, 24, 24, 24])

def sparse_matmul_get_configs(pre_hook=None):
    def valid(BM, BN, BK, warps_compute, warps_compress, buffers, SF):
        if (BN // SF) < 16: return False
        return True
    
    return [
        triton.Config(
            {
                "BLOCK_SIZE_M": BM,
                "BLOCK_SIZE_N": BN,
                "BLOCK_SIZE_K": BK,
                "num_buffers": buffers,
                "SUBTILE_FACTOR": SF,
                "num_warps_compress": warps_compress,
                "num_warps_compute": warps_compute,
                "a_comp_layout": gl.NVMMASharedLayout.get_default_for([BM, BK // 2], gl.float16),
                "e_layout": gl.NVMMASharedLayout.get_default_for([BM // 16, BK], gl.int16),
                "a_comp_shape_0": BM,
                "a_comp_shape_1": BK // 2,
                "e_shape_0": BM // 16,
                "e_shape_1": BK,
            },
            num_warps=16, # Fixed to 16
            pre_hook=pre_hook,
        )
        for BM in (128,)
        for BN in (128,)
        for BK in (64,)
        for warps_compute in (8,)
        for warps_compress in (4,)
        for buffers in (3,)
        for SF in (1,)
        if valid(BM, BN, BK, warps_compute, warps_compress, buffers, SF)
    ]

def sparse_matmul_tma_set_block_size_hook(nargs):
    block_m = nargs["BLOCK_SIZE_M"]
    block_n = nargs["BLOCK_SIZE_N"]
    block_k = nargs["BLOCK_SIZE_K"]
    split_n = nargs["BLOCK_SIZE_N"] // nargs["SUBTILE_FACTOR"]

    nargs["a_pruned_desc"].block_shape = [block_m, block_k]
    nargs["b_desc"].block_shape = [block_k, block_n]
    nargs["c_desc"].block_shape = [block_m, split_n]

    nargs["a_pruned_desc"].layout = gl.NVMMASharedLayout.get_default_for(nargs["a_pruned_desc"].block_shape, gl.float16)
    nargs["b_desc"].layout = gl.NVMMASharedLayout.get_default_for(nargs["b_desc"].block_shape, gl.float16)
    nargs["c_desc"].layout = gl.NVMMASharedLayout.get_default_for(nargs["c_desc"].block_shape, gl.float16)

sparse_ws_kernel = triton.autotune(
    configs=sparse_matmul_get_configs(pre_hook=sparse_matmul_tma_set_block_size_hook),
    key=["M", "N", "K"],
)(sparse_matmul_warp_specialized_kernel)

def run_sparse_ws_matmul(A_pruned, B):
    M, K = A_pruned.shape[0], A_pruned.shape[1]
    N = B.shape[1]

    c = torch.empty((M, N), device=A_pruned.device, dtype=torch.float16)
    dummy_block = [1, 1]
    dummy_layout_f16 = gl.NVMMASharedLayout.get_default_for(dummy_block, gl.float16)
    
    a_pruned_desc = TensorDescriptor.from_tensor(A_pruned, dummy_block, dummy_layout_f16)
    b_desc = TensorDescriptor.from_tensor(B, dummy_block, dummy_layout_f16)
    c_desc = TensorDescriptor.from_tensor(c, dummy_block, dummy_layout_f16)

    def grid(meta):
        num_sms = torch.cuda.get_device_properties("cuda").multi_processor_count
        num_pid = triton.cdiv(M, meta["BLOCK_SIZE_M"]) * triton.cdiv(N, meta["BLOCK_SIZE_N"])
        return (min(num_sms, num_pid), )
    
    sparse_ws_kernel[grid](a_pruned_desc, b_desc, c_desc, GroupedPersistentTileScheduler(8), M, N, K)

    return c

if __name__ == "__main__":
    os.environ["MLIR_ENABLE_DUMP"]="1"
    os.environ["MLIR_DUMP_PATH"] = "./MLIR_DUMP/7.6"
    os.environ["TRITON_ALWAYS_COMPILE"]="1"

    M, N, K = 128, 128, 128

    print(f"Testing 7.6_compression_ws: M={M}, N={N}, K={K}...", end=" ", flush=True)

    A = torch.randn(M, K, device="cuda", dtype=torch.float16)
    B = torch.randn((K, N), device="cuda", dtype=torch.float16)

    A_pruned = prune_2_4(A)

    C = run_sparse_ws_matmul(A_pruned, B)
    C_ref = A_pruned @ B

    torch.testing.assert_close(C_ref, C, rtol=1e-3, atol=1e-1)
    print("PASSED")

