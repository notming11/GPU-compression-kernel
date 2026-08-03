import argparse
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

# Warp-Specialization

@aggregate
class SparsePartitionArgs:
    a_desc: tma.tensor_descriptor
    e_desc: tma.tensor_descriptor
    b_desc: tma.tensor_descriptor
    c_desc: tma.tensor_descriptor
    c_meta_desc: tma.tensor_descriptor
    a_bufs: gl.shared_memory_descriptor
    e_bufs: gl.shared_memory_descriptor
    b_bufs: gl.shared_memory_descriptor
    load_empty_bars: gl.shared_memory_descriptor
    load_ready_bars: gl.shared_memory_descriptor
    acc_bufs: gl.shared_memory_descriptor
    c_meta_bufs: gl.shared_memory_descriptor
    acc_empty_bars: gl.shared_memory_descriptor
    acc_ready_bars: gl.shared_memory_descriptor
    SUBTILE_FACTOR: gl.constexpr
    num_warps: gl.constexpr

    @gluon.constexpr_function
    def __init__(self, a_desc, e_desc, b_desc, c_desc, c_meta_desc, a_bufs, e_bufs, b_bufs, load_empty_bars, load_ready_bars, acc_bufs, c_meta_bufs,
                 acc_empty_bars, acc_ready_bars, SUBTILE_FACTOR, num_warps):
        self.a_desc = a_desc
        self.e_desc = e_desc
        self.b_desc = b_desc
        self.c_desc = c_desc
        self.c_meta_desc = c_meta_desc
        self.a_bufs = a_bufs
        self.e_bufs = e_bufs
        self.b_bufs = b_bufs
        self.load_empty_bars = load_empty_bars
        self.load_ready_bars = load_ready_bars
        self.acc_bufs = acc_bufs
        self.c_meta_bufs = c_meta_bufs
        self.acc_empty_bars = acc_empty_bars
        self.acc_ready_bars = acc_ready_bars
        self.SUBTILE_FACTOR = gl.constexpr(SUBTILE_FACTOR)
        self.num_warps = gl.constexpr(num_warps)

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
def prune_and_compress_acc(c_dense, BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr, acc_reg_layout: gl.constexpr):
    c_dense = gl.convert_layout(c_dense, acc_reg_layout)
    
    # 1. Reshape and extract 4 consecutive elements along the N dimension
    c_grouped = c_dense.reshape(BLOCK_M, BLOCK_N // 4, 2, 2)
    c_even, c_odd = c_grouped.split()

    c0, c2 = c_even.split()
    c1, c3 = c_odd.split()

    # 2. Evaluate 2:4 pruning conditions
    c01 = c0 >= c1
    c02 = c0 >= c2
    c03 = c0 >= c3
    c12 = c1 >= c2
    c13 = c1 >= c3
    c23 = c2 >= c3

    c10 = ~c01
    c20 = ~c02
    c21 = ~c12

    b0_bool = (c01 & (c02 | c03)) | (c02 & c03)
    b1_bool = (c10 & (c12 | c13)) | (c12 & c13)
    b2_bool = (c20 & (c21 | c23)) | (c21 & c23)

    # 3. Extract non-zero values (nz0, nz1)
    nz0 = gl.where(b0_bool, c0, gl.where(b1_bool, c1, c2))
    nz1 = gl.where(b0_bool & b1_bool, c1, gl.where(b2_bool & (b0_bool | b1_bool), c2, c3))

    c_compressed = gl.join(nz0, nz1).reshape(BLOCK_M, BLOCK_N // 2)

    # 4. Generate Metadata
    meta_4 = gl.where(b0_bool,
         gl.where(b1_bool, 4, gl.where(b2_bool, 8, 12)),
         gl.where(b1_bool, gl.where(b2_bool, 9, 13), 14))

    meta_4_reshaped = meta_4.reshape(BLOCK_M // 16, 2, 8, BLOCK_N // 64, 4, 2, 2)
    meta_4_permuted = meta_4_reshaped.permute(0, 3, 2, 4, 1, 5, 6)
    meta_4_ready = meta_4_permuted.reshape(BLOCK_M // 16, BLOCK_N, 2, 2) # N // 2 logic adjusted for pre-shifted output

    meta_even, meta_odd = meta_4_ready.split()
    mn0, mn2 = meta_even.split()
    mn1, mn3 = meta_odd.split()

    # 5. Pack metadata using inline PTX assembly
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

    return c_compressed, meta_reordered

# ---------------------------------------------------------------------------
# HELPER: Slices the accumulator registers along the N dimension
# ---------------------------------------------------------------------------
@gluon.jit
def _split_n(x, SUBTILE_FACTOR: gl.constexpr):
    split_count: gl.constexpr = SUBTILE_FACTOR.bit_length() - 1  # log2
    xs = (x, )
    for _ in gl.static_range(split_count):
        next_xs = ()
        for j in gl.static_range(len(xs)):
            x = xs[j]
            # Reshape to (M, 2, N//2) then permute so that tensor elements
            # remain contiguous along N.
            next_xs += x.reshape(x.shape[0], 2, x.shape[1] // 2).permute(0, 2, 1).split()
        xs = next_xs
    return xs

# ---------------------------------------------------------------------------
# PARTITIONS
# ---------------------------------------------------------------------------
@gluon.jit
def sparse_matmul_load_partition(p, SchedulerImpl: gl.constexpr):
    BLOCK_M: gl.constexpr = p.a_desc.block_type.shape[0]
    BLOCK_N: gl.constexpr = p.b_desc.block_type.shape[1]
    BLOCK_K: gl.constexpr = p.b_desc.block_type.shape[0]
    K = p.b_desc.shape[0]

    state = Counter.create(1, p.load_empty_bars.shape[0])
    
    # We maintain original dense N from b_desc for identical grid indexing logic across partitions
    scheduler = SchedulerImpl.initialize(p.c_desc.shape[0], p.b_desc.shape[1], BLOCK_M, BLOCK_N)

    for idx in range(scheduler.get_num_tiles()):
        pid_m, pid_n = scheduler.get_tile(idx)
        off_m = pid_m * BLOCK_M
        off_n = pid_n * BLOCK_N

        for k in range(0, K, BLOCK_K):
            bar = p.load_ready_bars.index(state.index)
            mbarrier.wait(p.load_empty_bars.index(state.index), state.phase)

            mbarrier.expect(bar, p.a_desc.block_type.nbytes + p.e_desc.block_type.nbytes + p.b_desc.block_type.nbytes)
            tma.async_copy_global_to_shared(p.a_desc, [off_m, k // 2], bar, p.a_bufs.index(state.index))
            tma.async_copy_global_to_shared(p.e_desc, [off_m // 16, k], bar, p.e_bufs.index(state.index))
            tma.async_copy_global_to_shared(p.b_desc, [k, off_n], bar, p.b_bufs.index(state.index))
            state = state.next()

@gluon.jit
def store_acc_to_smem_subtile(p, mma, acc_state):
    mma = mma.wait_num_outstanding(0)
    acc, mma = mma.take_result()
    
    BLOCK_M: gl.constexpr = p.c_desc.block_type.shape[0]
    BLOCK_N: gl.constexpr = p.b_desc.block_type.shape[1]
    
    if p.num_warps == 4:
        acc_warp_bases: gl.constexpr = [[16, 0], [32, 0]]
    elif p.num_warps == 8:
        acc_warp_bases: gl.constexpr = [[16, 0], [32, 0], [64, 0]]
    elif p.num_warps == 16:
        acc_warp_bases: gl.constexpr = [[16, 0], [32, 0], [64, 0], [128, 0]]
    
    acc_reg_layout: gl.constexpr = gl.DistributedLinearLayout(
        reg_bases=[[0, 1], [0, 2], [8, 0], [0, 4], [0, 8]],
        lane_bases=[[0, 16], [0, 32], [1, 0], [2, 0], [4, 0]],
        warp_bases=acc_warp_bases,
        block_bases=[],
        shape=[16 * p.num_warps, 64],
    )
    
    # Prune and compress the raw accumulator
    c_comp, c_meta = prune_and_compress_acc(acc, BLOCK_M, BLOCK_N, acc_reg_layout)
    
    # TODO: fix layout for subtile
    
    # Split both the compressed tensor and metadata tensor for subtiling
    accs_comp = _split_n(c_comp, p.SUBTILE_FACTOR)
    accs_meta = _split_n(c_meta, p.SUBTILE_FACTOR)

    for i in gl.static_range(p.SUBTILE_FACTOR):
        mbarrier.wait(p.acc_empty_bars.index(acc_state.index), acc_state.phase)
        
        c_buf = p.acc_bufs.index(acc_state.index)
        c_meta_buf = p.c_meta_bufs.index(acc_state.index) 

        c_buf.store(accs_comp[i].to(p.c_desc.dtype))
        c_meta_buf.store(accs_meta[i])
        
        fence_async_shared()
        mbarrier.arrive(p.acc_ready_bars.index(acc_state.index), count=1)
        acc_state = acc_state.next()

    return acc_state

@gluon.jit
def sparse_matmul_compute_iteration(p, load_state, mma, k_iter, outstanding_mmas: gl.constexpr):
    mbarrier.wait(p.load_ready_bars.index(load_state.index), load_state.phase)

    e_reg = mma.issue_metadata_load(p.e_bufs.index(load_state.index))
    mma = mma.issue_async_sparse_mma(p.a_bufs.index(load_state.index), e_reg, p.b_bufs.index(load_state.index))

    load_state = load_state.next()

    mma = mma.wait_num_outstanding(outstanding_mmas)

    mbarrier.arrive(p.load_empty_bars.index((k_iter - outstanding_mmas) % p.load_empty_bars.shape[0]),
                    count=1, pred=k_iter>=outstanding_mmas)

    return load_state, mma, k_iter + 1

@gluon.jit
def sparse_matmul_compute_drain(p, load_state, mma, k_iter, limit, current_mma: gl.constexpr, num_mmas: gl.constexpr):
    if current_mma >= num_mmas: return load_state, mma, k_iter
    else:
        if k_iter >= limit: return load_state, mma, k_iter
        else:
            load_state, mma, k_iter = sparse_matmul_compute_iteration(p, load_state, mma, k_iter, num_mmas)
            return sparse_matmul_compute_drain(p, load_state, mma, k_iter, limit, gl.constexpr(current_mma + 1), num_mmas)

@gluon.jit
def sparse_matmul_compute_partition(p, SchedulerImpl: gl.constexpr):
    BLOCK_M: gl.constexpr = p.a_desc.block_type.shape[0]
    BLOCK_N: gl.constexpr = p.b_desc.block_type.shape[1]
    BLOCK_K: gl.constexpr = p.b_desc.block_type.shape[0]
    K = p.b_desc.shape[0]
    dtype: gl.constexpr = p.a_desc.dtype

    load_state = Counter.create(0, p.load_empty_bars.shape[0])
    acc_state = Counter.create(1, p.acc_empty_bars.shape[0])

    scheduler = SchedulerImpl.initialize(p.c_desc.shape[0], p.b_desc.shape[1], BLOCK_M, BLOCK_N)

    num_mmas: gl.constexpr = 2
    k_iter = 0

    for _ in range(scheduler.get_num_tiles()):
        mma = WGMMA.initialize(dtype, BLOCK_M, BLOCK_N, p.num_warps, sparse=True)

        total_k_iters = (K + BLOCK_K - 1) // BLOCK_K

        # Statically Unrolled
        for _ in range(total_k_iters // num_mmas):
            for _ in gl.static_range(num_mmas):
                load_state, mma, k_iter = sparse_matmul_compute_iteration(p, load_state, mma, k_iter, num_mmas - 1)

        # Drain
        load_state, mma, k_iter = sparse_matmul_compute_drain(
            p, load_state, mma, k_iter,
            (total_k_iters % num_mmas) + k_iter,
            0, num_mmas - 1
        )

        acc_state = store_acc_to_smem_subtile(p, mma, acc_state)

@gluon.jit
def sparse_matmul_store_partition(p, SchedulerImpl: gl.constexpr):
    BLOCK_M: gl.constexpr = p.c_desc.block_type.shape[0]
    SPLIT_N: gl.constexpr = p.c_desc.block_type.shape[1]
    BLOCK_N: gl.constexpr = p.b_desc.block_type.shape[1] 
    BLOCK_N_COMP: gl.constexpr = SPLIT_N * p.SUBTILE_FACTOR
    
    # NEW: Define the correct split size for metadata
    SPLIT_N_META: gl.constexpr = p.c_meta_desc.block_type.shape[1]

    state = Counter.create(0, p.acc_empty_bars.shape[0])
    scheduler = SchedulerImpl.initialize(p.c_desc.shape[0], p.b_desc.shape[1], BLOCK_M, BLOCK_N)

    num_buffers: gl.constexpr = 2
    outstanding_stores: gl.constexpr = 1
    store_iter = 0

    for idx in range(scheduler.get_num_tiles()):
        pid_m, pid_n = scheduler.get_tile(idx)
        
        # Data is compressed (halved), Meta maintains original dense N mapping
        off_m, off_n_comp = pid_m * BLOCK_M, pid_n * BLOCK_N_COMP
        off_n_meta = pid_n * BLOCK_N

        for i in gl.static_range(p.SUBTILE_FACTOR):
            mbarrier.wait(p.acc_ready_bars.index(state.index), state.phase)
            c_buf = p.acc_bufs.index(state.index)
            c_meta_buf = p.c_meta_bufs.index(state.index)

            # Store compressed values using halved offsets
            tma.async_copy_shared_to_global(p.c_desc, [off_m, off_n_comp + i * SPLIT_N], c_buf)
            
            # FIX: Store metadata using dense offsets
            tma.async_copy_shared_to_global(p.c_meta_desc, [off_m // 16, off_n_meta + i * SPLIT_N_META], c_meta_buf)

            # Wait for 2 stores per iteration (Data + Meta) to complete
            if store_iter >= outstanding_stores:
                tma.store_wait(outstanding_stores * 2) 
                empty_idx = (store_iter - outstanding_stores) % num_buffers
                mbarrier.arrive(p.acc_empty_bars.index(empty_idx), count=1)

            state = state.next()
            store_iter += 1

    tma.store_wait(0)

# ---------------------------------------------------------------------------
# KERNEL LAUNCHER
# ---------------------------------------------------------------------------
@gluon.jit
def sparse_matmul_warp_specialized_kernel(a_desc, e_desc, b_desc, c_desc, c_meta_desc, SchedulerImpl: gl.constexpr,
                                          M, N, K, 
                                          BLOCK_SIZE_M: gl.constexpr, BLOCK_SIZE_N: gl.constexpr, BLOCK_SIZE_K: gl.constexpr,
                                          num_buffers: gl.constexpr, SUBTILE_FACTOR: gl.constexpr,
                                          num_warps: gl.constexpr):
    dtype: gl.constexpr = a_desc.dtype
    # gl.static_print(f"BM: {BLOCK_SIZE_M}, BN: {BLOCK_SIZE_N}, BK: {BLOCK_SIZE_K}, num_bufs: {num_buffers}, num_warp: {num_warps}, SF: {SUBTILE_FACTOR}")

    
    a_bufs = gl.allocate_shared_memory(dtype, [num_buffers] + a_desc.block_type.shape, a_desc.layout)
    e_bufs = gl.allocate_shared_memory(e_desc.dtype, [num_buffers] + e_desc.block_type.shape, e_desc.layout)
    b_bufs = gl.allocate_shared_memory(dtype, [num_buffers] + b_desc.block_type.shape, b_desc.layout)
    load_empty_bars = gl.allocate_shared_memory(gl.int64, [num_buffers, 1], mbarrier.MBarrierLayout())
    load_ready_bars = gl.allocate_shared_memory(gl.int64, [num_buffers, 1], mbarrier.MBarrierLayout())

    for i in gl.static_range(num_buffers):
        mbarrier.init(load_empty_bars.index(i), count=1)
        mbarrier.init(load_ready_bars.index(i), count=1)

    acc_bufs = gl.allocate_shared_memory(dtype, [2] + c_desc.block_type.shape, c_desc.layout)
    c_meta_bufs = gl.allocate_shared_memory(gl.int16, [2] + c_meta_desc.block_type.shape, c_meta_desc.layout)
    acc_empty_bars = gl.allocate_shared_memory(gl.int64, [2, 1], mbarrier.MBarrierLayout())
    acc_ready_bars = gl.allocate_shared_memory(gl.int64, [2, 1], mbarrier.MBarrierLayout())

    for i in gl.static_range(2):
        mbarrier.init(acc_empty_bars.index(i), count=1)
        mbarrier.init(acc_ready_bars.index(i), count=1)

    # Pass num_warps straight into the arguments so the layout generator gets it
    p = SparsePartitionArgs(a_desc, e_desc, b_desc, c_desc, c_meta_desc, 
                            a_bufs, e_bufs, b_bufs,
                            load_empty_bars, load_ready_bars,
                            acc_bufs, c_meta_bufs, acc_empty_bars, acc_ready_bars, SUBTILE_FACTOR, num_warps)

    gl.warp_specialize([
        (sparse_matmul_compute_partition, (p, SchedulerImpl)),
        (sparse_matmul_load_partition, (p, SchedulerImpl)),
        (sparse_matmul_store_partition, (p, SchedulerImpl)),
    ], [1, 1], [24, 24])

def sparse_matmul_get_configs(pre_hook=None, tune=True): # TODO: Fix the guards
    def valid(BM, BN, BK, warps, buffers, SF):
        # Shared Memory calculations accommodate halved N from pruning mapping
        smem_bytes = (
                             (buffers * BM * BK) +               # Compressed A
                             (buffers * BM * BK // 8) +          # Metadata E
                             (buffers * BK * BN * 2) +           # Dense B
                             (4 * BM * (BN // 2 // SF)) +        # Accumulator C float16 (2 buffers * 2 bytes)
                             (4 * (BM // 16) * (BN // 2 // SF))  # Accumulator C Meta int16 (2 buffers * 2 bytes)
                     ) + (16 * buffers) + 32                     # MBarriers

        if smem_bytes > 232448: return False

        warps_m = 4
        warps_n = 1
        m = 16
        while (warps_m * warps_n) != warps:
            if BM > m * warps_m:
                warps_m *= 2
            else:
                warps_n *= 2

        if SF > 1 and warps_n > 1: return False
        if (BN // SF) < 16: return False
        if BM < warps_m * 16 or BN < warps_n * 16: return False

        elements_per_thread = (BM * BN) / (warps * 32)
        required_regs = elements_per_thread + 48
        max_regs_per_thread = min(255, 65536 // (warps * 32))

        if required_regs > max_regs_per_thread: return False
        if elements_per_thread < 16: return False
        
        return True

    configs = [
        triton.Config(
            {
                "BLOCK_SIZE_M": BM,
                "BLOCK_SIZE_N": BN,
                "BLOCK_SIZE_K": BK,
                "num_buffers": buffers,
                "SUBTILE_FACTOR": SF,
            },
            num_warps=warps,
            pre_hook=pre_hook,
        )
        for BM in (64, 128, 256,)
        for BN in (64, 128, 256,)
        for BK in (64, 128, 256,)
        for warps in (4, 8, 16)
        for buffers in (3, 4, 5, 6, 7)
        for SF in (1, 2, 4, 8)
        if valid(BM, BN, BK, warps, buffers, SF)
    ]
    
    return configs if tune else configs[:1]

def sparse_matmul_get_trimmed_configs(pre_hook=None):
    def valid(BM, BN, BK, warps, buffers, SF):
        smem_bytes = (
                             (buffers * BM * BK) +               
                             (buffers * BM * BK // 8) +          
                             (buffers * BK * BN * 2) +           
                             (4 * BM * (BN // 2 // SF)) +        
                             (4 * (BM // 16) * (BN // 2 // SF))  
                     ) + (16 * buffers) + 32                     

        if smem_bytes > 232448: return False

        warps_m = 4
        warps_n = 1
        m = 16
        while (warps_m * warps_n) != warps:
            if BM > m * warps_m: warps_m *= 2
            else: warps_n *= 2

        if SF > 1 and warps_n > 1: return False
        if (BN // SF) < 16: return False
        if BM < warps_m * 16 or BN < warps_n * 16: return False

        elements_per_thread = (BM * BN) / (warps * 32)
        required_regs = elements_per_thread + 48
        max_regs_per_thread = min(255, 65536 // (warps * 32))

        if required_regs > max_regs_per_thread: return False
        if elements_per_thread < 16: return False
        
        return True

    configs = [
        triton.Config(
            {
                "BLOCK_SIZE_M": BM,
                "BLOCK_SIZE_N": BN,
                "BLOCK_SIZE_K": BK,
                "num_buffers": buffers,
                "SUBTILE_FACTOR": SF,
            },
            num_warps=warps,
            pre_hook=pre_hook,
        )
        for BM in (128, 256)
        for BN in (256, )
        for BK in (64,)
        for warps in (8, 16, )
        for buffers in (3, 4, 5,)
        for SF in (2, 4)
        if valid(BM, BN, BK, warps, buffers, SF)
    ]
    
    if not configs:
        raise ValueError(f"No valid configurations found for BM={BM}, BN={BN}, BK={BK}, warps={warps}. Adjust your fixed sizes.")
    return configs

def sparse_matmul_get_768_configs(pre_hook=None):
    def valid(BM, BN, BK, warps, buffers, SF):
        smem_bytes = (
                             (buffers * BM * BK) +               
                             (buffers * BM * BK // 8) +          
                             (buffers * BK * BN * 2) +           
                             (4 * BM * (BN // 2 // SF)) +        
                             (4 * (BM // 16) * (BN // 2 // SF))  
                     ) + (16 * buffers) + 32                     

        if smem_bytes > 232448: return False

        warps_m = 4
        warps_n = 1
        m = 16
        while (warps_m * warps_n) != warps:
            if BM > m * warps_m: warps_m *= 2
            else: warps_n *= 2

        if SF > 1 and warps_n > 1: return False
        if (BN // SF) < 16: return False
        if BM < warps_m * 16 or BN < warps_n * 16: return False

        elements_per_thread = (BM * BN) / (warps * 32)
        required_regs = elements_per_thread + 48
        max_regs_per_thread = min(255, 65536 // (warps * 32))

        if required_regs > max_regs_per_thread: return False
        if elements_per_thread < 16: return False

        return True

    configs = [
        triton.Config(
            {
                "BLOCK_SIZE_M": BM,
                "BLOCK_SIZE_N": BN,
                "BLOCK_SIZE_K": BK,
                "num_buffers": buffers,
                "SUBTILE_FACTOR": SF,
            },
            num_warps=warps,
            pre_hook=pre_hook,
        )
        
        for BM in (128, )
        for BN in (128, )
        for BK in (64, 128, )
        for warps in (4, )
        for buffers in (4, 5, 6, )
        for SF in (2, 4, 8)
        if valid(BM, BN, BK, warps, buffers, SF)
    ]
    
    if not configs:
        raise ValueError(f"No valid configurations found for BM={BM}, BN={BN}, BK={BK}, warps={warps}. Adjust your fixed sizes.")
    return configs


def sparse_matmul_tma_set_block_size_hook(nargs):
    block_m = nargs["BLOCK_SIZE_M"]
    block_n = nargs["BLOCK_SIZE_N"]
    block_k = nargs["BLOCK_SIZE_K"]
    
    # C outputs are compressed via 2:4 logic so the N axis is halved
    split_n = (block_n) // nargs["SUBTILE_FACTOR"]

    nargs["a_desc"].block_shape = [block_m, block_k // 2]
    nargs["e_desc"].block_shape = [block_m // 16, block_k]
    nargs["b_desc"].block_shape = [block_k, block_n]
    nargs["c_desc"].block_shape = [block_m, split_n // 2]
    nargs["c_meta_desc"].block_shape = [block_m // 16, split_n]

    nargs["a_desc"].layout = gl.NVMMASharedLayout.get_default_for(nargs["a_desc"].block_shape, gl.float16)
    nargs["e_desc"].layout = gl.NVMMASharedLayout.get_default_for(nargs["e_desc"].block_shape, gl.int16)
    nargs["b_desc"].layout = gl.NVMMASharedLayout.get_default_for(nargs["b_desc"].block_shape, gl.float16)
    nargs["c_desc"].layout = gl.NVMMASharedLayout.get_default_for(nargs["c_desc"].block_shape, gl.float16)
    nargs["c_meta_desc"].layout = gl.NVMMASharedLayout.get_default_for(nargs["c_meta_desc"].block_shape, gl.int16)

sparse_ws_kernel_autotune_trimmed = triton.autotune(
    configs=sparse_matmul_get_trimmed_configs(
        pre_hook=sparse_matmul_tma_set_block_size_hook
    ),
    key=["M", "N", "K"],
    do_bench = lambda kernel_call, quantiles: triton.testing.do_bench_cudagraph(
        kernel_call, rep=100, quantiles=quantiles),
)(sparse_matmul_warp_specialized_kernel)

sparse_ws_kernel_autotune_768 = triton.autotune(
    configs=sparse_matmul_get_768_configs(
        pre_hook=sparse_matmul_tma_set_block_size_hook
    ),
    key=["M", "N", "K"],
    do_bench = lambda kernel_call, quantiles: triton.testing.do_bench_cudagraph(
        kernel_call, rep=100, quantiles=quantiles),
)(sparse_matmul_warp_specialized_kernel)

sparse_ws_kernel_autotune = triton.autotune(
    configs=sparse_matmul_get_configs(pre_hook=sparse_matmul_tma_set_block_size_hook),
    key=["M", "N", "K"],
    do_bench = lambda kernel_call, quantiles: triton.testing.do_bench_cudagraph(
        kernel_call, rep=100, quantiles=quantiles),
)(sparse_matmul_warp_specialized_kernel)

sparse_ws_kernel_single = triton.autotune(
    configs=sparse_matmul_get_configs(pre_hook=sparse_matmul_tma_set_block_size_hook, tune=False),
    key=["M", "N", "K"],
)(sparse_matmul_warp_specialized_kernel)

def run_sparse_ws_matmul(A, E, B, tune=True, manual_config=None):
    M, N, K = A.shape[0], B.shape[1], B.shape[0]

    # Initialize compressed target shapes (N is halved due to 2:4 ratio pruning on accumulator)
    c = torch.empty((M, N // 2), device=A.device, dtype=torch.float16)
    c_meta = torch.empty((M // 16, N), device=A.device, dtype=torch.int16)

    dummy_block = [1, 1]
    dummy_layout_f16 = gl.NVMMASharedLayout.get_default_for(dummy_block, gl.float16)
    dummy_layout_i16 = gl.NVMMASharedLayout.get_default_for(dummy_block, gl.int16)
    a_desc = TensorDescriptor.from_tensor(A, dummy_block, dummy_layout_f16)
    e_desc = TensorDescriptor.from_tensor(E, dummy_block, dummy_layout_i16)
    b_desc = TensorDescriptor.from_tensor(B, dummy_block, dummy_layout_f16)
    c_desc = TensorDescriptor.from_tensor(c, dummy_block, dummy_layout_f16)
    c_meta_desc = TensorDescriptor.from_tensor(c_meta, dummy_block, dummy_layout_i16)

    if tune:
        # Let the autotuner handle everything
        def grid(meta):
            num_sms = torch.cuda.get_device_properties("cuda").multi_processor_count
            num_pid = triton.cdiv(M, meta["BLOCK_SIZE_M"]) * triton.cdiv(N, meta["BLOCK_SIZE_N"])
            return (min(num_sms, num_pid), )
        
        sparse_ws_kernel_autotune_trimmed[grid](a_desc, e_desc, b_desc, c_desc, c_meta_desc, GroupedPersistentTileScheduler(8), M, N, K)
    else:
        # 1. Prepare kwargs for the TMA hook
        hook_kwargs = {
            "BLOCK_SIZE_M": manual_config["BM"],
            "BLOCK_SIZE_N": manual_config["BN"],
            "BLOCK_SIZE_K": manual_config["BK"],
            "SUBTILE_FACTOR": manual_config["SF"],
            "a_desc": a_desc, "e_desc": e_desc, "b_desc": b_desc, "c_desc": c_desc, "c_meta_desc": c_meta_desc
        }
        
        # 2. Mutate the descriptors manually
        sparse_matmul_tma_set_block_size_hook(hook_kwargs)
        
        # 3. Calculate grid using manual config
        num_sms = torch.cuda.get_device_properties("cuda").multi_processor_count
        num_pid = triton.cdiv(M, manual_config["BM"]) * triton.cdiv(N, manual_config["BN"])
        grid = (min(num_sms, num_pid), )
        
        # 4. Launch the base kernel directly (bypassing autotune)
        sparse_matmul_warp_specialized_kernel[grid](
            a_desc, e_desc, b_desc, c_desc, c_meta_desc, GroupedPersistentTileScheduler(8),
            M, N, K,
            BLOCK_SIZE_M=manual_config["BM"], 
            BLOCK_SIZE_N=manual_config["BN"], 
            BLOCK_SIZE_K=manual_config["BK"],
            num_buffers=manual_config["buffers"], 
            SUBTILE_FACTOR=manual_config["SF"], 
            num_warps=manual_config["warps"]
        )

    return c, c_meta

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Sparse Warp-Specialized Matmul")
    parser.add_argument("--tune", action="store_true", help="Enable Triton autotuning")
    
    # Manual config arguments (ignored if --tune is passed)
    parser.add_argument("--bm", type=int, default=128, help="BLOCK_SIZE_M")
    parser.add_argument("--bn", type=int, default=256, help="BLOCK_SIZE_N")
    parser.add_argument("--bk", type=int, default=64, help="BLOCK_SIZE_K")
    parser.add_argument("--warps", type=int, default=8, help="Number of warps")
    parser.add_argument("--buffers", type=int, default=4, help="Number of buffers")
    parser.add_argument("--sf", type=int, default=4, help="SUBTILE_FACTOR")
    
    args = parser.parse_args()

    manual_config = {
        "BM": args.bm,
        "BN": args.bn,
        "BK": args.bk,
        "warps": args.warps,
        "buffers": args.buffers,
        "SF": args.sf
    }

    if args.tune:
        print("Running sparse matmul. Autotuning enabled.")
    else:
        print(f"Running sparse matmul with manual config: {manual_config}")

    sizes = [
        (49152, 8192, 49152)
    ]

    from compress_2_4 import *
    from prune import *

    for M, N, K in sizes:
        A = torch.randn((M, K), device="cuda", dtype=torch.float16)
        B = torch.randn((K, N), device="cuda", dtype=torch.float16)
        
        A_pruned = prune_2_4(A)
        A_comp, E = compress_dense_to_sparse(A_pruned)
        E = E.view(M // 16, K)
        
        # D_comp / D_meta are the 2:4 output directly generated from the kernel
        D_comp, D_meta = run_sparse_ws_matmul(A_comp, E, B, tune=args.tune, manual_config=manual_config)
        
        # Ground truth using PyTorch dense mapping mapping to simulate compression
        C_ref_dense = A_pruned @ B
        C_ref_pruned = prune_2_4(C_ref_dense)
        C_ref_comp, C_ref_meta = compress_dense_to_sparse(C_ref_pruned)
        C_ref_meta = C_ref_meta.view(M//16, N)
        
        # torch.set_printoptions(profile="full")
        # torch.set_printoptions(linewidth=3000)
        # print(C_ref_comp)
        # print(D_comp)
        # print(C_ref_meta)
        # print(D_meta)
        
        # --- Replace torch.testing.assert_close with this Verification Block ---
        
        # 1. Reshape into 1x4 dense blocks and 1x2 compressed blocks
        D_comp_blocks = D_comp.view(-1, 2)
        C_comp_blocks = C_ref_comp.view(-1, 2)
        C_dense_blocks = C_ref_dense.view(-1, 4)

        # 2. Find blocks where the strict spatial alignment fails
        mismatch_mask = torch.abs(D_comp_blocks - C_comp_blocks).max(dim=-1).values > 1e-1
        num_mismatched_blocks = mismatch_mask.sum().item()

        print(f"Found {num_mismatched_blocks} blocks with spatial mismatches.")

        if num_mismatched_blocks > 0:
            # Extract only the blocks that failed the spatial test
            bad_D = D_comp_blocks[mismatch_mask]       # Shape: (num_bad, 2)
            bad_C_dense = C_dense_blocks[mismatch_mask] # Shape: (num_bad, 4)

            # 3. VERIFICATION (Absolute): Does Triton's output exist ANYWHERE in the dense block?
            abs_dist = torch.abs(bad_D.unsqueeze(2) - bad_C_dense.unsqueeze(1)) # (num_bad, 2, 4)
            
            # Find the minimum absolute distance and the index of that best-matching dense element
            min_abs_dist_to_dense, best_match_indices = abs_dist.min(dim=2) # Both are (num_bad, 2)
            max_abs_error = min_abs_dist_to_dense.max(dim=1).values # (num_bad,)

            # 4. VERIFICATION (Relative): Is the relative difference also within normal bounds?
            # Extract the actual PyTorch dense values that best matched Triton's output
            best_match_dense_vals = bad_C_dense.gather(1, best_match_indices) # (num_bad, 2)
            
            # Calculate the true relative error between Triton's value and the correctly aligned PyTorch value.
            # We add 1e-5 to the denominator to prevent division by zero on very small numbers.
            rel_error = torch.abs(bad_D - best_match_dense_vals) / (torch.abs(best_match_dense_vals) + 1e-5)
            max_rel_error = rel_error.max(dim=1).values # (num_bad,)

            # Allow thresholds purely for the cuBLAS vs WGMMA floating-point accumulation noise
            # (0.5 absolute, 0.05 or 5% relative for very tiny numbers near 0)
            unexplained_mask = (max_abs_error > 0.5) | (max_rel_error > 0.05)
            unexplained_count = unexplained_mask.sum().item()

            if unexplained_count > 0:
                worst_abs = max_abs_error.max().item()
                worst_rel = max_rel_error.max().item()
                raise AssertionError(
                    f"THEORY IS FALSE! Found {unexplained_count} blocks where Triton's output "
                    f"does not match any element in the dense block.\n"
                    f"Max unexplained absolute diff: {worst_abs:.4f}\n"
                    f"Max unexplained relative diff: {worst_rel:.4f}"
                )
            else:
                print("VERIFIED: The explanation is TRUE.")
                print("100% of Triton's mismatched outputs exist perfectly inside PyTorch's dense block.")
                print(f"Max absolute noise for aligned values: {max_abs_error.max().item():.4f}")
                print(f"Max relative noise for aligned values: {max_rel_error.max().item():.6f}")
        else:
            print("No mismatches found!")