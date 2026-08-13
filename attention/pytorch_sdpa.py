import torch
import torch.nn.functional as F
import torch.cuda.profiler as profiler

def main():
    # 1. Target Shape Configuration
    NUM_HEADS = 16
    SEQ_LEN = 4096
    HEAD_DIM = 128
    BATCH = max(1, 16384//SEQ_LEN)

    print(f"[INFO] Allocating tensors for PyTorch SDPA (BATCH={BATCH}, NUM_HEADS={NUM_HEADS}, SEQ_LEN={SEQ_LEN}, HEAD_DIM={HEAD_DIM})...")
    Q = torch.randn((BATCH, NUM_HEADS, SEQ_LEN, HEAD_DIM), device="cuda", dtype=torch.float16)
    K = torch.randn((BATCH, NUM_HEADS, SEQ_LEN, HEAD_DIM), device="cuda", dtype=torch.float16)
    V = torch.randn((BATCH, NUM_HEADS, SEQ_LEN, HEAD_DIM), device="cuda", dtype=torch.float16)

    # 2. Warmup Iterations (forces cuDNN / FlashAttention backend initialization)
    print("[INFO] Warming up PyTorch SDPA backend...")
    for _ in range(5):
        _ = F.scaled_dot_product_attention(Q, K, V)
    torch.cuda.synchronize()

    # 3. Profiled Iteration
    print("[INFO] Executing profiled PyTorch SDPA iteration...")
    profiler.start()
    
    # NVTX range allows NCU to target ONLY this execution
    torch.cuda.nvtx.range_push("PyTorch_SDPA_4096_128")
    _ = F.scaled_dot_product_attention(Q, K, V)
    torch.cuda.nvtx.range_pop()
    
    torch.cuda.synchronize()
    profiler.stop()
    
    print("[INFO] Execution complete.")

if __name__ == "__main__":
    main()