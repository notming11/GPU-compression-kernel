import triton
import triton.language as tl
import torch

@triton.jit
def test_shfl(out_ptr):
    pid = tl.program_id(0)
    tid = tl.arange(0, 32)
    
    val = tid
    # Try inline PTX for shfl.sync.bfly
    ptx = """
    {
        .reg .u32 %lane;
        mov.u32 %lane, %laneid;
        
        .reg .b32 %val;
        mov.b32 %val, $1;
        
        shfl.sync.bfly.b32 %val, %val, 1, 0x1f, 0xffffffff;
        
        mov.b32 $0, %val;
    }
    """
    res = tl.inline_asm_elementwise(
        ptx,
        "=r,r",
        [val],
        dtype=tl.int32,
        is_pure=True,
        pack=1
    )
    
    tl.store(out_ptr + tid, res)

def main():
    out = torch.empty(32, dtype=torch.int32, device='cuda')
    test_shfl[(1,)](out)
    print("Input: ", torch.arange(32))
    print("Output:", out)
    
if __name__ == '__main__':
    main()
