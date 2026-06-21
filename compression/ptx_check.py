import re
import ast
import torch

def parse_layout(lines):
    layout = []
    for line in lines:
        line = line.strip()
        if not line: continue
        # Find all T<thread>:<reg> patterns
        matches = re.findall(r'T\d+:\d+', line)
        if matches:
            layout.append(matches)
    return layout

def parse_tensor(lines):
    # Join all lines, then use regex to find all numbers
    text = " ".join(lines)
    # Extract the part inside tensor([...])
    match = re.search(r'tensor\(\[\[(.*?)\]\]', text)
    if not match:
        return []
    
    # We will just find all floats
    # Because it's a 2D array, we can parse each row
    rows_str = re.findall(r'\[(.*?)\]', text)
    tensor = []
    for row_str in rows_str:
        # Extract numbers from row_str
        nums = re.findall(r'[-+]?\d*\.\d+(?:e[-+]?\d+)?', row_str)
        if nums:
            tensor.append([float(n) for n in nums])
    return tensor

def main():
    with open('/home/notming/links/scratch/compression/ptx_correctness.txt', 'r') as f:
        content = f.read()
    
    sections = content.split('\n\n')
    
    ptx_layout_lines = []
    kernel_out_lines = []
    wgmma_layout_lines = []
    ref_out_lines = []
    
    current_section = None
    for line in content.split('\n'):
        if line.startswith('PTX:'):
            current_section = 'PTX'
            continue
        elif line.startswith('Kernel Output Sample:'):
            current_section = 'KernelOut'
            continue
        elif line.startswith('WGMMA:'):
            current_section = 'WGMMA'
            continue
        elif line.startswith('PyTorch Reference Sample:'):
            current_section = 'RefOut'
            continue
        elif line.startswith('❌ FAILED:'):
            break
            
        if current_section == 'PTX':
            ptx_layout_lines.append(line)
        elif current_section == 'KernelOut':
            kernel_out_lines.append(line)
        elif current_section == 'WGMMA':
            wgmma_layout_lines.append(line)
        elif current_section == 'RefOut':
            ref_out_lines.append(line)

    ptx_layout = parse_layout(ptx_layout_lines)
    kernel_out = parse_tensor(kernel_out_lines)
    wgmma_layout = parse_layout(wgmma_layout_lines)
    ref_out = parse_tensor(ref_out_lines)
    
    print(f"Parsed PTX layout: {len(ptx_layout)}x{len(ptx_layout[0]) if ptx_layout else 0}")
    print(f"Parsed Kernel Output: {len(kernel_out)}x{len(kernel_out[0]) if kernel_out else 0}")
    print(f"Parsed WGMMA layout: {len(wgmma_layout)}x{len(wgmma_layout[0]) if wgmma_layout else 0}")
    print(f"Parsed Ref Output: {len(ref_out)}x{len(ref_out[0]) if ref_out else 0}")

    # Create mapping from Thread:Reg to value
    # Kernel
    kernel_dict = {}
    for i, row in enumerate(ptx_layout):
        for j, tr in enumerate(row):
            kernel_dict[tr] = kernel_out[i][j]
            
    # Ref
    ref_dict = {}
    for i, row in enumerate(wgmma_layout):
        for j, tr in enumerate(row):
            ref_dict[tr] = ref_out[i][j]
            
    # Compare
    mismatch_count = 0
    max_diff = 0
    
    for tr in kernel_dict:
        if tr not in ref_dict:
            print(f"Missing {tr} in reference")
            continue
        v_kernel = kernel_dict[tr]
        v_ref = ref_dict[tr]
        
        diff = abs(v_kernel - v_ref)
        if diff > 1e-3:
            mismatch_count += 1
            max_diff = max(max_diff, diff)
        
        if "T0" in tr:
            if diff > 1e-3:
                print(f"Mismatch at {tr}: kernel={v_kernel}, ref={v_ref}, diff={diff}")
            # else:
            #     print(f"Match at {tr}: kernel={v_kernel}, ref={v_ref}, diff={diff}")


    total = len(kernel_dict)
    if mismatch_count == 0:
        print("✅ SUCCESS: Kernel has the right data in the right registers and threads!")
    else:
        print(f"❌ FAILED: {mismatch_count} / {total} elements mismatched.")
        print(f"Max difference: {max_diff}")

if __name__ == '__main__':
    main()
