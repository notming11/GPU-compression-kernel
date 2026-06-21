import re
import ast

def parse_tensor(lines):
    text = " ".join(lines)
    match = re.search(r'tensor\(\[\[(.*?)\]\]', text)
    if not match: return []
    rows_str = re.findall(r'\[(.*?)\]', text)
    tensor = []
    for row_str in rows_str:
        nums = re.findall(r'[-+]?\d*\.\d+(?:e[-+]?\d+)?', row_str)
        if nums:
            tensor.append([float(n) for n in nums])
    return tensor

with open('ptx_correctness.txt', 'r') as f:
    content = f.read()

# print whether T0:4 is mismatched in the actual strings
kernel_out = []
ref_out = []
current = None
for line in content.split('\n'):
    if line.startswith('Kernel Output'): current = 'k'
    elif line.startswith('PyTorch Ref'): current = 'r'
    elif line.startswith('WGMMA'): current = 'w'
    elif current == 'k' and '[' in line: kernel_out.append(line)
    elif current == 'r' and '[' in line: ref_out.append(line)

print("Kernel parsed:", len(parse_tensor(kernel_out)))
print("Ref parsed:", len(parse_tensor(ref_out)))

