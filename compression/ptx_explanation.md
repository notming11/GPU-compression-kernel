# Triton PTX WGMMA Layout Conversion

This document breaks down the python packing/unpacking logic and the PTX script used to reorganize `float16` tensor elements to match the strict `DotOperandLayout` required by NVIDIA's WGMMA (Warpgroup Matrix-Multiply-Accumulate) instructions.

## 1. Packing Float16 into Int32 Registers

Triton's `inline_asm_elementwise` processes operations on hardware registers, commonly 32-bit (`.b32`) registers. To safely process `float16` elements without data loss, we pack adjacent pairs of `float16`s into single `int32` registers before passing them to PTX.

```python
# 1. Safely pack pairs of float16 into single int32 registers for PTX
a_int16 = a_compressed.to(gl.int16, bitcast=True)
a_pairs = a_int16.reshape(BLOCK_M, BLOCK_K // 4, 2)
p0, p1 = a_pairs.split()

# Masking prevents sign-extension when upcasting negative floats
p0_32 = p0.to(gl.int32) & 0xFFFF
p1_32 = p1.to(gl.int32) & 0xFFFF
a_int32 = p0_32 | (p1_32 << 16)
```

**How it works:**
- We first `bitcast` the `float16`s to `int16`. This treats the raw bits as integers, preventing Triton from doing floating-point arithmetic.
- We group the `BLOCK_K // 2` elements into pairs `(BLOCK_K // 4, 2)` and split them into the lower (`p0`) and upper (`p1`) elements.
- When casting `int16` to `int32`, negative values will "sign-extend" (filling the upper 16 bits with `1`s). To prevent this from destroying the packed value, we apply a bitmask `& 0xFFFF`. 
- Finally, we shift the upper element left by 16 bits (`p1_32 << 16`) and use a bitwise OR (`|`) to pack them into a single `int32` tensor, `a_int32`.

## 2. Executing PTX with `pack=8`

```python
# 2. EXECUTE PTX
(y_int32_reassembled,) = gl.inline_asm_elementwise(
    TRANSPOSE_PTX,
    "=r,=r,=r,=r,=r,=r,=r,=r,r,r,r,r,r,r,r,r",
    [a_int32],
    dtype=(gl.int32,),
    is_pure=True,
    pack=8,
)
```

The tensor `a_int32` possesses exactly 8 elements per thread. By passing `pack=8`, we instruct Triton to unpack all 8 elements simultaneously and feed them directly into the PTX input operands `$8` through `$15`. 

Because Triton iterates over the innermost dimension (columns) first, the 8 input elements for a given thread naturally correspond to:
- **Row 0's columns**: `$8, $9, $10, $11`
- **Row 8's columns**: `$12, $13, $14, $15`

### 2a. The Shuffle Algorithm (4x4 Transpose)

Inside the PTX script, a **butterfly transpose** is performed across a "quad" (a group of 4 threads). The script applies this exact logic independently for the Row 0 variables and the Row 8 variables.

Let's look at how the 4 registers for Row 0 are transposed across Threads 0, 1, 2, and 3:

#### Initial State
Each thread holds 4 column chunks.
| Thread | `$8` (Col A) | `$9` (Col B) | `$10` (Col C) | `$11` (Col D) |
|---|---|---|---|---|
| **T0** | T0:$8 | T0:$9 | T0:$10 | T0:$11 |
| **T1** | T1:$8 | T1:$9 | T1:$10 | T1:$11 |
| **T2** | T2:$8 | T2:$9 | T2:$10 | T2:$11 |
| **T3** | T3:$8 | T3:$9 | T3:$10 | T3:$11 |

#### Step 1: Exchange Distance 1 (XOR by 1)
Threads pair up and swap data using `shfl.sync.bfly` with mask `1` (T0 swaps with T1, T2 swaps with T3).
They swap specific columns based on their Lane ID (`%p1` predicate). `T0` gives up its `B` and `D` columns and takes `T1`'s `A` and `C` columns.

| Thread | `%r0_0` | `%r1_0` | `%r2_0` | `%r3_0` |
|---|---|---|---|---|
| **T0** | T0:$8 | **T1:$8** | T0:$10 | **T1:$10** |
| **T1** | **T0:$9** | T1:$9 | **T0:$11** | T1:$11 |
| **T2** | T2:$8 | **T3:$8** | T2:$10 | **T3:$10** |
| **T3** | **T2:$9** | T3:$9 | **T2:$11** | T3:$11 |

#### Step 2: Exchange Distance 2 (XOR by 2)
Threads pair up and swap data with distance 2 using mask `2` (T0 swaps with T2, T1 swaps with T3).

| Thread | `%r0_0` | `%r1_0` | `%r2_0` | `%r3_0` |
|---|---|---|---|---|
| **T0** | T0:$8 | T1:$8 | **T2:$8** | **T3:$8** |
| **T1** | T0:$9 | T1:$9 | **T2:$9** | **T3:$9** |
| **T2** | **T0:$10** | **T1:$10** | T2:$10 | T3:$10 |
| **T3** | **T0:$11** | **T1:$11** | T2:$11 | T3:$11 |

**Result:** Thread 0 now holds *all* the `A` columns from the entire quad! The 4x4 matrix of registers across the 4 threads has been successfully transposed.

### 2b. Reassembling the WGMMA Layout

WGMMA's `DotOperandLayout` dictates that for a single thread, the data for **Row 0** and **Row 8** of the *same* column block must be physically adjacent in the register file.

If we let Triton automatically pack the 8 registers blindly, it would place the 4 variables from Row 0 sequentially, followed by the 4 variables from Row 8. Instead, the final block of the PTX manually assigns the outputs (`$0` to `$7`) in an interleaved pattern:

```ptx
// Pair 1: Column block 0
mov.b32 $0, %r0_0;  // Row 0
mov.b32 $1, %r0_8;  // Row 8

// Pair 2: Column block 1
mov.b32 $2, %r1_0;  // Row 0
mov.b32 $3, %r1_8;  // Row 8

...
```

Triton's `pack=8` constraint automatically groups these 8 perfectly interleaved outputs back into the `y_int32_reassembled` tensor.

## 3. Unpacking the Registers

```python
# 5. Unpack back to int16 pairs and bitcast to float16
y_p0 = (y_int32_reassembled & 0xFFFF).to(gl.int16)
y_p1 = ((y_int32_reassembled >> 16) & 0xFFFF).to(gl.int16)
y_pairs = gl.join(y_p0, y_p1)
y_reshaped = y_pairs.reshape(BLOCK_M, BLOCK_K // 2)

a_compressed_swapped = y_reshaped.to(gl.float16, bitcast=True)
```

**How it works:**
- We extract the lower 16 bits of the output using `& 0xFFFF` and downcast it back to `int16`.
- We extract the upper 16 bits by shifting the tensor down (`>> 16`) and applying the same mask and downcast.
- We `join` the upper and lower halves back into a `(BLOCK_M, BLOCK_K // 4, 2)` tensor, and reshape it back into the flat `(BLOCK_M, BLOCK_K // 2)` tensor.
- Finally, a `bitcast` back to `float16` recovers the true floating-point values.

`a_compressed_swapped` is now exactly formatted to perfectly match `DotOperandLayout` without needing any memory-bound `convert_layout` calls!
