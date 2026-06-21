import triton
import triton.language as tl
import gluon.language as gl

@gl.jit
def test_kernel():
    a = gl.zeros((64, 16), dtype=gl.int32)
    # Try reshaping and transposing
    a_trans = a.reshape(4, 2, 8, 16).trans(0, 2, 3, 1)
    a0, a1 = a_trans.split()
    gl.static_print(a0.shape)

