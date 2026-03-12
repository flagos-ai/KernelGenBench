from sandbox.register import register
from flagbench.dataset import Autograd
import torch
import triton
import triton.language as tl

@triton.jit
def dgemv_kernel(
    A_ptr, x_ptr, y_ptr,
    m, n,
    stride_am, stride_an,
    stride_x, stride_y,
    alpha, beta,
    BLOCK_SIZE_M: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    
    # Create block pointers
    A_block_ptr = tl.make_block_ptr(
        base=A_ptr,
        shape=(m, n),
        strides=(stride_am, stride_an),
        offsets=(pid * BLOCK_SIZE_M, 0),
        block_shape=(BLOCK_SIZE_M, n),
        order=(1, 0)
    )
    
    y_block_ptr = tl.make_block_ptr(
        base=y_ptr,
        shape=(m,),
        strides=(stride_y,),
        offsets=(pid * BLOCK_SIZE_M,),
        block_shape=(BLOCK_SIZE_M,),
        order=(0,)
    )
    
    # Load A block and x vector
    A_block = tl.load(A_block_ptr, boundary_check=(0, 1))
    x = tl.load(x_ptr + tl.arange(0, n) * stride_x)
    
    # Compute dot product
    acc = tl.sum(A_block * x, axis=1) * alpha
    
    if beta != 0.0:
        y_block = tl.load(y_block_ptr, boundary_check=(0,))
        acc += y_block * beta
    
    # Store result
    tl.store(y_block_ptr, acc.to(y_ptr.dtype.element_ty), boundary_check=(0,))

@register("CUDA", "dgemv", has_backward=Autograd.disable, namespace="triton")
def dgemv(
    handle,  # ignored
    trans,
    m,
    n,
    alpha,
    A: torch.Tensor,
    lda,
    x: torch.Tensor,
    incx,
    beta,
    y: torch.Tensor,
    incy,
):
    # Validate transpose flag
    if trans != 0:  # 0 is CUBLAS_OP_N
        raise NotImplementedError("Transpose operation not implemented")
    
    # Validate strides
    if lda != m or incx != 1 or incy != 1:
        raise NotImplementedError("Non-unit strides not implemented")
    
    BLOCK_SIZE_M = 64  # Tuned block size
    
    grid = lambda META: (triton.cdiv(m, META['BLOCK_SIZE_M']),)
    
    dgemv_kernel[grid](
        A, x, y,
        m, n,
        A.stride(0), A.stride(1),
        x.stride(0), y.stride(0),
        alpha, beta,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
    )
    
    return y