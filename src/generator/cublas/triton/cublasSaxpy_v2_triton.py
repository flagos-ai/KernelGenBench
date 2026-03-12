import torch
import triton
import triton.language as tl


@triton.jit
def saxpy(x_ptr, y_ptr, alpha, n, incx, incy, start_x, start_y, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n

    x_idx = start_x + offs * incx
    y_idx = start_y + offs * incy

    x_val = tl.load(x_ptr + x_idx, mask=mask, other=0.0)
    y_val = tl.load(y_ptr + y_idx, mask=mask, other=0.0)

    out = alpha * x_val + y_val
    tl.store(y_ptr + y_idx, out, mask=mask)


def cublasSaxpy_v2(n, alpha, x, incx, y, incy):
    # Basic validations to match cublasSaxpy_v2 expectations
    assert x.is_cuda and y.is_cuda, "x and y must be CUDA tensors"
    assert x.dtype == torch.float32 and y.dtype == torch.float32, "x and y must be float32"
    assert x.dim() == 1 and y.dim() == 1, "x and y must be 1D tensors"
    if n <= 0:
        return y

    # Handle negative increments by adjusting starting offsets
    start_x = 0 if incx >= 0 else (-incx) * (n - 1)
    start_y = 0 if incy >= 0 else (-incy) * (n - 1)

    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n, BLOCK_SIZE),)

    saxpy[grid](
        x, y,
        float(alpha), n, int(incx), int(incy), int(start_x), int(start_y),
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=4,
        num_stages=2
    )
    return y
