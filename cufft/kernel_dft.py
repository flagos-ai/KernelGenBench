import triton
import triton.language as tl
import torch


@triton.jit
def dft_1d_kernel(
    input_ptr,
    output_ptr,
    n,
    direction,
    BLOCK: tl.constexpr,
):
    """Minimal, correctness-first O(N^2) DFT kernel.

    input_ptr / output_ptr: real-view of complex tensor, shape [n, 2].
    direction: +1 for forward, -1 for inverse (sign of exponent).
    """
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n

    TWO_PI = 6.283185307179586
    sign = -1.0 if direction == 1 else 1.0

    k = offsets
    is_valid = mask & (k < n)

    acc_real = tl.zeros([BLOCK], dtype=tl.float32)
    acc_imag = tl.zeros([BLOCK], dtype=tl.float32)

    for n_idx in range(0, n):
        angle = sign * TWO_PI * k * n_idx / n
        twiddle_real = tl.cos(angle)
        twiddle_imag = tl.sin(angle)

        in_real = tl.load(input_ptr + n_idx * 2 + 0)
        in_imag = tl.load(input_ptr + n_idx * 2 + 1)

        acc_real += in_real * twiddle_real - in_imag * twiddle_imag
        acc_imag += in_real * twiddle_imag + in_imag * twiddle_real

    tl.store(output_ptr + k * 2 + 0, acc_real, mask=is_valid)
    tl.store(output_ptr + k * 2 + 1, acc_imag, mask=is_valid)


def fft_1d_c2c(input: torch.Tensor, nx: int, direction: int):
    """Very small, correctness-first 1D complex-to-complex FFT using Triton.

    Currently implemented as an O(N^2) DFT for arbitrary nx, to make
    debugging and correctness checking against torch.fft.fft easy.
    """
    assert input.is_cuda, "Input tensor must be on CUDA"
    assert input.dtype in (torch.complex64, torch.complex128), "Input must be complex64/complex128"
    assert input.shape == (nx,), f"Input shape {input.shape} doesn't match (nx={nx})"

    input_real = torch.view_as_real(input).contiguous()
    output_real = torch.empty_like(input_real)

    BLOCK = 256
    grid = lambda meta: (triton.cdiv(nx, meta['BLOCK']),)

    dft_1d_kernel[grid](
        input_real,
        output_real,
        nx,
        direction,
        BLOCK=BLOCK,
    )

    output = torch.view_as_complex(output_real)

    if direction == -1:
        output = output / nx

    return output, 0
