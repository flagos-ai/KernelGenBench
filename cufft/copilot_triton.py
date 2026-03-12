import torch
import triton
import triton.language as tl
import math

@triton.jit
def fft_kernel(
    real_ptr,
    imag_ptr,
    out_real_ptr,
    out_imag_ptr,
    n,
    stride,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)

    # Each program handles one output element
    k = pid
    if k >= n:
        return

    # Initialize accumulators
    sum_real = 0.0
    sum_imag = 0.0

    # Compute DFT for element k
    for j in range(n):
        # Load input values
        x_real = tl.load(real_ptr + j * stride)
        x_imag = tl.load(imag_ptr + j * stride)

        # Compute twiddle factor: e^(-2πi*jk/n) = cos(-2π*jk/n) + i*sin(-2π*jk/n)
        angle = -2.0 * 3.141592653589793 * j * k / n
        cos_val = tl.cos(angle)
        sin_val = tl.sin(angle)

        # Complex multiplication: (x_real + i*x_imag) * (cos_val + i*sin_val)
        sum_real += x_real * cos_val - x_imag * sin_val
        sum_imag += x_real * sin_val + x_imag * cos_val

    # Store output
    tl.store(out_real_ptr + k * stride, sum_real)
    tl.store(out_imag_ptr + k * stride, sum_imag)

def fft(input):
    # Handle input shape and dtype
    original_shape = input.shape
    input_flat = input.reshape(-1, input.shape[-1])
    batch_size = input_flat.shape[0]
    n = input_flat.shape[-1]

    # Convert to complex if needed
    if not input_flat.is_complex():
        input_flat = input_flat.to(torch.complex64)

    # Prepare output tensor
    output = torch.empty_like(input_flat)

    # Process each batch
    for b in range(batch_size):
        batch_input = input_flat[b]

        # Get real and imaginary parts
        real_input = batch_input.real.contiguous()
        imag_input = batch_input.imag.contiguous()
        real_output = torch.empty_like(real_input)
        imag_output = torch.empty_like(imag_input)

        # Launch kernel
        grid = (n,)
        fft_kernel[grid](
            real_input,
            imag_input,
            real_output,
            imag_output,
            n,
            1,
            BLOCK_SIZE=1024
        )

        # Combine real and imaginary parts
        output[b] = torch.complex(real_output, imag_output)

    # Reshape output to original shape
    output = output.reshape(original_shape)
    return output