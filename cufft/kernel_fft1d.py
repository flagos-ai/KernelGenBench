import triton
import triton.language as tl
import torch


@triton.jit
def fft1d_ct_n8_kernel(x_real, x_imag, y_real, y_imag, direction: tl.constexpr):
	"""Cooley–Tukey 1D FFT kernel for fixed N=8, single vector.

	This is a correctness-first, minimal version: one program instance
	computes full FFT of length 8 using registers only.
	"""
	# single program, 8 lanes
	offs = tl.arange(0, 8)

	# load input
	xr = tl.load(x_real + offs)
	xi = tl.load(x_imag + offs)

	# bit-reversal for N=8 (3 bits)
	# indices 0..7 -> reversed bits
	idx = offs
	rev = tl.zeros_like(idx)
	tmp = idx
	for _ in range(3):
		bit = tmp & 1
		rev = (rev << 1) | bit
		tmp = tmp >> 1

	# apply permutation: y[rev[i]] = x[i]
	yr = tl.zeros_like(xr)
	yi = tl.zeros_like(xi)
	for _ in range(8):
		idx_mask = offs == _
		val_r = tl.sum(xr * idx_mask, axis=0)
		val_i = tl.sum(xi * idx_mask, axis=0)
		rev_mask = rev == _
		yr = tl.where(rev_mask, val_r, yr)
		yi = tl.where(rev_mask, val_i, yi)

	xr = yr
	xi = yi

	TWO_PI = 6.283185307179586
	sign = -1.0 if direction == 1 else 1.0

	# stage 1: m=2, half=1
	for i in range(0, 8, 2):
		mask1 = offs == i
		mask2 = offs == i + 1
		xr1 = tl.sum(xr * mask1, axis=0)
		xi1 = tl.sum(xi * mask1, axis=0)
		xr2 = tl.sum(xr * mask2, axis=0)
		xi2 = tl.sum(xi * mask2, axis=0)

		tr = xr2
		ti = xi2

		yr1 = xr1 + tr
		yi1 = xi1 + ti
		yr2 = xr1 - tr
		yi2 = xi1 - ti

		xr = tl.where(mask1, yr1, xr)
		xi = tl.where(mask1, yi1, xi)
		xr = tl.where(mask2, yr2, xr)
		xi = tl.where(mask2, yi2, xi)

	# stage 2: m=4, half=2
	for group_base in (0, 4):
		for j in (0, 1):
			i1 = group_base + j
			i2 = i1 + 2
			mask1 = offs == i1
			mask2 = offs == i2
			xr1 = tl.sum(xr * mask1, axis=0)
			xi1 = tl.sum(xi * mask1, axis=0)
			xr2 = tl.sum(xr * mask2, axis=0)
			xi2 = tl.sum(xi * mask2, axis=0)

			angle = sign * TWO_PI * j / 4.0
			wr = tl.cos(angle)
			wi = tl.sin(angle)

			tr = xr2 * wr - xi2 * wi
			ti = xr2 * wi + xi2 * wr

			yr1 = xr1 + tr
			yi1 = xi1 + ti
			yr2 = xr1 - tr
			yi2 = xi1 - ti

			xr = tl.where(mask1, yr1, xr)
			xi = tl.where(mask1, yi1, xi)
			xr = tl.where(mask2, yr2, xr)
			xi = tl.where(mask2, yi2, xi)

	# stage 3: m=8, half=4
	for j in range(4):
		i1 = j
		i2 = j + 4
		mask1 = offs == i1
		mask2 = offs == i2
		xr1 = tl.sum(xr * mask1, axis=0)
		xi1 = tl.sum(xi * mask1, axis=0)
		xr2 = tl.sum(xr * mask2, axis=0)
		xi2 = tl.sum(xi * mask2, axis=0)

		angle = sign * TWO_PI * j / 8.0
		wr = tl.cos(angle)
		wi = tl.sin(angle)

		tr = xr2 * wr - xi2 * wi
		ti = xr2 * wi + xi2 * wr

		yr1 = xr1 + tr
		yi1 = xi1 + ti
		yr2 = xr1 - tr
		yi2 = xi1 - ti

		xr = tl.where(mask1, yr1, xr)
		xi = tl.where(mask1, yi1, xi)
		xr = tl.where(mask2, yr2, xr)
		xi = tl.where(mask2, yi2, xi)

	# write back
	tl.store(y_real + offs, xr)
	tl.store(y_imag + offs, xi)


def fft1d_ct_n8(x: torch.Tensor, direction: int = 1) -> torch.Tensor:
	"""Cooley–Tukey FFT for N=8, complex64, single vector."""
	assert x.is_cuda
	assert x.dtype == torch.complex64
	assert x.numel() == 8

	xr = x.real.contiguous()
	xi = x.imag.contiguous()
	yr = torch.empty_like(xr)
	yi = torch.empty_like(xi)

	fft1d_ct_n8_kernel[(1,)](xr, xi, yr, yi, direction=direction, num_warps=1)

	out = torch.complex(yr, yi)
	if direction == -1:
		out = out / 8
	return out
