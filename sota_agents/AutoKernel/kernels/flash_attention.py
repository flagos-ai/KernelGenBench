# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
AutoKernel -- Flash Attention kernel.

Current kernel: Flash Attention (block-wise online softmax)
Target metric: throughput (higher is better)
Secondary: correctness must ALWAYS pass

Implements the Flash Attention algorithm (Dao et al.):
  - Block-wise computation of attention
  - Online softmax for numerical stability without materializing full attention matrix
  - Causal masking support
"""

KERNEL_TYPE = "flash_attention"

import torch
import triton
import triton.language as tl
import math


@triton.jit
def flash_attention_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_oz, stride_oh, stride_om, stride_ok,
    Z, H, M_size, N_size,
    D: tl.constexpr,
    sm_scale,
    IS_CAUSAL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Flash attention with online softmax. One program per (batch, head, query-block)."""
    pid_z = tl.program_id(2)  # batch
    pid_h = tl.program_id(1)  # head
    pid_m = tl.program_id(0)  # query block

    # Offsets into the batch and head
    qkv_offset_z = pid_z * stride_qz
    qkv_offset_h = pid_h * stride_qh

    k_offset_z = pid_z * stride_kz
    k_offset_h = pid_h * stride_kh

    v_offset_z = pid_z * stride_vz
    v_offset_h = pid_h * stride_vh

    o_offset_z = pid_z * stride_oz
    o_offset_h = pid_h * stride_oh

    # Query block offsets
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, D)

    # Load Q block [BLOCK_M, D]
    q_ptrs = Q_ptr + qkv_offset_z + qkv_offset_h + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
    q_mask = offs_m[:, None] < M_size
    q = tl.load(q_ptrs, mask=q_mask, other=0.0)

    # Initialize running max and sum for online softmax
    m_i = tl.full((BLOCK_M,), float("-inf"), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_M, D), dtype=tl.float32)

    # Determine the range of KV blocks to iterate over
    if IS_CAUSAL:
        kv_end = tl.minimum(N_size, (pid_m + 1) * BLOCK_M)
    else:
        kv_end = N_size

    # Iterate over KV blocks
    for start_n in range(0, kv_end, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)

        # Load K block [BLOCK_N, D]
        k_ptrs = K_ptr + k_offset_z + k_offset_h + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kk
        k_mask = offs_n[:, None] < N_size
        k = tl.load(k_ptrs, mask=k_mask, other=0.0)

        # Compute QK^T [BLOCK_M, BLOCK_N]
        qk = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        qk += tl.dot(q.to(tl.float32), tl.trans(k.to(tl.float32)))
        qk *= sm_scale

        # Apply causal mask
        if IS_CAUSAL:
            causal_mask = offs_m[:, None] >= offs_n[None, :]
            qk = tl.where(causal_mask, qk, float("-inf"))

        # Mask out-of-bounds keys
        kv_mask = offs_n[None, :] < N_size
        qk = tl.where(kv_mask, qk, float("-inf"))

        # Online softmax update
        m_ij = tl.max(qk, axis=1)  # [BLOCK_M]
        m_new = tl.maximum(m_i, m_ij)

        # Correction factor for previous accumulator
        alpha = tl.exp(m_i - m_new)
        # New attention weights
        p = tl.exp(qk - m_new[:, None])

        # Update running sum
        l_i = l_i * alpha + tl.sum(p, axis=1)

        # Update accumulator: rescale old and add new
        acc = acc * alpha[:, None]

        # Load V block [BLOCK_N, D]
        v_ptrs = V_ptr + v_offset_z + v_offset_h + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vk
        v_mask = offs_n[:, None] < N_size
        v = tl.load(v_ptrs, mask=v_mask, other=0.0)

        acc += tl.dot(p.to(v.dtype), v).to(tl.float32)

        m_i = m_new

    # Final normalization
    acc = acc / l_i[:, None]

    # Store output
    o_ptrs = O_ptr + o_offset_z + o_offset_h + offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok
    o_mask = offs_m[:, None] < M_size
    tl.store(o_ptrs, acc.to(O_ptr.dtype.element_ty), mask=o_mask)


def kernel_fn(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    causal: bool = True,
    sm_scale: float = None,
) -> torch.Tensor:
    """
    Entry point called by bench.py. Must match reference.flash_attention_ref signature.

    Args:
        Q: [batch, heads, seq_len, head_dim]
        K: [batch, heads, seq_len, head_dim]
        V: [batch, heads, seq_len, head_dim]
        causal: whether to apply causal masking
        sm_scale: softmax scale factor, default 1/sqrt(head_dim)
    """
    assert Q.is_cuda and K.is_cuda and V.is_cuda

    Z, H, M_size, D = Q.shape
    _, _, N_size, _ = K.shape

    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(D)

    O = torch.empty_like(Q)

    # Block sizes -- must be powers of 2
    # D (head_dim) must be a constexpr and power of 2 for tl.trans to work
    assert D in (16, 32, 64, 128, 256), f"Head dim {D} not supported, must be power of 2 in [16..256]"

    BLOCK_M = 64
    BLOCK_N = 64

    grid = (triton.cdiv(M_size, BLOCK_M), H, Z)

    flash_attention_kernel[grid](
        Q, K, V, O,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        Z, H, M_size, N_size,
        D=D,
        sm_scale=sm_scale,
        IS_CAUSAL=causal,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
    )

    return O
