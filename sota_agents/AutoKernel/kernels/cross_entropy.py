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
AutoKernel -- Fused Cross Entropy kernel.

Current kernel: Fused log-softmax + NLL loss (row-parallel)
Target metric: throughput (higher is better)
Secondary: correctness must ALWAYS pass

Fuses log-softmax and negative log-likelihood loss into a single kernel pass
per row, avoiding materializing the full softmax output in global memory.
Each program handles one row (one sample in the batch).
"""

KERNEL_TYPE = "cross_entropy"

import torch
import triton
import triton.language as tl


@triton.jit
def cross_entropy_kernel(
    logits_ptr,
    targets_ptr,
    losses_ptr,
    n_cols,
    stride_logits_row,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused cross-entropy: log_softmax + nll_loss per row.
    One program per row (batch element).
    """
    row_idx = tl.program_id(0)

    row_start = logits_ptr + row_idx * stride_logits_row
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    # Load logits row in float32
    logits = tl.load(row_start + col_offsets, mask=mask, other=float("-inf")).to(tl.float32)

    # Numerically stable log-softmax
    # Step 1: find max
    row_max = tl.max(logits, axis=0)

    # Step 2: subtract max, exp, sum
    logits_shifted = logits - row_max
    exp_logits = tl.exp(logits_shifted)
    sum_exp = tl.sum(exp_logits, axis=0)
    log_sum_exp = tl.log(sum_exp)

    # log_softmax = logits_shifted - log_sum_exp
    # We only need the value at the target index

    # Load target for this row
    target = tl.load(targets_ptr + row_idx)

    # Get the logit at the target position
    # log_softmax[target] = logits[target] - max - log(sum(exp(logits - max)))
    target_logit = tl.load(row_start + target).to(tl.float32)
    log_softmax_target = (target_logit - row_max) - log_sum_exp

    # NLL loss = -log_softmax[target]
    loss = -log_softmax_target

    # Store per-sample loss
    tl.store(losses_ptr + row_idx, loss)


def kernel_fn(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Entry point called by bench.py. Must match reference.cross_entropy_ref signature.

    Args:
        logits: [batch_size, vocab_size] raw logits (float16 or float32)
        targets: [batch_size] integer class indices (long)

    Returns:
        Scalar mean cross-entropy loss
    """
    assert logits.is_cuda and targets.is_cuda

    # Handle multi-dim: flatten to 2D
    if logits.ndim > 2:
        logits = logits.view(-1, logits.shape[-1])
        targets = targets.view(-1)

    n_rows, n_cols = logits.shape
    assert targets.shape[0] == n_rows

    losses = torch.empty(n_rows, device=logits.device, dtype=torch.float32)

    BLOCK_SIZE = triton.next_power_of_2(n_cols)

    grid = (n_rows,)
    cross_entropy_kernel[grid](
        logits,
        targets,
        losses,
        n_cols,
        logits.stride(0),
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return losses.mean()
