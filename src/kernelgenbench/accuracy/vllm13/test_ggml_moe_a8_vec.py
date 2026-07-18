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

import kernelgenbench
from sandbox.config import DEVICE as device
from sandbox.verifier.test_parametrize import parametrize, label
from sandbox.utils.accuracy_utils import kernelgenbench_assert_close as assert_close
from sandbox.utils.accuracy_utils import CustomBenchmarkResult
import torch
import triton
import struct

def _make_q8_0_weight(num_experts, row, n):
    """Construct valid GGML Q8_0 quantized MoE weight tensor.

    Q8_0: each block of 32 values = 2 bytes fp16 scale + 32 bytes int8 = 34 bytes.
    Shape: (num_experts, row, n_blocks_per_row * 34)
    """
    bytes_per_block = 34
    n_blocks = n // 32
    W = torch.zeros(num_experts, row, n_blocks * bytes_per_block, dtype=torch.uint8)
    for e in range(num_experts):
        for r in range(row):
            for b in range(n_blocks):
                off = b * bytes_per_block
                scale = torch.randn(1).abs().item() * 0.5 + 0.1
                sb = struct.pack('<e', scale)
                W[e, r, off] = sb[0]
                W[e, r, off + 1] = sb[1]
                W[e, r, off + 2:off + 34] = torch.randint(0, 256, (32,), dtype=torch.uint8)
    return W.to(device)

@label("ggml_moe_a8_vec")
@parametrize("config", [
    # (num_experts, tokens, top_k, hidden, row)
    # --- Small: accuracy-only ---
    (4, 2, 2, 64, 32),
    (4, 4, 2, 128, 64),
    # --- Medium: accuracy + perf ---
    (8, 8, 2, 256, 128),
    (8, 16, 2, 512, 256),
    # --- Large / real MoE shapes: accuracy + perf ---
    (8, 32, 2, 1024, 512),       # Mixtral-scale experts
    (8, 64, 2, 2048, 1024),      # larger MoE
    (16, 32, 2, 4096, 2048),     # Qwen-MoE scale
])
def test_accuracy_ggml_moe_a8_vec(config):
    # ===== Accuracy Test =====
    num_experts, tokens, top_k, hidden, row = config
    quant_type = 8  # Q8_0

    X = torch.randn(tokens, hidden, dtype=torch.float32, device=device)
    W = _make_q8_0_weight(num_experts, row, hidden)
    topk_ids = torch.randint(0, num_experts, (tokens, top_k), dtype=torch.int32, device=device)

    ref_out = kernelgenbench.baseline.ggml_moe_a8_vec(X, W, topk_ids, top_k, quant_type, row, tokens)
    act_out = kernelgenbench.triton.ggml_moe_a8_vec(X, W, topk_ids, top_k, quant_type, row, tokens)

    assert_close(act_out, ref_out, torch.float32)

    # ===== Performance Test =====
    if tokens * hidden < 4096:
        return None

    ms_baseline = triton.testing.do_bench(
        lambda: kernelgenbench.baseline.ggml_moe_a8_vec(X, W, topk_ids, top_k, quant_type, row, tokens),
        warmup=25, rep=100)
    ms_triton = triton.testing.do_bench(
        lambda: kernelgenbench.triton.ggml_moe_a8_vec(X, W, topk_ids, top_k, quant_type, row, tokens),
        warmup=25, rep=100)

    speedup = ms_baseline / ms_triton if ms_triton > 0 else float("inf")
    return CustomBenchmarkResult(ref_time=ms_baseline, res_time=ms_triton, speedup=speedup)
