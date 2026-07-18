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

def _make_q8_0_data(rows, cols):
    """Construct valid GGML Q8_0 quantized data.

    Q8_0 format: each block of 32 values = 2 bytes fp16 scale + 32 bytes int8.
    Total 34 bytes per block.
    """
    n_blocks_per_row = cols // 32
    bytes_per_block = 34
    data = torch.zeros(rows, n_blocks_per_row * bytes_per_block, dtype=torch.uint8)
    for r in range(rows):
        for b in range(n_blocks_per_row):
            offset = b * bytes_per_block
            scale = torch.randn(1).abs().item() * 0.5 + 0.1
            scale_bytes = struct.pack('<e', scale)
            data[r, offset] = scale_bytes[0]
            data[r, offset + 1] = scale_bytes[1]
            data[r, offset + 2:offset + 34] = torch.randint(0, 256, (32,), dtype=torch.uint8)
    return data.to(device)

@label("ggml_dequantize")
@parametrize("config", [
    # (rows, cols, quant_type, dtype)
    # --- Small: accuracy-only ---
    (32, 128, 8, torch.float16),
    (64, 256, 8, torch.float16),
    # --- Medium: accuracy + perf ---
    (128, 512, 8, torch.float16),
    (256, 1024, 8, torch.float32),
    (512, 2048, 8, torch.float16),
    # --- Large / real model weight shapes: accuracy + perf ---
    (1024, 4096, 8, torch.float16),    # Llama-7B hidden
    (4096, 4096, 8, torch.float16),    # Llama-7B square weight
    (4096, 11008, 8, torch.float32),   # Llama-7B FFN weight
])
def test_accuracy_ggml_dequantize(config):
    # ===== Accuracy Test =====
    rows, cols, quant_type, dtype = config

    W = _make_q8_0_data(rows, cols)

    ref_out = kernelgenbench.baseline.ggml_dequantize(W, quant_type, rows, cols, dtype)
    act_out = kernelgenbench.triton.ggml_dequantize(W, quant_type, rows, cols, dtype)

    assert_close(act_out, ref_out, dtype)

    # ===== Performance Test =====
    if rows * cols < 65536:
        return None

    ms_baseline = triton.testing.do_bench(
        lambda: kernelgenbench.baseline.ggml_dequantize(W, quant_type, rows, cols, dtype),
        warmup=25, rep=100)
    ms_triton = triton.testing.do_bench(
        lambda: kernelgenbench.triton.ggml_dequantize(W, quant_type, rows, cols, dtype),
        warmup=25, rep=100)

    speedup = ms_baseline / ms_triton if ms_triton > 0 else float("inf")
    return CustomBenchmarkResult(ref_time=ms_baseline, res_time=ms_triton, speedup=speedup)
