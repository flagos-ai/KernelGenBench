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

@label("ggml_mul_mat_vec_a8")
@parametrize("config", [
    (64, 32, 2),
    (128, 64, 2),
    (256, 128, 2),
    (256, 256, 2),
    (512, 128, 2),
    (512, 256, 2),
    (512, 512, 2),
    (1024, 128, 2),
    (1024, 256, 2),
    (1024, 512, 2),
    (2048, 256, 2),
    (2048, 512, 2),
    (4096, 512, 2),
])
def test_accuracy_ggml_mul_mat_vec_a8(config):
    m = config[0]
    n = config[1]
    quant_type = config[2]
    block_size = 18

    # Construct valid Q4_0 data: each 32-element block = 2 bytes (fp16 scale) + 16 bytes (4-bit values)
    num_qblocks = n // 32
    W = torch.zeros(m, num_qblocks * block_size, device=device, dtype=torch.uint8)
    for bi in range(num_qblocks):
        off = bi * block_size
        # Set fp16 scale = 1.0 (0x3C00) in little-endian
        W[:, off] = 0x00
        W[:, off + 1] = 0x3C
        # Random 4-bit quantized values
        W[:, off + 2:off + 18] = torch.randint(0, 255, (m, 16), device=device, dtype=torch.uint8)
    X = torch.randn(1, m, device=device, dtype=torch.float32)

    ref_out = kernelgenbench.baseline.ggml_mul_mat_vec_a8(W, X, quant_type, n)
    act_out = kernelgenbench.triton.ggml_mul_mat_vec_a8(W, X, quant_type, n)

    assert_close(act_out, ref_out, torch.float32)

    # ===== Performance Test =====
    if m < 256 or n < 128:
        return None

    W_bench = W.clone()
    X_bench = torch.randn(1, m, device=device, dtype=torch.float32)

    ms_baseline = triton.testing.do_bench(
        lambda: kernelgenbench.baseline.ggml_mul_mat_vec_a8(W_bench, X_bench, quant_type, n),
        warmup=25, rep=100
    )

    ms_triton = triton.testing.do_bench(
        lambda: kernelgenbench.triton.ggml_mul_mat_vec_a8(W_bench, X_bench, quant_type, n),
        warmup=25, rep=100
    )

    speedup = ms_baseline / ms_triton if ms_triton > 0 else float('inf')
    return CustomBenchmarkResult(
        ref_time=ms_baseline, res_time=ms_triton, speedup=speedup
    )
