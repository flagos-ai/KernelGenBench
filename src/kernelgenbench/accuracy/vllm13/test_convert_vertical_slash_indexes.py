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

@label("convert_vertical_slash_indexes")
@parametrize("config", [
    (2, 4, 8, 4, 4),
    (4, 8, 16, 4, 4),
    (2, 8, 16, 8, 8),
    (4, 16, 32, 8, 8),
    (2, 4, 16, 4, 4),
    (4, 16, 32, 8, 8),
    (4, 32, 64, 8, 8),
    (8, 16, 32, 8, 8),
    (8, 32, 64, 16, 16),
    (8, 32, 64, 8, 8),
    (8, 64, 128, 8, 8),
    (16, 32, 64, 8, 8),
    (16, 64, 128, 16, 16),
])
def test_accuracy_convert_vertical_slash_indexes(config):
    B = config[0]
    q_len = config[1]
    kv_len = config[2]
    bsM = config[3]
    bsN = config[4]

    q_sl = torch.full((B,), q_len, device=device, dtype=torch.int32)
    kv_sl = torch.full((B,), kv_len, device=device, dtype=torch.int32)
    vi = torch.randint(0, kv_len, (B, q_len, q_len), device=device, dtype=torch.int32)
    si = torch.randint(0, kv_len, (B, q_len, q_len), device=device, dtype=torch.int32)

    ref_out = kernelgenbench.baseline.convert_vertical_slash_indexes(
        q_sl, kv_sl, vi, si, kv_len, bsM, bsN)
    act_out = kernelgenbench.triton.convert_vertical_slash_indexes(
        q_sl, kv_sl, vi, si, kv_len, bsM, bsN)

    for i in range(len(ref_out)):
        assert torch.equal(act_out[i], ref_out[i]), \
            f"Output {i} mismatch: max diff={(act_out[i] - ref_out[i]).abs().max()}"

    # ===== Performance Test =====
    if B < 4 or q_len < 16 or kv_len < 32:
        return None

    q_sl_bench = torch.full((B,), q_len, device=device, dtype=torch.int32)
    kv_sl_bench = torch.full((B,), kv_len, device=device, dtype=torch.int32)
    vi_bench = torch.randint(0, kv_len, (B, q_len, q_len), device=device, dtype=torch.int32)
    si_bench = torch.randint(0, kv_len, (B, q_len, q_len), device=device, dtype=torch.int32)

    ms_baseline = triton.testing.do_bench(
        lambda: kernelgenbench.baseline.convert_vertical_slash_indexes(
            q_sl_bench, kv_sl_bench, vi_bench, si_bench, kv_len, bsM, bsN),
        warmup=25, rep=100
    )

    ms_triton = triton.testing.do_bench(
        lambda: kernelgenbench.triton.convert_vertical_slash_indexes(
            q_sl_bench, kv_sl_bench, vi_bench, si_bench, kv_len, bsM, bsN),
        warmup=25, rep=100
    )

    speedup = ms_baseline / ms_triton if ms_triton > 0 else float('inf')
    return CustomBenchmarkResult(
        ref_time=ms_baseline, res_time=ms_triton, speedup=speedup
    )
