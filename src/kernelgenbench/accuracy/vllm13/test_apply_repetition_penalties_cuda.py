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


@label("apply_repetition_penalties_cuda")
@parametrize("shape", [(1, 32), (4, 100), (128, 512), (256, 1024), (512, 4096), (1024, 32000), (2048, 16000), (4096, 8000), (2048, 32000), (4096, 16000)])
@parametrize("dtype", [torch.float32])
def test_accuracy_apply_repetition_penalties_cuda(shape, dtype):
    batch, vocab = shape

    logits_ref = torch.randn(batch, vocab, device=device, dtype=dtype)
    logits_act = logits_ref.clone()
    prompt_mask = torch.randint(0, 2, (batch, vocab), dtype=torch.bool, device=device)
    output_mask = torch.randint(0, 2, (batch, vocab), dtype=torch.bool, device=device)
    penalties = torch.ones(batch, device=device, dtype=dtype) * 1.1

    kernelgenbench.baseline.apply_repetition_penalties_cuda(logits_ref, prompt_mask, output_mask, penalties)
    kernelgenbench.triton.apply_repetition_penalties_cuda(logits_act, prompt_mask, output_mask, penalties)

    assert_close(logits_act, logits_ref, dtype)

    if batch * vocab < 1024 * 32000:
        return None

    logits_baseline = torch.randn(batch, vocab, device=device, dtype=dtype)
    logits_triton = logits_baseline.clone()
    pm = torch.randint(0, 2, (batch, vocab), dtype=torch.bool, device=device)
    om = torch.randint(0, 2, (batch, vocab), dtype=torch.bool, device=device)
    pen = torch.ones(batch, device=device, dtype=dtype) * 1.1

    ms_baseline = triton.testing.do_bench(
        lambda: kernelgenbench.baseline.apply_repetition_penalties_cuda(logits_baseline, pm, om, pen),
        warmup=25, rep=100)
    ms_triton = triton.testing.do_bench(
        lambda: kernelgenbench.triton.apply_repetition_penalties_cuda(logits_triton, pm, om, pen),
        warmup=25, rep=100)

    speedup = ms_baseline / ms_triton if ms_triton > 0 else float("inf")
    return CustomBenchmarkResult(ref_time=ms_baseline, res_time=ms_triton, speedup=speedup)
