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
from sandbox.verifier.test_parametrize import parametrize, label
from sandbox.utils.accuracy_utils import kernelgenbench_assert_close as assert_close
from sandbox.utils.accuracy_utils import CustomBenchmarkResult
import torch
import triton

@label("moe_sum")
@parametrize("num_tokens", [1, 128, 1024, 5333])
@parametrize("hidden_size", [32, 512, 4096])
@parametrize("topk", [2, 4])
@parametrize("dtype", [torch.float16, torch.bfloat16])
def test_accuracy_moe_sum(num_tokens, hidden_size, topk, dtype):
    input_t = torch.randn(num_tokens * topk, hidden_size, device='cuda', dtype=dtype)
    ref_out = torch.zeros(num_tokens, hidden_size, device='cuda', dtype=dtype)
    act_out = torch.zeros(num_tokens, hidden_size, device='cuda', dtype=dtype)

    kernelgenbench.baseline.moe_sum(input_t, ref_out)
    kernelgenbench.triton.moe_sum(input_t, act_out)
    assert_close(act_out, ref_out, dtype)

    if num_tokens * hidden_size < (1 << 20):
        return None
    inp_b = torch.randn(num_tokens * topk, hidden_size, device='cuda', dtype=dtype)
    out_baseline = torch.zeros(num_tokens, hidden_size, device='cuda', dtype=dtype)
    out_triton = torch.zeros(num_tokens, hidden_size, device='cuda', dtype=dtype)

    ms_base = triton.testing.do_bench(
        lambda: kernelgenbench.baseline.moe_sum(inp_b, out_baseline),
        warmup=25, rep=100)
    ms_tri = triton.testing.do_bench(
        lambda: kernelgenbench.triton.moe_sum(inp_b, out_triton),
        warmup=25, rep=100)
    return CustomBenchmarkResult(ref_time=ms_base, res_time=ms_tri,
                                 speedup=ms_base / ms_tri if ms_tri > 0 else float("inf"))
