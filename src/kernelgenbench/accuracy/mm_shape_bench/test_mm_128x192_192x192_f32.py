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
import torch
from sandbox.config import DEVICE as device
from sandbox.verifier.test_parametrize import label
from sandbox.register import REGISTERED_OPS
from sandbox.utils.accuracy_utils import kernelgenbench_assert_close, to_reference
from runtime import get_triton_testing
from sandbox.utils.accuracy_utils import CustomBenchmarkResult


@label("mm_128x192_192x192_f32")
def test_accuracy_mm_128x192_192x192_f32():
    """mm: (128,192) x (192,192) dtype=f32 — freq=708"""
    M, K, N = 128, 192, 192
    dtype = torch.float32
    inp1 = torch.randn(M, K, dtype=dtype, device=device)
    inp2 = torch.randn(K, N, dtype=dtype, device=device)
    ref_inp1 = to_reference(inp1, True)
    ref_inp2 = to_reference(inp2, True)

    ref_out = torch.mm(ref_inp1, ref_inp2)
    with kernelgenbench.use_ops(REGISTERED_OPS):
        res_out = torch.mm(inp1, inp2)

    kernelgenbench_assert_close(res_out, ref_out, dtype, reduce_dim=K)

    quantiles = [0.5, 0.2, 0.8]
    ms_torch, _, _ = get_triton_testing().do_bench(
        lambda: torch.mm(ref_inp1.clone(), ref_inp2.clone()), rep=100, quantiles=quantiles)
    with kernelgenbench.use_ops(REGISTERED_OPS):
        ms_triton, _, _ = get_triton_testing().do_bench(
            lambda: torch.mm(inp1.clone(), inp2.clone()), rep=100, quantiles=quantiles)
    return CustomBenchmarkResult(ref_time=ms_torch, res_time=ms_triton, speedup=ms_torch / ms_triton)
