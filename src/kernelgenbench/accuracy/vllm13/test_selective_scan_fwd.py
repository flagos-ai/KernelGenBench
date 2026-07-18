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

@label("selective_scan_fwd")
@parametrize("ssm_cfg", [
    (1, 16, 8, 1, [1]),           # minimal single-seq
    (2, 32, 16, 1, [3, 5]),       # small 2-batch
    (4, 64, 16, 1, [7, 3, 12, 1]),# 4-batch varied seqlens
    (2, 128, 32, 2, [16, 32]),    # larger dim/dstate, 2 groups
    (3, 256, 16, 4, [64, 32, 48]),# large dim, 4 groups
])
@parametrize("dtype", [torch.float16, torch.bfloat16])
@parametrize("delta_softplus", [False, True])
@parametrize("opt_combo", [0, 1, 3, 7])  # bitmask: 1->D_, 2->z_, 4->delta_bias_
@parametrize("with_initial_state", [False, True])
def test_accuracy_selective_scan_fwd(ssm_cfg, dtype, delta_softplus, opt_combo, with_initial_state):
    # ===== Accuracy Test =====
    # Unpack SSM config: varlen mode with query_start_loc
    batch, dim, dstate, ngroups, seqlens = ssm_cfg
    total_length = sum(seqlens)

    has_D = bool(opt_combo & 0x1)
    has_z = bool(opt_combo & 0x2)
    has_bias = bool(opt_combo & 0x4)

    # input_t tensors: u, delta, B, C, z_ share the same dtype
    # u, delta: (dim, total_length)
    u = torch.randn(dim, total_length, device='cuda', dtype=dtype)
    delta = torch.randn(dim, total_length, device='cuda', dtype=dtype)
    # A: (dim, dstate) — float32, must be negative for numerical stability (Mamba convention)
    A = -torch.rand(dim, dstate, device='cuda', dtype=torch.float32) - 0.1
    # B, C: (ngroups, dstate, total_length) — input dtype
    B = torch.randn(ngroups, dstate, total_length, device='cuda', dtype=dtype)
    C = torch.randn(ngroups, dstate, total_length, device='cuda', dtype=dtype)

    # Optional tensors
    D_ = torch.randn(dim, device='cuda', dtype=torch.float32) if has_D else None
    z_ = torch.randn(dim, total_length, device='cuda', dtype=dtype) if has_z else None
    delta_bias_ = (torch.rand(dim, device='cuda', dtype=torch.float32) * 0.2 - 0.1) if has_bias else None

    # ssm_states: (batch, dim, dstate) — float32
    ssm_states = torch.zeros(batch, dim, dstate, device='cuda', dtype=torch.float32)

    # Varlen mode: query_start_loc is cumulative seqlens prepended with 0
    cumsum = [0]
    for s in seqlens:
        cumsum.append(cumsum[-1] + s)
    query_start_loc = torch.tensor(cumsum, device='cuda', dtype=torch.int32)
    cache_indices = torch.arange(batch, device='cuda', dtype=torch.int32)

    if with_initial_state:
        has_initial_state = torch.randint(0, 2, (batch,), device='cuda', dtype=torch.bool)
    else:
        has_initial_state = torch.zeros(batch, device='cuda', dtype=torch.bool)

    # Clone tensors for ref and act (in-place op modifies delta/z_ and ssm_states)
    ref_u = u.clone()
    ref_delta = delta.clone()
    ref_z = z_.clone() if has_z else None
    ref_ssm = ssm_states.clone()

    act_u = u.clone()
    act_delta = delta.clone()
    act_z = z_.clone() if has_z else None
    act_ssm = ssm_states.clone()

    # Common kwargs
    common = dict(
        A=A, B=B, C=C, D_=D_, delta_bias_=delta_bias_,
        delta_softplus=delta_softplus,
        query_start_loc=query_start_loc,
        cache_indices=cache_indices,
        has_initial_state=has_initial_state,
        pad_slot_id=-1, block_size=1024,
        block_idx_first_scheduled_token=None,
        block_idx_last_scheduled_token=None,
        initial_state_idx=None,
    )

    # Call baseline
    kernelgenbench.baseline.selective_scan_fwd(
        u=ref_u, delta=ref_delta, z_=ref_z, ssm_states=ref_ssm, **common)

    # Call triton
    kernelgenbench.triton.selective_scan_fwd(
        u=act_u, delta=act_delta, z_=act_z, ssm_states=act_ssm, **common)

    # Compare ssm_states (always modified in-place)
    assert_close(act_ssm, ref_ssm, torch.float32)
    # Compare output: z_ if present, else delta
    if has_z:
        assert_close(act_z, ref_z, dtype)
    else:
        assert_close(act_delta, ref_delta, dtype)

    # ===== Performance Test =====
    # Skip small sizes for performance test
    if total_length * dim < 4096:
        return None

    # Benchmark helper: create fresh cloned inputs each call (in-place op)
    def bench_fn(impl):
        def fn():
            d = delta.clone()
            z = z_.clone() if has_z else None
            s = ssm_states.clone()
            impl(u=u.clone(), delta=d, z_=z, ssm_states=s, **common)
        return fn

    ms_baseline = triton.testing.do_bench(
        bench_fn(kernelgenbench.baseline.selective_scan_fwd), warmup=25, rep=100)
    ms_triton = triton.testing.do_bench(
        bench_fn(kernelgenbench.triton.selective_scan_fwd), warmup=25, rep=100)

    speedup = ms_baseline / ms_triton
    return CustomBenchmarkResult(
        ref_time=ms_baseline, res_time=ms_triton, speedup=speedup)
