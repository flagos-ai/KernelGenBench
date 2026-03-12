from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Tuple

import torch

from sandbox.utils.accuracy_utils import to_reference


@dataclass(frozen=True)
class Case:
    name: str
    op_name: str
    dtype: torch.dtype
    build: Callable[[torch.device], Tuple[Tuple, dict, Tuple, dict]]

    def op(self):
        return _OP_REGISTRY[self.op_name]


def _device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _clone_if_tensor(obj):
    return obj.clone() if torch.is_tensor(obj) else obj


def _map_nested(obj, fn):
    if torch.is_tensor(obj):
        return fn(obj)
    if isinstance(obj, (list, tuple)):
        mapped = [
            _map_nested(item, fn)
            for item in obj
        ]
        return type(obj)(mapped)
    return obj


def _split_ref_act(args, kwargs):
    act_args = _map_nested(args, _clone_if_tensor)
    act_kwargs = {k: _map_nested(v, _clone_if_tensor) for k, v in kwargs.items()}

    ref_args = _map_nested(act_args, to_reference)
    ref_kwargs = {k: _map_nested(v, to_reference) for k, v in act_kwargs.items()}
    return ref_args, ref_kwargs, act_args, act_kwargs


def _rng_pair(seed: int, device: torch.device):
    gen_ref = torch.Generator(device=device).manual_seed(seed)
    gen_act = torch.Generator(device=device).manual_seed(seed)
    return gen_ref, gen_act


def _case_log_normal(shape=(16, 32), dtype=torch.float32, mean=0.0, std=1.0, seed=2025):
    def build(device: torch.device):
        inp = torch.empty(shape, dtype=dtype, device=device)
        ref_inp = inp.clone()
        act_inp = inp.clone()
        gen_ref, gen_act = _rng_pair(seed, device)
        ref_args = (ref_inp, mean, std)
        ref_kwargs = {"generator": gen_ref}
        act_args = (act_inp, mean, std)
        act_kwargs = {"generator": gen_act}
        return ref_args, ref_kwargs, act_args, act_kwargs

    return Case(
        name=f"log_normal_shape{shape}_dtype{dtype}",
        op_name="torch.ops.aten.log_normal",
        dtype=dtype,
        build=build,
    )


def _case_bernoulli(shape=(16, 32), dtype=torch.float32, seed=1234):
    def build(device: torch.device):
        probs = torch.rand(shape, dtype=dtype, device=device)
        ref_probs = probs.clone()
        act_probs = probs.clone()
        gen_ref, gen_act = _rng_pair(seed, device)
        ref_args = (ref_probs,)
        ref_kwargs = {"generator": gen_ref}
        act_args = (act_probs,)
        act_kwargs = {"generator": gen_act}
        return ref_args, ref_kwargs, act_args, act_kwargs

    return Case(
        name=f"bernoulli_shape{shape}_dtype{dtype}",
        op_name="torch.ops.aten.bernoulli",
        dtype=dtype,
        build=build,
    )


def _case_unfold_backward(input_sizes=(16, 32), dim=1, size=4, step=2, dtype=torch.float32):
    def build(device: torch.device):
        S = input_sizes[dim]
        num_windows = (S - size) // step + 1
        grad_shape = list(input_sizes)
        grad_shape[dim] = num_windows
        grad_shape.append(size)
        grad_in = torch.randn(tuple(grad_shape), dtype=dtype, device=device)
        args = (grad_in, list(input_sizes), dim, size, step)
        return _split_ref_act(args, {})

    return Case(
        name=f"unfold_backward_sizes{input_sizes}_d{dim}_s{size}_t{step}",
        op_name="torch.ops.aten.unfold_backward",
        dtype=dtype,
        build=build,
    )


def _case_logit_backward(shape=(16, 32), eps=1e-6, dtype=torch.float32):
    def build(device: torch.device):
        inp = torch.rand(shape, dtype=dtype, device=device)
        grad = torch.randn(shape, dtype=dtype, device=device)
        args = (grad, inp, eps)
        return _split_ref_act(args, {})

    return Case(
        name=f"logit_backward_shape{shape}_eps{eps}",
        op_name="torch.ops.aten.logit_backward",
        dtype=dtype,
        build=build,
    )


def _case_convolution(dtype=torch.float32):
    def build(device: torch.device):
        x = torch.randn((2, 3, 16, 16), dtype=dtype, device=device)
        w = torch.randn((4, 3, 3, 3), dtype=dtype, device=device)
        bias = torch.randn((4,), dtype=dtype, device=device)
        stride = [1, 1]
        padding = [1, 1]
        dilation = [1, 1]
        transposed = False
        output_padding = [0, 0]
        groups = 1
        args = (x, w, bias, stride, padding, dilation, transposed, output_padding, groups)
        return _split_ref_act(args, {})

    return Case(
        name="convolution_2d_basic",
        op_name="torch.ops.aten.convolution",
        dtype=dtype,
        build=build,
    )


def _case_linalg_cross(dtype=torch.float32):
    def build(device: torch.device):
        x = torch.randn((8, 3), dtype=dtype, device=device)
        y = torch.randn((8, 3), dtype=dtype, device=device)
        args = (x, y, -1)
        return _split_ref_act(args, {})

    return Case(
        name="linalg_cross_last_dim",
        op_name="torch.ops.aten.linalg_cross",
        dtype=dtype,
        build=build,
    )


def _case_avg_pool3d(dtype=torch.float32):
    def build(device: torch.device):
        x = torch.randn((1, 2, 6, 8, 8), dtype=dtype, device=device)
        kernel_size = [2, 2, 2]
        stride = [2, 2, 2]
        padding = [0, 0, 0]
        ceil_mode = False
        count_include_pad = False
        divisor_override = None
        args = (x, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override)
        return _split_ref_act(args, {})

    return Case(
        name="avg_pool3d_basic",
        op_name="torch.ops.aten.avg_pool3d",
        dtype=dtype,
        build=build,
    )


def _case_round(dtype=torch.float32):
    def build(device: torch.device):
        x = torch.tensor([-2.5, -1.1, -0.5, 0.0, 0.5, 1.1, 2.5], dtype=dtype, device=device)
        args = (x,)
        return _split_ref_act(args, {})

    return Case(
        name="round_basic",
        op_name="torch.ops.aten.round",
        dtype=dtype,
        build=build,
    )


def _case_baddbmm(dtype=torch.float32):
    def build(device: torch.device):
        batch, m, n, k = 2, 4, 5, 3
        self_t = torch.randn((batch, m, n), dtype=dtype, device=device)
        batch1 = torch.randn((batch, m, k), dtype=dtype, device=device)
        batch2 = torch.randn((batch, k, n), dtype=dtype, device=device)
        beta = 0.5
        alpha = 1.2
        args = (self_t, batch1, batch2, beta, alpha)
        return _split_ref_act(args, {})

    return Case(
        name="baddbmm_basic",
        op_name="torch.ops.aten.baddbmm",
        dtype=dtype,
        build=build,
    )


def _case_addbmm(dtype=torch.float32):
    def build(device: torch.device):
        batch, m, n, k = 2, 4, 5, 3
        self_t = torch.randn((m, n), dtype=dtype, device=device)
        batch1 = torch.randn((batch, m, k), dtype=dtype, device=device)
        batch2 = torch.randn((batch, k, n), dtype=dtype, device=device)
        beta = 0.25
        alpha = 0.75
        args = (self_t, batch1, batch2, beta, alpha)
        return _split_ref_act(args, {})

    return Case(
        name="addbmm_basic",
        op_name="torch.ops.aten.addbmm",
        dtype=dtype,
        build=build,
    )


_OP_REGISTRY = {
    "torch.ops.aten.log_normal": torch.ops.aten.log_normal,
    "torch.ops.aten.bernoulli": torch.ops.aten.bernoulli,
    "torch.ops.aten.unfold_backward": torch.ops.aten.unfold_backward,
    "torch.ops.aten.logit_backward": torch.ops.aten.logit_backward,
    "torch.ops.aten.convolution": torch.ops.aten.convolution,
    "torch.ops.aten.linalg_cross": torch.ops.aten.linalg_cross,
    "torch.ops.aten.avg_pool3d": torch.ops.aten.avg_pool3d,
    "torch.ops.aten.round": torch.ops.aten.round,
    "torch.ops.aten.baddbmm": torch.ops.aten.baddbmm,
    "torch.ops.aten.addbmm": torch.ops.aten.addbmm,
}


def get_non_flaggems_casebank() -> Dict[str, List[Case]]:
    cases = [
        _case_log_normal(),
        _case_bernoulli(),
        _case_unfold_backward(),
        _case_logit_backward(),
        _case_convolution(),
        _case_linalg_cross(),
        _case_avg_pool3d(),
        _case_round(),
        _case_baddbmm(),
        _case_addbmm(),
    ]
    bank: Dict[str, List[Case]] = {}
    for case in cases:
        bank.setdefault(case.op_name, []).append(case)
    return bank


def get_all_cases() -> List[Case]:
    bank = get_non_flaggems_casebank()
    return [case for cases in bank.values() for case in cases]
