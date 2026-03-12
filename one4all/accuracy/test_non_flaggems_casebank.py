import os
import sys
from typing import Iterable

import pytest
import torch


THIS_DIR = os.path.dirname(__file__)
FLAGBENCH_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
FLAGBENCH_SRC = os.path.join(FLAGBENCH_ROOT, "src")
if FLAGBENCH_SRC not in sys.path:
    sys.path.insert(0, FLAGBENCH_SRC)
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)

from flagbench.dataset.kernel_list import NON_FLAGGEMS_OPERATORS  # noqa: E402
from sandbox.utils.accuracy_utils import gems_assert_close  # noqa: E402
from casebank_non_flaggems import get_all_cases  # noqa: E402


def _device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _assert_outputs(actual, ref, dtype):
    if isinstance(actual, (tuple, list)):
        assert isinstance(ref, type(actual))
        assert len(actual) == len(ref)
        for act_item, ref_item in zip(actual, ref):
            _assert_outputs(act_item, ref_item, dtype)
        return
    gems_assert_close(actual, ref, dtype=dtype)


CASES = [c for c in get_all_cases() if c.op_name in NON_FLAGGEMS_OPERATORS]
CASE_IDS = [c.name for c in CASES]


@pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
def test_non_flaggems_casebank(case):
    device = _device()
    op = case.op()
    ref_args, ref_kwargs, act_args, act_kwargs = case.build(device)
    ref_out = op(*ref_args, **ref_kwargs)
    act_out = op(*act_args, **act_kwargs)
    _assert_outputs(act_out, ref_out, case.dtype)
