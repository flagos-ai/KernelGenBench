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

"""
Integration tests for the full 7-layer competition anti-cheat pipeline.

Tests that cheating kernels are caught and clean kernels pass.
"""
import os
import sys
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from sandbox.competition import run_check, CheckConfig, CheckResult

KERNELS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cheat_kernels")


def _gen_inputs():
    """Generate test inputs for a matmul kernel."""
    import torch
    return (
        torch.randn(128, 256, device='cuda', dtype=torch.float16),
        torch.randn(256, 128, device='cuda', dtype=torch.float16),
    )


class TestIntegration:
    """Full pipeline integration tests."""

    @pytest.mark.skipif(not os.environ.get('RUN_INTEGRATION'), reason="Set RUN_INTEGRATION=1 to run slow tests")
    def test_clean_kernel_passes(self):
        """A clean kernel should pass all checks."""
        path = os.path.join(KERNELS, "clean_kernel.py")
        result = run_check(
            kernel_path=path,
            generate_inputs=_gen_inputs,
            num_tests=3,
            verbose=False,
        )
        assert result.passed, f"Clean kernel should pass, got: {result.reason}"

    @pytest.mark.skipif(not os.environ.get('RUN_INTEGRATION'), reason="Set RUN_INTEGRATION=1 to run slow tests")
    def test_disk_cache_cheat_caught(self):
        """A kernel that caches to disk should be caught."""
        path = os.path.join(KERNELS, "disk_cache_cheat.py")
        result = run_check(
            kernel_path=path,
            generate_inputs=_gen_inputs,
            num_tests=3,
            verbose=False,
        )
        assert not result.passed, f"Disk cache cheat should be caught, got: {result.reason}"

    @pytest.mark.skipif(not os.environ.get('RUN_INTEGRATION'), reason="Set RUN_INTEGRATION=1 to run slow tests")
    def test_forbidden_import_cheat_caught(self):
        """A kernel that imports forbidden modules should be caught."""
        path = os.path.join(KERNELS, "forbidden_import_cheat.py")
        result = run_check(
            kernel_path=path,
            generate_inputs=_gen_inputs,
            num_tests=3,
            verbose=False,
        )
        assert not result.passed, f"Forbidden import cheat should be caught, got: {result.reason}"

    @pytest.mark.skipif(not os.environ.get('RUN_INTEGRATION'), reason="Set RUN_INTEGRATION=1 to run slow tests")
    def test_print_cheat_blocked(self):
        """A kernel that uses print() should be blocked."""
        path = os.path.join(KERNELS, "print_cheat.py")
        result = run_check(
            kernel_path=path,
            generate_inputs=_gen_inputs,
            num_tests=3,
            verbose=False,
        )
        assert not result.passed, f"Print cheat should be blocked, got: {result.reason}"

    @pytest.mark.skipif(not os.environ.get('RUN_INTEGRATION'), reason="Set RUN_INTEGRATION=1 to run slow tests")
    def test_global_state_cheat_caught(self):
        """A kernel with global state caching should be caught."""
        path = os.path.join(KERNELS, "global_state_cheat.py")
        result = run_check(
            kernel_path=path,
            generate_inputs=_gen_inputs,
            num_tests=3,
            verbose=False,
        )
        assert not result.passed, f"Global state cheat should be caught, got: {result.reason}"

    def test_check_config(self):
        """CheckConfig works correctly."""
        config = CheckConfig(
            kernel_path="/tmp/test.py",
            generate_inputs=_gen_inputs,
            num_tests=5,
        )
        assert config.kernel_path == "/tmp/test.py"
        assert config.num_tests == 5
        assert config.timing_runs == 4  # default

    def test_check_result_pass(self):
        """CheckResult.passed is True for clean results."""
        result = CheckResult(passed=True, reason="All clear")
        assert result.passed
        assert result.reason == "All clear"

    def test_check_result_fail(self):
        """CheckResult.passed is False for caught results."""
        result = CheckResult(passed=False, reason="Import hook: os.socket blocked")
        assert not result.passed

    def test_check_result_to_dict(self):
        """CheckResult.to_dict() works."""
        result = CheckResult(passed=True, reason="OK", details={"cv": 0.05})
        d = result.to_dict()
        assert d["passed"] is True
        assert d["reason"] == "OK"
        assert d["details"]["cv"] == 0.05


class TestCheatKernelDetection:
    """Test that cheat kernels are detected at import time (without running full pipeline)."""

    @pytest.mark.skipif(not os.environ.get('RUN_INTEGRATION'), reason="Set RUN_INTEGRATION=1 to run slow tests")
    def test_cuda_graph_cheat_fails_at_import(self):
        """CUDA Graph cheat should fail when loaded in sandbox."""
        path = os.path.join(KERNELS, "cuda_graph_cheat.py")
        result = run_check(
            kernel_path=path,
            generate_inputs=_gen_inputs,
            num_tests=3,
            verbose=False,
        )
        assert not result.passed, f"CUDA Graph cheat should be caught, got: {result.reason}"
