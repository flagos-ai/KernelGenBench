"""Iterative OpenCode method - multi-round optimization with OpenCode."""

import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

from ..base import BaseMethod, MethodResult

logger = logging.getLogger(__name__)

METHOD_DIR = Path(__file__).resolve().parent
TIMEOUT_BUFFER_SECONDS = 600


def _extract_token_usage_from_workers(workspace_dir: Path) -> dict:
    """Aggregate token usage from all OpenCode output files in workspace."""
    usage = {
        "input_tokens": 0,
        "output_tokens": 0,
        "cache_creation": 0,
        "cache_read": 0,
        "total": 0,
    }
    for oc_file in workspace_dir.glob("oc_output*.json"):
        try:
            with open(oc_file, "r", errors="replace") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        event = json.loads(line)
                        if event.get("type") != "step_finish":
                            continue
                        tokens = event.get("part", {}).get("tokens", {})
                        usage["input_tokens"] += tokens.get("input", 0)
                        usage["output_tokens"] += tokens.get("output", 0)
                        usage["total"] += tokens.get("total", 0)
                        cache = tokens.get("cache", {})
                        usage["cache_read"] += cache.get("read", 0)
                        usage["cache_creation"] += cache.get("write", 0)
                    except json.JSONDecodeError:
                        continue
        except Exception:
            continue
    return usage


class IterativeOpenCodeMethod(BaseMethod):
    """Multi-round iterative optimization method using OpenCode.

    Same as IterativeOptimizerMethod but uses OpenCode CLI instead of Claude Code.
    """

    name = "iterative_opencode"

    def launch(
        self,
        operator: str,
        prompt_path: Path,
        workspace_dir: Path,
        gpu_id: int,
        config: dict,
    ) -> Any:
        workspace_dir.mkdir(parents=True, exist_ok=True)

        agent_config = config.get("agent", {})
        max_oc_calls = agent_config.get("max_optimize_calls", 10)
        target_speedup = agent_config.get("target_speedup", 1.0)
        verify_timeout = agent_config.get("verify_timeout", 600)
        dataset = config.get("dataset", "v2_1")

        config_file = workspace_dir / "_config.yaml"
        with open(config_file, "w") as f:
            yaml.dump(config, f, default_flow_style=False)

        stdout_path = workspace_dir / "worker_output.log"
        stderr_path = workspace_dir / "worker_error.log"

        env = os.environ.copy()
        env["IS_SANDBOX"] = "1"

        from device_manager import get_device_env_var
        env[get_device_env_var()] = str(gpu_id)

        worker_script = METHOD_DIR / "worker.py"

        cmd = [
            sys.executable,
            str(worker_script),
            "--workspace", str(workspace_dir),
            "--operator", operator,
            "--prompt-path", str(prompt_path),
            "--dataset", dataset,
            "--gpu-id", str(gpu_id),
            "--config-path", str(config_file),
            "--max-oc-calls", str(max_oc_calls),
            "--target-speedup", str(target_speedup),
            "--verify-timeout", str(verify_timeout),
        ]

        stdout_file = open(stdout_path, "w")
        stderr_file = open(stderr_path, "w")

        try:
            proc = subprocess.Popen(
                cmd,
                cwd=str(workspace_dir),
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=stdout_file,
                stderr=stderr_file,
                start_new_session=True,
            )
        except Exception:
            stdout_file.close()
            stderr_file.close()
            raise

        logger.info(f"Launched iterative opencode for {operator} (GPU={gpu_id}, max_calls={max_oc_calls})")

        return {
            "proc": proc,
            "workspace_dir": workspace_dir,
            "stdout_file": stdout_file,
            "stderr_file": stderr_file,
        }

    def finish(
        self,
        operator: str,
        handle: Any,
        workspace_dir: Path,
        config: dict,
    ) -> MethodResult:
        stdout_file = handle["stdout_file"]
        stderr_file = handle["stderr_file"]

        for f in (stdout_file, stderr_file):
            try:
                if not f.closed:
                    f.close()
            except Exception:
                pass

        token_usage = _extract_token_usage_from_workers(workspace_dir)

        result_path = workspace_dir / "_result.json"
        if result_path.exists():
            try:
                with open(result_path) as f:
                    result_data = json.load(f)

                return MethodResult(
                    code=result_data.get("best_code"),
                    passed=result_data.get("passed", False),
                    speedup=result_data.get("best_speedup"),
                    metadata={
                        "best_version": result_data.get("best_version"),
                        "oc_calls": result_data.get("oc_calls", 0),
                        "method": "iterative_opencode",
                        "token_usage": token_usage,
                    },
                )
            except Exception as e:
                logger.warning(f"Failed to read result for {operator}: {e}")

        code, version, oc_calls = self._find_best_code(workspace_dir)

        return MethodResult(
            code=code,
            passed=code is not None,
            speedup=None,
            metadata={
                "best_version": version,
                "oc_calls": oc_calls,
                "method": "iterative_opencode",
                "token_usage": token_usage,
            },
        )

    def _find_best_code(self, workspace_dir: Path) -> tuple[str | None, str | None, int]:
        versions = []
        for d in workspace_dir.iterdir():
            if d.is_dir() and d.name.startswith("v") and d.name[1:].isdigit():
                versions.append(d.name)

        if not versions:
            return None, None, 0

        versions = sorted(versions, key=lambda x: int(x[1:]))
        version_count = len(versions)

        best_code = None
        best_speedup = 0.0
        best_version = None

        for v in versions:
            verify_path = workspace_dir / v / "verify.json"
            kernel_path = workspace_dir / v / "kernel.py"

            if verify_path.exists() and kernel_path.exists():
                try:
                    with open(verify_path) as f:
                        data = json.load(f)
                    if data.get("passed", False):
                        speedup = data.get("speedup") or 0.0
                        if speedup >= best_speedup:
                            best_speedup = speedup
                            with open(kernel_path) as f:
                                best_code = f.read()
                            best_version = v
                except Exception:
                    continue

        if best_code is None:
            for v in reversed(versions):
                verify_path = workspace_dir / v / "verify.json"
                kernel_path = workspace_dir / v / "kernel.py"
                if verify_path.exists() and kernel_path.exists():
                    try:
                        with open(verify_path) as f:
                            data = json.load(f)
                        if data.get("passed", False):
                            with open(kernel_path) as f:
                                best_code = f.read()
                            best_version = v
                            break
                    except Exception:
                        continue

        return best_code, best_version, version_count

    def get_process(self, handle: Any) -> subprocess.Popen:
        return handle["proc"]

    def get_timeout(self, config: dict) -> int:
        agent_config = config.get("agent", {})
        explicit_timeout = agent_config.get("iterative_timeout")
        if explicit_timeout:
            return explicit_timeout
        max_oc_calls = agent_config.get("max_optimize_calls", 10)
        oc_timeout = agent_config.get("oc_timeout", 1800)
        verify_timeout = agent_config.get("verify_timeout", 600)
        return max_oc_calls * (oc_timeout + verify_timeout) + TIMEOUT_BUFFER_SECONDS
