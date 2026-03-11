"""Iterative optimizer method - multi-round optimization with CC."""

import json
import logging
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import yaml

from ..base import BaseMethod, MethodResult

logger = logging.getLogger(__name__)

# Directory containing this method's resources
METHOD_DIR = Path(__file__).resolve().parent

# Timeout buffer in seconds for overhead (process startup, file I/O, etc.)
TIMEOUT_BUFFER_SECONDS = 600


class IterativeOptimizerMethod(BaseMethod):
    """Multi-round iterative optimization method.

    This method runs CC multiple times, each time analyzing previous versions
    and creating a new optimized version. The loop continues until:
    - Target speedup is achieved
    - Maximum CC calls reached
    - No new versions are being produced

    The optimization loop runs in a subprocess (worker.py) to allow
    proper timeout handling and parallel execution of multiple operators.

    Directory structure:
        methods/iterative_optimizer/
        ├── __init__.py
        ├── method.py          # This file
        ├── worker.py          # Subprocess worker
        ├── templates/
        │   └── optimize.md    # CC prompt template
        └── tools/
            └── verify_single.py  # Verification tool for CC
    """

    name = "iterative_optimizer"

    def launch(
        self,
        operator: str,
        prompt_path: Path,
        workspace_dir: Path,
        gpu_id: int,
        config: dict,
    ) -> Any:
        """Launch the iterative optimization subprocess.

        Args:
            operator: Operator name to optimize
            prompt_path: Path to the operator's prompt/context file
            workspace_dir: Working directory for this operator
            gpu_id: GPU ID for verification
            config: Configuration dict (must include 'dataset')

        Returns:
            Handle dict containing process and file references
        """
        workspace_dir.mkdir(parents=True, exist_ok=True)

        # Get configuration
        agent_config = config.get("agent", {})
        max_cc_calls = agent_config.get("max_optimize_calls", 10)
        target_speedup = agent_config.get("target_speedup", 1.0)
        verify_timeout = agent_config.get("verify_timeout", 300)

        # Get dataset from config (required)
        dataset = config.get("dataset", "v2_1")

        # Write config to temp file for worker to read
        # This avoids fragile path discovery in the worker
        config_file = workspace_dir / "_config.yaml"
        with open(config_file, "w") as f:
            yaml.dump(config, f, default_flow_style=False)

        # Prepare output paths
        stdout_path = workspace_dir / "worker_output.log"
        stderr_path = workspace_dir / "worker_error.log"

        # Environment
        env = os.environ.copy()
        env.pop("CLAUDECODE", None)  # Allow launching CC from within CC
        env["IS_SANDBOX"] = "1"

        # Build command for worker script
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
            "--max-cc-calls", str(max_cc_calls),
            "--target-speedup", str(target_speedup),
            "--verify-timeout", str(verify_timeout),
        ]

        # Launch process
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

        logger.info(f"Launched iterative optimizer for {operator} (GPU={gpu_id}, max_calls={max_cc_calls})")

        # Store context for finish()
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
        """Extract results from the completed optimization process.

        Args:
            operator: Operator name
            handle: Handle from launch()
            workspace_dir: Working directory
            config: Configuration dict

        Returns:
            MethodResult with best code and metadata
        """
        stdout_file = handle["stdout_file"]
        stderr_file = handle["stderr_file"]

        # Close file handles
        try:
            if not stdout_file.closed:
                stdout_file.close()
        except Exception:
            pass
        try:
            if not stderr_file.closed:
                stderr_file.close()
        except Exception:
            pass

        # Read result from worker output
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
                        "cc_calls": result_data.get("cc_calls", 0),
                        "method": "iterative_optimizer",
                    },
                )
            except Exception as e:
                logger.warning(f"Failed to read result for {operator}: {e}")

        # Fallback: try to find the best version from workspace
        code, version, cc_calls = self._find_best_code(workspace_dir)

        return MethodResult(
            code=code,
            passed=code is not None,
            speedup=None,
            metadata={
                "best_version": version,
                "cc_calls": cc_calls,
                "method": "iterative_optimizer",
            },
        )

    def _find_best_code(self, workspace_dir: Path) -> tuple[str | None, str | None, int]:
        """Fallback: find the best passing version's code.

        Args:
            workspace_dir: Path to workspace directory

        Returns:
            Tuple of (best_code, best_version, version_count)
        """
        versions = []
        for d in workspace_dir.iterdir():
            if d.is_dir() and d.name.startswith("v") and d.name[1:].isdigit():
                versions.append(d.name)

        if not versions:
            return None, None, 0

        versions = sorted(versions, key=lambda x: int(x[1:]))
        version_count = len(versions)

        # Find the best passing version by speedup
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

        # If no passing version with speedup, return the latest passing version
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
        """Get the subprocess.Popen object from handle."""
        return handle["proc"]

    def get_timeout(self, config: dict) -> int:
        """Get timeout in seconds.

        For iterative optimizer, we need more time since it runs multiple CC calls.
        Total timeout = max_cc_calls * (cc_timeout + verify_timeout) + buffer

        Args:
            config: Configuration dict

        Returns:
            Timeout in seconds
        """
        agent_config = config.get("agent", {})

        # Check for explicit timeout override first
        explicit_timeout = agent_config.get("iterative_timeout")
        if explicit_timeout:
            return explicit_timeout

        # Calculate total timeout based on iteration parameters
        max_cc_calls = agent_config.get("max_optimize_calls", 10)
        cc_timeout = agent_config.get("cc_timeout", 1800)
        verify_timeout = agent_config.get("verify_timeout", 300)

        # Each iteration needs: cc_timeout + verify_timeout
        # Plus buffer for process startup, file I/O, etc.
        total_timeout = max_cc_calls * (cc_timeout + verify_timeout) + TIMEOUT_BUFFER_SECONDS

        return total_timeout
