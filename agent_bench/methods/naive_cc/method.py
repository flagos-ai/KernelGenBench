"""Naive CC method - single call to Claude Code."""

import json
import logging
import os
import re
import subprocess
from pathlib import Path
from typing import Any

from ..base import BaseMethod, MethodResult

logger = logging.getLogger(__name__)


class NaiveCCMethod(BaseMethod):
    """Single-call Claude Code method.

    This is the simplest method: one CC call per operator.
    CC is expected to output code in a ```python block.
    """

    name = "naive_cc"

    def launch(
        self,
        operator: str,
        prompt_path: Path,
        workspace_dir: Path,
        gpu_id: int,
        config: dict,
    ) -> Any:
        """Launch CC process."""
        workspace_dir.mkdir(parents=True, exist_ok=True)

        # Read and prepare prompt
        with open(prompt_path) as f:
            prompt = f.read()
        prompt = prompt.replace("{{GPU_ID}}", str(gpu_id))

        # Prepare output paths
        stdout_path = workspace_dir / "cc_output.jsonl"
        log_path = workspace_dir / "cc.log"

        # Environment
        env = os.environ.copy()
        env.pop("CLAUDECODE", None)  # Allow launching CC from within CC
        env["IS_SANDBOX"] = "1"

        # Build command
        agent_config = config.get("agent", {})
        claude_bin = agent_config.get("bin", "claude")
        budget = agent_config.get("budget")

        cmd = [
            claude_bin,
            "-p", prompt,
            "--dangerously-skip-permissions",
            "--output-format", "stream-json",
            "--verbose",
        ]

        if budget:
            cmd.extend(["--max-budget-usd", str(budget)])

        # Launch process
        stdout_file = open(stdout_path, "w")
        stderr_file = open(log_path, "w")

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

        # Store context for finish()
        return {
            "proc": proc,
            "stdout_path": stdout_path,
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
        """Extract code from CC output."""
        stdout_path = handle["stdout_path"]
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

        # Extract code from output
        code = self._extract_code(stdout_path)

        # Save kernel if extracted
        if code:
            kernel_path = workspace_dir / "kernel.py"
            kernel_path.write_text(code)

        return MethodResult(
            code=code,
            passed=None,
            speedup=None,
            metadata={"cc_calls": 1},
        )

    def _extract_code(self, output_path: Path) -> str | None:
        """Extract Python code from CC stream-json output."""
        try:
            result_text = ""
            with open(output_path, "r", errors="replace") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        event = json.loads(line)
                        if event.get("type") == "result":
                            result_text = event.get("result", "")
                            break
                    except json.JSONDecodeError:
                        continue

            if not result_text:
                return None

            # Extract Python code block
            code_match = re.search(r"```python\s*(.*?)\s*```", result_text, re.DOTALL)
            if code_match:
                return code_match.group(1).strip()

            return None
        except Exception as e:
            logger.warning(f"Failed to extract code: {e}")
            return None

    def get_process(self, handle: Any) -> subprocess.Popen:
        """Get the subprocess.Popen object from handle."""
        return handle["proc"]
