"""Normal OpenCode method - OpenCode with self-verification loop."""

import json
import logging
import os
import re
import subprocess
from pathlib import Path
from typing import Any

from ..base import BaseMethod, MethodResult

logger = logging.getLogger(__name__)

# Paths relative to this method's directory
METHOD_DIR = Path(__file__).parent
TEMPLATES_DIR = METHOD_DIR / "templates"
INSTRUCTIONS_TEMPLATE = TEMPLATES_DIR / "instructions.md"

# Verification tool path (shared tools directory)
AGENT_BENCH_DIR = METHOD_DIR.parent.parent
VERIFY_SCRIPT = AGENT_BENCH_DIR / "tools" / "verify_single.py"


def _extract_token_usage(output_path: Path) -> dict:
    """Extract token usage from OpenCode JSON output.

    Parses step_finish events and accumulates token counts.
    Returns dict with: input_tokens, output_tokens, cache_creation, cache_read, total.
    """
    usage = {
        "input_tokens": 0,
        "output_tokens": 0,
        "cache_creation": 0,
        "cache_read": 0,
        "total": 0,
    }
    try:
        with open(output_path, "r", errors="replace") as f:
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
    except Exception as e:
        logger.warning(f"Failed to extract token usage: {e}")
    return usage


class NormalOpenCodeMethod(BaseMethod):
    """Enhanced OpenCode method with self-verification capability.

    Same as NormalCCMethod but uses OpenCode CLI instead of Claude Code.
    OpenCode can:
    1. Generate initial implementation
    2. Verify using verify_single.py
    3. Iterate and fix based on errors
    4. Output final code in standard format
    """

    name = "normal_opencode"

    def _load_instructions_template(self) -> str:
        """Load instructions template from file."""
        if INSTRUCTIONS_TEMPLATE.exists():
            return INSTRUCTIONS_TEMPLATE.read_text()
        else:
            logger.warning(f"Instructions template not found: {INSTRUCTIONS_TEMPLATE}")
            return ""

    def _replace_output_section(self, base_prompt: str, new_instructions: str) -> str:
        """Replace the output requirements section with new instructions."""
        pattern = r"(##\s*输出要求.*?)$"
        match = re.search(pattern, base_prompt, re.DOTALL | re.IGNORECASE)

        if match:
            final_prompt = base_prompt[:match.start()].rstrip() + "\n\n" + new_instructions
        else:
            final_prompt = base_prompt.rstrip() + "\n\n" + new_instructions

        return final_prompt

    def _build_enhanced_prompt(
        self,
        base_prompt: str,
        operator: str,
        gpu_id: int,
        workspace_dir: Path,
        dataset: str,
    ) -> str:
        """Build enhanced prompt with verification instructions."""
        instructions = self._load_instructions_template()

        # Replace placeholders in instructions
        instructions = instructions.replace("{{GPU_ID}}", str(gpu_id))
        instructions = instructions.replace("{{VERIFY_SCRIPT}}", str(VERIFY_SCRIPT))
        instructions = instructions.replace("{{OPERATOR}}", operator)
        instructions = instructions.replace("{{DATASET}}", dataset)

        from device_manager import get_device_env_var
        instructions = instructions.replace("{{DEVICE_ENV}}", get_device_env_var())

        # Replace output section with method-specific instructions
        enhanced_prompt = self._replace_output_section(base_prompt, instructions)
        enhanced_prompt = enhanced_prompt.replace("{{GPU_ID}}", str(gpu_id))

        return enhanced_prompt

    def launch(
        self,
        operator: str,
        prompt_path: Path,
        workspace_dir: Path,
        gpu_id: int,
        config: dict,
    ) -> Any:
        """Launch OpenCode process with enhanced prompt."""
        workspace_dir.mkdir(parents=True, exist_ok=True)

        # Read base prompt
        with open(prompt_path) as f:
            base_prompt = f.read()

        # Get dataset from config
        dataset = config.get("dataset", "v2_1")

        # Build enhanced prompt with verification instructions
        prompt = self._build_enhanced_prompt(
            base_prompt=base_prompt,
            operator=operator,
            gpu_id=gpu_id,
            workspace_dir=workspace_dir,
            dataset=dataset,
        )

        # Save prompt for debugging
        prompt_save_path = workspace_dir / "prompt.md"
        prompt_save_path.write_text(prompt)

        # Prepare output paths
        stdout_path = workspace_dir / "oc_output.json"
        log_path = workspace_dir / "oc.log"

        # Environment
        env = os.environ.copy()
        env["IS_SANDBOX"] = "1"

        from device_manager import get_device_env_var
        env[get_device_env_var()] = str(gpu_id)

        # Build command
        agent_config = config.get("agent", {})
        opencode_bin = agent_config.get("opencode_bin", "opencode")
        opencode_model = agent_config.get("opencode_model")

        cmd = [
            opencode_bin,
            "run", prompt,
            "--format", "json",
            "--dir", str(workspace_dir),
        ]

        if opencode_model:
            cmd.extend(["--model", opencode_model])

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
        """Extract code from OpenCode output."""
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

        # Extract token usage
        token_usage = _extract_token_usage(stdout_path)

        # If no code found in output, try to read from kernel.py
        if not code:
            kernel_path = workspace_dir / "kernel.py"
            if kernel_path.exists():
                code = kernel_path.read_text()
                logger.info(f"Code extracted from kernel.py")

        # Save kernel if extracted
        if code:
            kernel_path = workspace_dir / "kernel.py"
            kernel_path.write_text(code)

        return MethodResult(
            code=code,
            passed=None,
            speedup=None,
            metadata={"method": "normal_opencode", "token_usage": token_usage},
        )

    def _extract_code(self, output_path: Path) -> str | None:
        """Extract Python code from OpenCode JSON output.

        OpenCode --format json outputs JSON events.
        We try multiple strategies to handle different output formats.
        """
        try:
            content = output_path.read_text(errors="replace")
            if not content.strip():
                return None

            # Strategy 1: Try parsing as JSONL (line-by-line JSON events)
            result_text = ""
            for line in content.splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                    etype = event.get("type")
                    part = event.get("part", {})
                    if etype == "result":
                        result_text = event.get("result", "") or part.get("result", "")
                        break
                    elif etype == "text":
                        result_text += part.get("text", "") or event.get("text", "")
                    elif etype == "message" and "content" in event:
                        result_text = event.get("content", "")
                except json.JSONDecodeError:
                    continue

            # Strategy 2: Try parsing entire content as single JSON
            if not result_text:
                try:
                    data = json.loads(content)
                    if isinstance(data, dict):
                        result_text = (
                            data.get("result", "")
                            or data.get("text", "")
                            or data.get("content", "")
                            or data.get("message", "")
                        )
                    elif isinstance(data, list):
                        for event in data:
                            if isinstance(event, dict):
                                if event.get("type") == "result":
                                    result_text = event.get("result", "")
                                    break
                except json.JSONDecodeError:
                    pass

            # Strategy 3: Use raw content as text
            if not result_text:
                result_text = content

            # Extract the last Python code block
            code_matches = re.findall(r"```python\s*(.*?)\s*```", result_text, re.DOTALL)
            if code_matches:
                return code_matches[-1].strip()

            return None
        except Exception as e:
            logger.warning(f"Failed to extract code: {e}")
            return None

    def get_process(self, handle: Any) -> subprocess.Popen:
        """Get the subprocess.Popen object from handle."""
        return handle["proc"]
