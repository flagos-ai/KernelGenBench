"""Normal CC method - CC with self-verification loop."""

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


class NormalCCMethod(BaseMethod):
    """Enhanced CC method with self-verification capability.

    This method allows CC to:
    1. Generate initial implementation
    2. Verify using verify_single.py
    3. Iterate and fix based on errors
    4. Output final code in standard format
    """

    name = "normal_cc"

    def _load_instructions_template(self) -> str:
        """Load instructions template from file."""
        if INSTRUCTIONS_TEMPLATE.exists():
            return INSTRUCTIONS_TEMPLATE.read_text()
        else:
            logger.warning(f"Instructions template not found: {INSTRUCTIONS_TEMPLATE}")
            return ""

    def _replace_output_section(self, base_prompt: str, new_instructions: str) -> str:
        """Replace the output requirements section with new instructions.

        Handles multiple possible section headers and removes everything after
        the section header until the end of file or next major section.

        Args:
            base_prompt: The original operator prompt
            new_instructions: New instructions to append

        Returns:
            Prompt with output section replaced
        """
        # Pattern to match "## 输出要求" section header and everything after it
        # This captures from the header to the end of the file
        pattern = r"(##\s*输出要求.*?)$"
        match = re.search(pattern, base_prompt, re.DOTALL | re.IGNORECASE)

        if match:
            # Remove the old section and append new instructions
            final_prompt = base_prompt[:match.start()].rstrip() + "\n\n" + new_instructions
        else:
            # No output section found, just append
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
        """Build enhanced prompt with verification instructions.

        Args:
            base_prompt: The original operator prompt
            operator: Operator name
            gpu_id: GPU ID for CUDA_VISIBLE_DEVICES
            workspace_dir: Working directory (unused but kept for API consistency)
            dataset: Dataset name for verification

        Returns:
            Final prompt with method-specific instructions
        """
        # Load instructions template
        instructions = self._load_instructions_template()

        # Replace placeholders in instructions
        instructions = instructions.replace("{{GPU_ID}}", str(gpu_id))
        instructions = instructions.replace("{{VERIFY_SCRIPT}}", str(VERIFY_SCRIPT))
        instructions = instructions.replace("{{OPERATOR}}", operator)
        instructions = instructions.replace("{{DATASET}}", dataset)

        # Replace output section with method-specific instructions
        enhanced_prompt = self._replace_output_section(base_prompt, instructions)

        # Replace any remaining GPU_ID placeholders in the base prompt
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
        """Launch CC process with enhanced prompt."""
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
        stdout_path = workspace_dir / "cc_output.jsonl"
        log_path = workspace_dir / "cc.log"

        # Environment
        env = os.environ.copy()
        env.pop("CLAUDECODE", None)  # Allow launching CC from within CC
        env["IS_SANDBOX"] = "1"
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

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

        # If no code found in output, try to read from kernel.py
        if not code:
            kernel_path = workspace_dir / "kernel.py"
            if kernel_path.exists():
                code = kernel_path.read_text()
                logger.info(f"Code extracted from kernel.py")

        # Save kernel if extracted (and not already saved)
        if code:
            kernel_path = workspace_dir / "kernel.py"
            kernel_path.write_text(code)

        return MethodResult(
            code=code,
            passed=None,
            speedup=None,
            metadata={"method": "normal_cc"},
        )

    def _extract_code(self, output_path: Path) -> str | None:
        """Extract Python code from CC stream-json output.

        Looks for the last ```python ... ``` block in the result.
        """
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

            # Extract the LAST Python code block (which should be the final version)
            code_matches = re.findall(r"```python\s*(.*?)\s*```", result_text, re.DOTALL)
            if code_matches:
                # Return the last code block
                return code_matches[-1].strip()

            return None
        except Exception as e:
            logger.warning(f"Failed to extract code: {e}")
            return None

    def get_process(self, handle: Any) -> subprocess.Popen:
        """Get the subprocess.Popen object from handle."""
        return handle["proc"]
