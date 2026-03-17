#!/usr/bin/env python3
"""Worker script for iterative optimizer method.

This script runs in a subprocess and manages the iterative optimization loop.
It handles CC calls, version detection, verification, and progress tracking.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

# Constants for display truncation (not for verify.json - CC needs full info)
ERROR_DISPLAY_LENGTH = 100
NOTE_TRUNCATE_LENGTH = 50

# Resolve paths
SCRIPT_DIR = Path(__file__).resolve().parent
AGENT_BENCH_DIR = SCRIPT_DIR.parent.parent
TOOLS_DIR = AGENT_BENCH_DIR / "tools"  # Shared tools directory
TEMPLATES_DIR = SCRIPT_DIR / "templates"


def list_versions(workspace_dir: Path) -> list[str]:
    """List all version directories (v1, v2, ...) in workspace.

    Args:
        workspace_dir: Path to workspace directory

    Returns:
        Sorted list of version directory names (e.g., ["v1", "v2", "v3"])
    """
    versions = []
    if not workspace_dir.exists():
        return versions
    for d in workspace_dir.iterdir():
        if d.is_dir() and d.name.startswith("v") and d.name[1:].isdigit():
            versions.append(d.name)
    return sorted(versions, key=lambda x: int(x[1:]))


def detect_new_version(before: list[str], after: list[str]) -> str | None:
    """Detect newly created version directory.

    Args:
        before: List of versions before CC call
        after: List of versions after CC call

    Returns:
        Name of new version directory, or None if no new version
    """
    new = set(after) - set(before)
    if new:
        return sorted(new, key=lambda x: int(x[1:]))[-1]
    return None


def load_optimize_prompt(gpu_id: int, operator: str, dataset: str) -> str:
    """Load and prepare the optimization prompt template.

    Args:
        gpu_id: GPU ID for CUDA_VISIBLE_DEVICES
        operator: Operator name
        dataset: Dataset name

    Returns:
        Prepared prompt string with placeholders replaced
    """
    template_path = TEMPLATES_DIR / "optimize.md"
    with open(template_path) as f:
        prompt = f.read()

    prompt = prompt.replace("{{GPU_ID}}", str(gpu_id))
    prompt = prompt.replace("{{OP_NAME}}", operator)
    prompt = prompt.replace("{{DATASET}}", dataset)
    prompt = prompt.replace("{{TOOLS_DIR}}", str(TOOLS_DIR))

    return prompt


def call_cc(prompt: str, working_dir: Path, config: dict, output_suffix: str = "") -> bool:
    """Call Claude Code with the given prompt.

    Args:
        prompt: The prompt to send to CC
        working_dir: Working directory for CC
        config: Configuration dict with agent settings
        output_suffix: Optional suffix for output files

    Returns:
        True if CC completed successfully (returncode 0)
    """
    suffix = f"_{output_suffix}" if output_suffix else ""
    base_output = working_dir / f"cc_output{suffix}.jsonl"
    base_log = working_dir / f"cc{suffix}.log"

    # Avoid overwriting existing logs from previous runs (e.g., retries)
    # Find a unique filename by adding a counter
    output_path = base_output
    log_path = base_log
    counter = 1
    while output_path.exists() or log_path.exists():
        output_path = working_dir / f"cc_output{suffix}_{counter}.jsonl"
        log_path = working_dir / f"cc{suffix}_{counter}.log"
        counter += 1

    # Environment
    env = os.environ.copy()
    env.pop("CLAUDECODE", None)  # Allow launching CC from within CC
    env["IS_SANDBOX"] = "1"

    # Build command
    agent_config = config.get("agent", {})
    claude_bin = agent_config.get("bin", "claude")
    budget = agent_config.get("budget")
    cc_timeout = agent_config.get("cc_timeout", 1800)

    cmd = [
        claude_bin,
        "-p", prompt,
        "--dangerously-skip-permissions",
        "--output-format", "stream-json",
        "--verbose",
    ]

    if budget:
        cmd.extend(["--max-budget-usd", str(budget)])

    try:
        with open(output_path, "w") as stdout_f, open(log_path, "w") as stderr_f:
            result = subprocess.run(
                cmd,
                cwd=str(working_dir),
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=stdout_f,
                stderr=stderr_f,
                timeout=cc_timeout,
            )
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        print(f"CC call timed out after {cc_timeout}s")
        return False
    except Exception as e:
        print(f"CC call failed: {e}")
        return False


def verify_kernel(
    kernel_path: Path,
    operator: str,
    dataset: str,
    gpu_id: int,
    timeout: int = 600,
) -> dict:
    """Verify a single kernel for correctness using verify_single.py.

    Args:
        kernel_path: Path to the kernel Python file
        operator: Operator name
        dataset: Dataset name (v2, v2_1, cupy)
        gpu_id: GPU ID for verification
        timeout: Timeout in seconds

    Returns:
        Dict with: passed, total_tests, passed_tests, failed_tests, speedup, error
    """
    if not kernel_path.exists():
        return {
            "passed": False,
            "error": f"Kernel file not found: {kernel_path}",
            "total_tests": 0,
            "passed_tests": 0,
            "failed_tests": 0,
        }

    # Validate gpu_id
    if not isinstance(gpu_id, int) or gpu_id < 0:
        return {
            "passed": False,
            "error": f"Invalid GPU ID: {gpu_id}",
            "total_tests": 0,
            "passed_tests": 0,
            "failed_tests": 0,
        }

    # Set environment for verification
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env["DISPATCH_TORCH_LIB"] = "1"
    env["FLAGBENCH_SKIP_BOTH_TEST"] = "1"

    # Call verify_single.py with JSON output
    verify_script = TOOLS_DIR / "verify_single.py"

    # Save logs to kernel's parent directory (the version directory)
    output_dir = kernel_path.parent

    cmd = [
        sys.executable,
        str(verify_script),
        "--code", str(kernel_path),
        "--op", operator,
        "--dataset", dataset,
        "--output-dir", str(output_dir),  # Save logs to version directory
        "--output-json",  # Always use JSON for programmatic parsing
    ]

    try:
        result = subprocess.run(
            cmd,
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout,
        )

        # Parse JSON output - this is the required format
        try:
            output = json.loads(result.stdout)
            return {
                "passed": output.get("passed", False),
                "total_tests": output.get("total_tests", 0),
                "passed_tests": output.get("passed_tests", 0),
                "failed_tests": output.get("failed_tests", 0),
                "speedup": output.get("speedup"),
                "error": output.get("error"),
            }
        except json.JSONDecodeError:
            # JSON parsing failed - this is an error, don't guess
            error_msg = f"Failed to parse verification output as JSON. stdout: {result.stdout[:200]}"
            return {
                "passed": False,
                "error": error_msg,
                "total_tests": 0,
                "passed_tests": 0,
                "failed_tests": 0,
            }

    except subprocess.TimeoutExpired:
        return {
            "passed": False,
            "error": f"Verification timed out after {timeout}s",
            "total_tests": 0,
            "passed_tests": 0,
            "failed_tests": 0,
        }
    except Exception as e:
        # Don't truncate error - full info needed for debugging
        return {
            "passed": False,
            "error": str(e),
            "total_tests": 0,
            "passed_tests": 0,
            "failed_tests": 0,
        }


def get_latest_unverified_version(workspace_dir: Path) -> str | None:
    """Find the latest version that needs verification.

    A version needs verification if:
    - It has kernel.py but no verify.json
    - Or kernel.py is newer than verify.json

    Args:
        workspace_dir: Path to workspace directory

    Returns:
        Name of version directory that needs verification, or None
    """
    versions = list_versions(workspace_dir)
    if not versions:
        return None

    # Check versions from newest to oldest
    for v in reversed(versions):
        version_dir = workspace_dir / v
        kernel_path = version_dir / "kernel.py"
        verify_path = version_dir / "verify.json"

        if not kernel_path.exists():
            continue

        # Needs verification if no verify.json
        if not verify_path.exists():
            return v

        # Needs verification if kernel is newer than verify.json
        if kernel_path.stat().st_mtime > verify_path.stat().st_mtime:
            return v

    return None


def load_existing_verify_result(version_dir: Path) -> dict | None:
    """Load existing verification result from version directory if available.

    Args:
        version_dir: Path to version directory (e.g., workspace/v1/)

    Returns:
        Verification result dict if verify.json exists and is valid, None otherwise
    """
    verify_path = version_dir / "verify.json"
    if not verify_path.exists():
        return None

    try:
        with open(verify_path) as f:
            result = json.load(f)

        # Validate required fields are present
        required_fields = ["passed", "total_tests", "passed_tests", "failed_tests"]
        if all(field in result for field in required_fields):
            return result
        return None
    except (json.JSONDecodeError, IOError):
        return None


def save_verify_result(version_dir: Path, result: dict):
    """Save verification result to version directory.

    Args:
        version_dir: Path to version directory (e.g., workspace/v1/)
        result: Verification result dict
    """
    verify_path = version_dir / "verify.json"
    with open(verify_path, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)


def update_performance_md(
    workspace_dir: Path,
    operator: str,
    target_speedup: float,
    max_calls: int,
    current_call: int,
):
    """Update PERFORMANCE.md with all version results.

    Args:
        workspace_dir: Path to workspace directory
        operator: Operator name
        target_speedup: Target speedup to achieve
        max_calls: Maximum number of CC calls
        current_call: Current CC call number
    """
    versions = list_versions(workspace_dir)
    rows = []
    best_version = None
    best_speedup = 0.0

    for v in versions:
        verify_path = workspace_dir / v / "verify.json"
        notes_path = workspace_dir / v / "notes.md"

        if verify_path.exists():
            with open(verify_path) as f:
                data = json.load(f)

            passed = data.get("passed", False)
            passed_str = "Y" if passed else "N"
            speedup = data.get("speedup")
            speedup_str = f"{speedup:.2f}x" if speedup else "-"

            # Get note from notes.md or verify.json
            note = ""
            if notes_path.exists():
                with open(notes_path) as f:
                    note = f.read().strip()[:NOTE_TRUNCATE_LENGTH]
            elif "note" in data:
                note = str(data["note"])[:NOTE_TRUNCATE_LENGTH]

            if passed and speedup and speedup > best_speedup:
                best_speedup = speedup
                best_version = v

            rows.append(f"| {v} | {passed_str} | {speedup_str} | {note} |")

    best_str = f"{best_version} ({best_speedup:.2f}x)" if best_version else "None"

    content = f"""# {operator} Optimization History

| Version | Passed | Speedup | Notes |
|---------|--------|---------|-------|
{chr(10).join(rows) if rows else "| - | - | - | No versions yet |"}

**Best**: {best_str}
**Target**: {target_speedup}x
**Calls**: {current_call} / {max_calls}
"""
    (workspace_dir / "PERFORMANCE.md").write_text(content)


def run_iteration_loop(
    workspace_dir: Path,
    operator: str,
    prompt_path: Path,
    dataset: str,
    gpu_id: int,
    config: dict,
    max_cc_calls: int,
    target_speedup: float,
    verify_timeout: int,
) -> tuple[str | None, str | None, float, int]:
    """Run the iterative optimization loop.

    Args:
        workspace_dir: Working directory for this operator
        operator: Operator name
        prompt_path: Path to operator context/prompt file
        dataset: Dataset name
        gpu_id: GPU ID for verification
        config: Configuration dict
        max_cc_calls: Maximum number of CC calls
        target_speedup: Target speedup to achieve
        verify_timeout: Timeout for verification in seconds

    Returns:
        Tuple of (best_code, best_version, best_speedup, total_cc_calls)
    """
    # Initialize workspace
    workspace_dir.mkdir(parents=True, exist_ok=True)

    # Copy context file
    context_dest = workspace_dir / "context.md"
    if not context_dest.exists():
        shutil.copy(prompt_path, context_dest)

    # Load optimize prompt
    optimize_prompt = load_optimize_prompt(gpu_id, operator, dataset)

    # Initialize tracking
    best_code = None
    best_speedup = 0.0
    best_version = None
    total_cc_calls = 0

    # Check for existing passing versions from previous runs
    existing_versions = list_versions(workspace_dir)
    for v in existing_versions:
        version_dir = workspace_dir / v
        kernel_path = version_dir / "kernel.py"
        existing_result = load_existing_verify_result(version_dir)

        if existing_result and existing_result.get("passed") and kernel_path.exists():
            speedup = existing_result.get("speedup") or 0.0
            # Accept passing version if: no best yet, OR this has better speedup
            if best_version is None or speedup > best_speedup:
                best_speedup = speedup
                best_code = kernel_path.read_text()
                best_version = v
                print(f"Found existing passing version: {v} (speedup: {speedup:.2f}x)")

    # If we already have a passing version meeting target, return early
    if best_version and best_speedup >= target_speedup:
        print(f"Existing version {best_version} already meets target speedup {target_speedup}x")
        return best_code, best_version, best_speedup, 0

    # Initialize PERFORMANCE.md
    update_performance_md(workspace_dir, operator, target_speedup, max_cc_calls, 0)

    for call_num in range(1, max_cc_calls + 1):
        print(f"\n=== CC Call {call_num}/{max_cc_calls} ===")
        total_cc_calls = call_num

        before = list_versions(workspace_dir)

        # Call CC
        success = call_cc(optimize_prompt, workspace_dir, config, str(call_num))
        if not success:
            print(f"CC call {call_num} failed")
            update_performance_md(workspace_dir, operator, target_speedup, max_cc_calls, call_num)
            continue

        # Detect new or modified version
        after = list_versions(workspace_dir)
        target_version = detect_new_version(before, after)

        if target_version:
            print(f"New version detected: {target_version}")
            target_version = target_version
        else:
            # No new version - check if CC modified an existing version
            target_version = get_latest_unverified_version(workspace_dir)
            if target_version:
                print(f"Modified version detected: {target_version}")
            else:
                print(f"No new or modified version detected after CC call {call_num}")
                update_performance_md(workspace_dir, operator, target_speedup, max_cc_calls, call_num)
                continue

        # Read kernel code
        kernel_path = workspace_dir / target_version / "kernel.py"
        if not kernel_path.exists():
            print(f"Kernel file not found: {kernel_path}")
            update_performance_md(workspace_dir, operator, target_speedup, max_cc_calls, call_num)
            continue

        code = kernel_path.read_text()

        # Check if CC already verified this version (verify.json exists)
        version_dir = workspace_dir / target_version
        existing_result = load_existing_verify_result(version_dir)

        if existing_result:
            print(f"Reusing existing verification result for {target_version}")
            result = existing_result
        else:
            # Verify kernel
            print(f"Verifying {target_version}...")
            result = verify_kernel(kernel_path, operator, dataset, gpu_id, verify_timeout)

            # Save verification result (verify_single.py already saves if --output-dir is used,
            # but we save again to ensure consistency)
            save_verify_result(version_dir, result)

        # Update PERFORMANCE.md
        update_performance_md(workspace_dir, operator, target_speedup, max_cc_calls, call_num)

        if result["passed"]:
            print(f"{target_version} passed verification")
            speedup = result.get("speedup") or 0.0

            if speedup > best_speedup:
                best_speedup = speedup
                best_code = code
                best_version = target_version
                print(f"New best: {best_version} with speedup {best_speedup:.2f}x")

            if best_speedup >= target_speedup:
                print(f"Target speedup {target_speedup}x achieved!")
                break
        else:
            error_display = (result.get("error") or "Unknown error")[:ERROR_DISPLAY_LENGTH]
            print(f"{target_version} failed verification: {error_display}")

    return best_code, best_version, best_speedup, total_cc_calls


def main():
    parser = argparse.ArgumentParser(description="Iterative optimizer worker")
    parser.add_argument("--workspace", required=True, help="Workspace directory")
    parser.add_argument("--operator", required=True, help="Operator name")
    parser.add_argument("--prompt-path", required=True, help="Path to operator prompt")
    parser.add_argument("--dataset", required=True, help="Dataset name")
    parser.add_argument("--gpu-id", type=int, required=True, help="GPU ID (must be >= 0)")
    parser.add_argument("--config-path", required=True, help="Path to config.yaml")
    parser.add_argument("--max-cc-calls", type=int, default=10, help="Maximum CC calls")
    parser.add_argument("--target-speedup", type=float, default=1.0, help="Target speedup")
    parser.add_argument("--verify-timeout", type=int, default=600, help="Verification timeout")

    args = parser.parse_args()

    # Validate gpu_id
    if args.gpu_id < 0:
        print(f"Error: GPU ID must be >= 0, got {args.gpu_id}")
        sys.exit(1)

    # Load config
    try:
        import yaml
        with open(args.config_path) as f:
            config = yaml.safe_load(f)
    except ImportError:
        print("Error: pyyaml required. Install: pip install pyyaml")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading config: {e}")
        sys.exit(1)

    workspace_dir = Path(args.workspace)
    prompt_path = Path(args.prompt_path)

    # Run iteration loop
    best_code, best_version, best_speedup, total_cc_calls = run_iteration_loop(
        workspace_dir=workspace_dir,
        operator=args.operator,
        prompt_path=prompt_path,
        dataset=args.dataset,
        gpu_id=args.gpu_id,
        config=config,
        max_cc_calls=args.max_cc_calls,
        target_speedup=args.target_speedup,
        verify_timeout=args.verify_timeout,
    )

    # Write final result
    result = {
        "best_code": best_code,
        "best_version": best_version,
        "best_speedup": best_speedup,
        "passed": best_version is not None,
        "cc_calls": total_cc_calls,
    }

    result_path = workspace_dir / "_result.json"
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"\n=== Final Result ===")
    print(f"Best version: {best_version}")
    if best_speedup:
        print(f"Best speedup: {best_speedup:.2f}x")
    else:
        print("No passing version")
    print(f"Total CC calls: {total_cc_calls}")
    print(f"Result saved to: {result_path}")

    # Exit with success if we have a passing version
    sys.exit(0 if best_version else 1)


if __name__ == "__main__":
    main()
