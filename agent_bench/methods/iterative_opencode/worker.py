#!/usr/bin/env python3
"""Worker script for iterative OpenCode optimizer method.

Based on iterative_optimizer/worker.py but uses OpenCode CLI instead of Claude Code.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ERROR_DISPLAY_LENGTH = 100
NOTE_TRUNCATE_LENGTH = 50

SCRIPT_DIR = Path(__file__).resolve().parent
AGENT_BENCH_DIR = SCRIPT_DIR.parent.parent
TOOLS_DIR = AGENT_BENCH_DIR / "tools"
TEMPLATES_DIR = SCRIPT_DIR / "templates"


def list_versions(workspace_dir: Path) -> list[str]:
    versions = []
    if not workspace_dir.exists():
        return versions
    for d in workspace_dir.iterdir():
        if d.is_dir() and d.name.startswith("v") and d.name[1:].isdigit():
            versions.append(d.name)
    return sorted(versions, key=lambda x: int(x[1:]))


def detect_new_version(before: list[str], after: list[str]) -> str | None:
    new = set(after) - set(before)
    if new:
        return sorted(new, key=lambda x: int(x[1:]))[-1]
    return None


def load_optimize_prompt(gpu_id: int, operator: str, dataset: str) -> str:
    template_path = TEMPLATES_DIR / "optimize.md"
    with open(template_path) as f:
        prompt = f.read()
    prompt = prompt.replace("{{GPU_ID}}", str(gpu_id))
    prompt = prompt.replace("{{OP_NAME}}", operator)
    prompt = prompt.replace("{{DATASET}}", dataset)
    prompt = prompt.replace("{{TOOLS_DIR}}", str(TOOLS_DIR))
    return prompt


def call_opencode(prompt: str, working_dir: Path, config: dict, output_suffix: str = "") -> bool:
    """Call OpenCode with the given prompt."""
    suffix = f"_{output_suffix}" if output_suffix else ""
    base_output = working_dir / f"oc_output{suffix}.json"
    base_log = working_dir / f"oc{suffix}.log"

    output_path = base_output
    log_path = base_log
    counter = 1
    while output_path.exists() or log_path.exists():
        output_path = working_dir / f"oc_output{suffix}_{counter}.json"
        log_path = working_dir / f"oc{suffix}_{counter}.log"
        counter += 1

    env = os.environ.copy()
    env["IS_SANDBOX"] = "1"

    agent_config = config.get("agent", {})
    opencode_bin = agent_config.get("opencode_bin", "opencode")
    opencode_model = agent_config.get("opencode_model")
    oc_timeout = agent_config.get("oc_timeout", 1800)

    cmd = [
        opencode_bin,
        "run", prompt,
        "--format", "json",
        "--dir", str(working_dir),
    ]

    if opencode_model:
        cmd.extend(["--model", opencode_model])

    try:
        with open(output_path, "w") as stdout_f, open(log_path, "w") as stderr_f:
            result = subprocess.run(
                cmd,
                cwd=str(working_dir),
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=stdout_f,
                stderr=stderr_f,
                timeout=oc_timeout,
            )
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        print(f"OpenCode call timed out after {oc_timeout}s")
        return False
    except Exception as e:
        print(f"OpenCode call failed: {e}")
        return False


def verify_kernel(
    kernel_path: Path,
    operator: str,
    dataset: str,
    gpu_id: int,
    timeout: int = 600,
) -> dict:
    if not kernel_path.exists():
        return {"passed": False, "error": f"Kernel file not found: {kernel_path}",
                "total_tests": 0, "passed_tests": 0, "failed_tests": 0}

    if not isinstance(gpu_id, int) or gpu_id < 0:
        return {"passed": False, "error": f"Invalid GPU ID: {gpu_id}",
                "total_tests": 0, "passed_tests": 0, "failed_tests": 0}

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env["DISPATCH_TORCH_LIB"] = "1"
    env["FLAGBENCH_SKIP_BOTH_TEST"] = "1"

    verify_script = TOOLS_DIR / "verify_single.py"
    output_dir = kernel_path.parent

    cmd = [
        sys.executable, str(verify_script),
        "--code", str(kernel_path),
        "--op", operator,
        "--dataset", dataset,
        "--output-dir", str(output_dir),
        "--output-json",
    ]

    try:
        result = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=timeout)
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
            return {"passed": False, "error": f"Failed to parse verification output: {result.stdout[:200]}",
                    "total_tests": 0, "passed_tests": 0, "failed_tests": 0}
    except subprocess.TimeoutExpired:
        return {"passed": False, "error": f"Verification timed out after {timeout}s",
                "total_tests": 0, "passed_tests": 0, "failed_tests": 0}
    except Exception as e:
        return {"passed": False, "error": str(e),
                "total_tests": 0, "passed_tests": 0, "failed_tests": 0}


def get_latest_unverified_version(workspace_dir: Path) -> str | None:
    versions = list_versions(workspace_dir)
    if not versions:
        return None
    for v in reversed(versions):
        version_dir = workspace_dir / v
        kernel_path = version_dir / "kernel.py"
        verify_path = version_dir / "verify.json"
        if not kernel_path.exists():
            continue
        if not verify_path.exists():
            return v
        if kernel_path.stat().st_mtime > verify_path.stat().st_mtime:
            return v
    return None


def load_existing_verify_result(version_dir: Path) -> dict | None:
    verify_path = version_dir / "verify.json"
    if not verify_path.exists():
        return None
    try:
        with open(verify_path) as f:
            result = json.load(f)
        required_fields = ["passed", "total_tests", "passed_tests", "failed_tests"]
        if all(field in result for field in required_fields):
            return result
        return None
    except (json.JSONDecodeError, IOError):
        return None


def save_verify_result(version_dir: Path, result: dict):
    verify_path = version_dir / "verify.json"
    with open(verify_path, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)


def update_performance_md(
    workspace_dir: Path, operator: str, target_speedup: float,
    max_calls: int, current_call: int,
):
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
    workspace_dir: Path, operator: str, prompt_path: Path,
    dataset: str, gpu_id: int, config: dict,
    max_oc_calls: int, target_speedup: float, verify_timeout: int,
) -> tuple[str | None, str | None, float, int]:
    workspace_dir.mkdir(parents=True, exist_ok=True)

    context_dest = workspace_dir / "context.md"
    if not context_dest.exists():
        shutil.copy(prompt_path, context_dest)

    optimize_prompt = load_optimize_prompt(gpu_id, operator, dataset)

    best_code = None
    best_speedup = 0.0
    best_version = None
    total_oc_calls = 0

    existing_versions = list_versions(workspace_dir)
    for v in existing_versions:
        version_dir = workspace_dir / v
        kernel_path = version_dir / "kernel.py"
        existing_result = load_existing_verify_result(version_dir)
        if existing_result and existing_result.get("passed") and kernel_path.exists():
            speedup = existing_result.get("speedup") or 0.0
            if best_version is None or speedup > best_speedup:
                best_speedup = speedup
                best_code = kernel_path.read_text()
                best_version = v
                print(f"Found existing passing version: {v} (speedup: {speedup:.2f}x)")

    if best_version and best_speedup >= target_speedup:
        print(f"Existing version {best_version} already meets target speedup {target_speedup}x")
        return best_code, best_version, best_speedup, 0

    update_performance_md(workspace_dir, operator, target_speedup, max_oc_calls, 0)

    for call_num in range(1, max_oc_calls + 1):
        print(f"\n=== OpenCode Call {call_num}/{max_oc_calls} ===")
        total_oc_calls = call_num

        before = list_versions(workspace_dir)

        success = call_opencode(optimize_prompt, workspace_dir, config, str(call_num))
        if not success:
            print(f"OpenCode call {call_num} failed")
            update_performance_md(workspace_dir, operator, target_speedup, max_oc_calls, call_num)
            continue

        after = list_versions(workspace_dir)
        target_version = detect_new_version(before, after)

        if target_version:
            print(f"New version detected: {target_version}")
        else:
            target_version = get_latest_unverified_version(workspace_dir)
            if target_version:
                print(f"Modified version detected: {target_version}")
            else:
                print(f"No new or modified version detected after OpenCode call {call_num}")
                update_performance_md(workspace_dir, operator, target_speedup, max_oc_calls, call_num)
                continue

        kernel_path = workspace_dir / target_version / "kernel.py"
        if not kernel_path.exists():
            print(f"Kernel file not found: {kernel_path}")
            update_performance_md(workspace_dir, operator, target_speedup, max_oc_calls, call_num)
            continue

        code = kernel_path.read_text()

        version_dir = workspace_dir / target_version
        existing_result = load_existing_verify_result(version_dir)

        if existing_result:
            print(f"Reusing existing verification result for {target_version}")
            result = existing_result
        else:
            print(f"Verifying {target_version}...")
            result = verify_kernel(kernel_path, operator, dataset, gpu_id, verify_timeout)
            save_verify_result(version_dir, result)

        update_performance_md(workspace_dir, operator, target_speedup, max_oc_calls, call_num)

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

    return best_code, best_version, best_speedup, total_oc_calls


def main():
    parser = argparse.ArgumentParser(description="Iterative OpenCode optimizer worker")
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--operator", required=True)
    parser.add_argument("--prompt-path", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--gpu-id", type=int, required=True)
    parser.add_argument("--config-path", required=True)
    parser.add_argument("--max-oc-calls", type=int, default=10)
    parser.add_argument("--target-speedup", type=float, default=1.0)
    parser.add_argument("--verify-timeout", type=int, default=600)

    args = parser.parse_args()

    if args.gpu_id < 0:
        print(f"Error: GPU ID must be >= 0, got {args.gpu_id}")
        sys.exit(1)

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

    best_code, best_version, best_speedup, total_oc_calls = run_iteration_loop(
        workspace_dir=workspace_dir,
        operator=args.operator,
        prompt_path=prompt_path,
        dataset=args.dataset,
        gpu_id=args.gpu_id,
        config=config,
        max_oc_calls=args.max_oc_calls,
        target_speedup=args.target_speedup,
        verify_timeout=args.verify_timeout,
    )

    result = {
        "best_code": best_code,
        "best_version": best_version,
        "best_speedup": best_speedup,
        "passed": best_version is not None,
        "oc_calls": total_oc_calls,
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
    print(f"Total OpenCode calls: {total_oc_calls}")
    print(f"Result saved to: {result_path}")

    sys.exit(0 if best_version else 1)


if __name__ == "__main__":
    main()
