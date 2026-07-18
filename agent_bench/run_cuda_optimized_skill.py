#!/usr/bin/env python3

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

"""Run cuda-optimized-skill on KernelGenBench operators.

Takes baseline kernels and feeds them to cuda-optimized-skill for optimization
via Claude Code.

Usage:
    # Optimize from a directory of kernel files (e.g. pass@1 round_0/)
    python run_cuda_optimized_skill.py --baseline-dir /path/to/round_0

    # Optimize from a previous run's kernels/ directory
    python run_cuda_optimized_skill.py --baseline-run runs/normal_cc_xxx

    # Specific operators only
    python run_cuda_optimized_skill.py --baseline-dir /path/to/round_0 -k aten__add,aten__mul

    # Resume a previous run
    python run_cuda_optimized_skill.py --resume cuda_optimized_round_0_20260423_222602
"""

import argparse
import json
import logging
import os
import shutil
import signal
import subprocess
import sys
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

from device_manager import get_device_env_var

try:
    import yaml
except ImportError:
    yaml = None

from device_manager import DeviceManager, detect_device_type

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent
CUDA_OPTIMIZED_TEMPLATE = SCRIPT_DIR.parent / "sota_agents" / "cuda-optimized-skill"


def load_config(config_path: Path) -> dict:
    """Load YAML config file."""
    if yaml is None:
        print("Error: 'pyyaml' required. Install: pip install pyyaml")
        sys.exit(1)
    with open(config_path) as f:
        return yaml.safe_load(f)


def kill_process(proc: subprocess.Popen, stdout_file, stderr_file):
    """Kill a process and its process group."""
    if proc is None:
        return
    try:
        pgid = os.getpgid(proc.pid)
        os.killpg(pgid, signal.SIGKILL)
    except (ProcessLookupError, OSError):
        try:
            proc.kill()
        except OSError:
            pass
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        logger.warning(f"Process {proc.pid} did not exit after SIGKILL")

    for fh in (stdout_file, stderr_file):
        if fh:
            try:
                if not fh.closed:
                    fh.close()
            except Exception:
                pass


class Progress:
    """Manages progress.json with real-time updates."""

    def __init__(self, path: Path):
        self.path = path
        self.data = {
            "start_time": datetime.now(timezone.utc).isoformat(),
            "end_time": None,
            "agent": "cuda-optimized-skill",
            "summary": {
                "total": 0,
                "completed": 0,
                "running": 0,
                "failed": 0,
                "pending": 0,
            },
            "kernels": {},
        }
        self._save()

    def add_kernel(self, kernel_name: str, gpu_id: int, attempt: int):
        self.data["kernels"][kernel_name] = {
            "status": "running",
            "gpu_id": gpu_id,
            "attempt": attempt,
            "start_time": datetime.now(timezone.utc).isoformat(),
            "end_time": None,
            "duration_seconds": None,
            "error": None,
        }
        self._recount()
        self._save()

    def update_kernel(self, kernel_name: str, **kwargs):
        if kernel_name in self.data["kernels"]:
            self.data["kernels"][kernel_name].update(kwargs)
            self._recount()
            self._save()

    def finalize(self):
        self.data["end_time"] = datetime.now(timezone.utc).isoformat()
        self._save()

    def _recount(self):
        kernels = self.data["kernels"]
        self.data["summary"]["total"] = len(kernels)
        self.data["summary"]["completed"] = sum(
            1 for v in kernels.values() if v["status"] == "completed"
        )
        self.data["summary"]["failed"] = sum(
            1 for v in kernels.values() if v["status"] in ("failed", "timeout")
        )
        self.data["summary"]["running"] = sum(
            1 for v in kernels.values() if v["status"] in ("running", "retrying")
        )
        self.data["summary"]["pending"] = (
            self.data["summary"]["total"]
            - self.data["summary"]["completed"]
            - self.data["summary"]["failed"]
            - self.data["summary"]["running"]
        )

    def _save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w") as f:
            json.dump(self.data, f, indent=2, ensure_ascii=False)


def setup_cuda_optimized_workspace(
    workspace_dir: Path,
    kernel_path: Path,
    op_name: str,
    dataset: str,
    max_iterations: int = 5,
    attempt: int = 0,
    prompts_dir: Path | None = None,
) -> None:
    """Set up a cuda-optimized-skill workspace for one operator.

    Copies the skills directory from the template and sets up the workspace
    with the baseline kernel for optimization.
    """
    if workspace_dir.exists() and attempt == 0:
        shutil.rmtree(workspace_dir)
    workspace_dir.mkdir(parents=True, exist_ok=True)

    # Copy skills/ from template
    skills_src = CUDA_OPTIMIZED_TEMPLATE / "skills"
    skills_dst = workspace_dir / "skills"
    if skills_src.exists():
        shutil.copytree(skills_src, skills_dst, dirs_exist_ok=True)

    # Copy baseline kernel as kernel.py and kernel_original.py
    kernel_dst = workspace_dir / "kernel.py"
    shutil.copy2(kernel_path, kernel_dst)
    shutil.copy2(kernel_path, workspace_dir / "kernel_original.py")

    # Copy operator spec from prompts directory
    if prompts_dir is not None:
        safe_name = op_name.replace("::", "__")
        prompt_file = prompts_dir / f"{safe_name}.md"
        if not prompt_file.exists():
            # Try original op_name as filename
            prompt_file = prompts_dir / f"{op_name}.md"
        if prompt_file.exists():
            shutil.copy2(prompt_file, workspace_dir / "operator_spec.md")

    # Convert op_name format for display
    display_name = op_name
    if "__" in op_name and "::" not in op_name:
        ns, name = op_name.split("__", 1)
        display_name = f"{ns}::{name}"

    # No CLAUDE.md - all instructions are in SKILL.md


def _normalize_kernel_name(filename: str) -> str:
    """Normalize kernel filename to internal name format."""
    stem = Path(filename).stem
    if "::" in stem:
        stem = stem.replace("::", "__")
    return stem


def _collect_baseline_kernels(baseline_dir: Path) -> list[tuple[str, Path]]:
    """Collect kernel files from a directory, normalizing names."""
    results = []
    for f in sorted(baseline_dir.glob("*.py")):
        name = _normalize_kernel_name(f.name)
        results.append((name, f))
    return results


def build_agent_prompt(op_name: str, dataset: str, max_iterations: int = 5) -> str:
    """Build the prompt to send to the Claude agent."""
    display_name = op_name
    if "__" in op_name and "::" not in op_name:
        ns, name = op_name.split("__", 1)
        display_name = f"{ns}::{name}"

    return (
        f"Optimize the Triton kernel in kernel.py for operator `{display_name}`.\n"
        f"Read skills/optimized-skill/operator-optimize-loop/SKILL.md for:\n"
        f"  - Pre-requisites: kernel.py must have complete PyTorch wrappers + setup()/run_kernel()\n"
        f"  - Optimization workflow and strategy\n"
        f"Read operator_spec.md for the operator specification.\n"
        f"Use --max-iterations={max_iterations} --backend=triton\n"
        f"Do NOT modify files in skills/. Only modify kernel.py.\n"
        f"Ensure correctness is maintained after each optimization."
    )


def run(args):
    """Main run loop."""
    if yaml is None:
        print("Error: 'pyyaml' required. Install: pip install pyyaml")
        sys.exit(1)

    queue = deque()
    max_retries = args.max_retries

    # Resolve baseline kernel directory
    baseline_dir = None
    baseline_label = None

    if args.baseline_dir:
        baseline_dir = Path(args.baseline_dir)
        if not baseline_dir.is_absolute():
            baseline_dir = Path.cwd() / baseline_dir
        if not baseline_dir.exists():
            print(f"Error: Baseline directory not found: {baseline_dir}")
            sys.exit(1)
        baseline_label = baseline_dir.name
    elif args.baseline_run:
        baseline_run = Path(args.baseline_run)
        if not baseline_run.is_absolute():
            baseline_run = SCRIPT_DIR / baseline_run
        baseline_dir = baseline_run / "kernels"
        if not baseline_dir.exists():
            print(f"Error: No kernels directory in baseline run: {baseline_dir}")
            sys.exit(1)
        baseline_label = baseline_run.name
    else:
        print("Error: --baseline-run or --baseline-dir is required")
        sys.exit(1)

    all_kernels = _collect_baseline_kernels(baseline_dir)
    if not all_kernels:
        print(f"Error: No kernel files found in {baseline_dir}")
        sys.exit(1)

    logger.info(f"Found {len(all_kernels)} kernels in {baseline_dir}")

    if args.kernels:
        filter_names = set(args.kernels.split(","))
        all_kernels = [
            (name, path) for name, path in all_kernels
            if name in filter_names
            or name.split("__", 1)[-1] in filter_names
        ]
        logger.info(f"Filtered to {len(all_kernels)} kernels")

    for kernel_name, kernel_path in all_kernels:
        queue.append((kernel_name, kernel_path, 0))

    run_name = f"cuda_optimized_{baseline_label}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Handle resume
    if args.resume:
        resume_dir = SCRIPT_DIR / "runs" / args.resume
        if not resume_dir.exists():
            print(f"Error: Resume run not found: {resume_dir}")
            sys.exit(1)
        run_name = args.resume

    # Create run directory
    run_dir = SCRIPT_DIR / "runs" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    workspaces_dir = run_dir / "workspaces_cuda_optimized"
    workspaces_dir.mkdir(exist_ok=True)

    kernels_dir = run_dir / "kernels"
    kernels_dir.mkdir(exist_ok=True)

    kernels_optimized_dir = run_dir / "kernels_optimized"
    kernels_optimized_dir.mkdir(exist_ok=True)

    # Skip existing kernels when resuming
    if args.resume:
        force_rerun = set()
        if args.kernels:
            force_rerun = set(args.kernels.split(","))

        existing_kernels = set()
        for f in kernels_dir.glob("*.py"):
            if f.stem not in force_rerun:
                existing_kernels.add(f.stem)

        before = len(queue)
        queue = deque((k, p, a) for k, p, a in queue if k not in existing_kernels)
        logger.info(f"Resume: skipping {before - len(queue)} existing kernels, {len(queue)} remaining")

    # Save config
    config = {
        "agent": "cuda-optimized-skill",
        "mode": "optimize",
        "dataset": args.dataset,
        "baseline_run": str(args.baseline_run) if args.baseline_run else None,
        "baseline_dir": str(args.baseline_dir) if args.baseline_dir else None,
        "timeout": args.timeout,
        "claude_bin": args.claude_bin,
        "budget": args.budget,
        "max_iterations": args.max_iterations,
        "max_retries": args.max_retries,
    }
    with open(run_dir / "config.yaml", "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    # Initialize device manager
    device_count = args.device_count if hasattr(args, "device_count") and args.device_count else 8
    device_mgr = DeviceManager(
        lock_dir="/tmp/cuda_optimized_gpu_locks",
        gpu_ids=list(range(device_count)),
    )

    progress = Progress(run_dir / "progress.json")

    # Running tasks: kernel_name -> (proc, stdout_file, stderr_file, gpu_id, start_time, workspace_dir, attempt, kernel_path)
    running = {}
    last_snapshot_mtime = {}

    # Resolve prompts directory for operator specs
    prompts_dir = SCRIPT_DIR / "prompts" / args.dataset
    if not prompts_dir.exists():
        # KernelGenBench sub-datasets share the same prompts directory
        prompts_dir = SCRIPT_DIR / "prompts" / "KernelGenBench"
    if not prompts_dir.exists():
        logger.warning(f"Prompts directory not found: {prompts_dir}, operator_spec.md will not be generated")
        prompts_dir = None

    logger.info(f"Run: {run_name}")
    logger.info(f"Dataset: {args.dataset}, Queue: {len(queue)}")

    while queue or running:
        # Launch new tasks
        while queue and device_mgr.available_count() > 0:
            kernel_name, kernel_path, attempt = queue.popleft()

            gpu_id = device_mgr.acquire()
            if gpu_id is None:
                queue.appendleft((kernel_name, kernel_path, attempt))
                break

            try:
                workspace_dir = workspaces_dir / kernel_name
                setup_cuda_optimized_workspace(
                    workspace_dir=workspace_dir,
                    kernel_path=kernel_path,
                    op_name=kernel_name,
                    dataset=args.dataset,
                    max_iterations=args.max_iterations,
                    attempt=attempt,
                    prompts_dir=prompts_dir,
                )

                prompt = build_agent_prompt(
                    op_name=kernel_name,
                    dataset=args.dataset,
                    max_iterations=args.max_iterations,
                )

                # Build claude command
                claude_bin = args.claude_bin or "claude"
                cmd = [
                    claude_bin,
                    "-p", prompt,
                    "--dangerously-skip-permissions",
                    "--output-format", "stream-json",
                    "--verbose",
                ]

                if args.budget:
                    cmd.extend(["--max-cost", str(args.budget)])

                env = os.environ.copy()
                env[get_device_env_var()] = str(gpu_id)
                env["PROJECT_ROOT"] = str(SCRIPT_DIR.parent)
                env["IS_SANDBOX"] = "1"

                stdout_file = open(workspace_dir / "cuda_optimized_output.jsonl", "w")
                stderr_file = open(workspace_dir / "cuda_optimized_stderr.log", "w")

                proc = subprocess.Popen(
                    cmd,
                    cwd=str(workspace_dir),
                    env=env,
                    stdin=subprocess.DEVNULL,
                    stdout=stdout_file,
                    stderr=stderr_file,
                    start_new_session=True,
                )

                running[kernel_name] = (
                    proc, stdout_file, stderr_file, gpu_id,
                    time.time(), workspace_dir, attempt, kernel_path,
                )
                progress.add_kernel(kernel_name, gpu_id, attempt)

                retry_str = f" (retry {attempt})" if attempt > 0 else ""
                logger.info(f"Launched cuda-optimized-skill for {kernel_name} (GPU={gpu_id}){retry_str}")

            except Exception as e:
                logger.error(f"Failed to launch cuda-optimized-skill for {kernel_name}: {e}")
                device_mgr.release(gpu_id)
                if attempt + 1 < max_retries:
                    queue.append((kernel_name, kernel_path, attempt + 1))

        # Check running tasks
        for kernel_name in list(running.keys()):
            proc, stdout_file, stderr_file, gpu_id, start_time, workspace_dir, attempt, kernel_path = running[
                kernel_name
            ]
            elapsed = time.time() - start_time

            # Incremental snapshot: save kernel.py whenever it's updated
            solution_path = workspace_dir / "kernel.py"
            if solution_path.exists():
                try:
                    mtime = solution_path.stat().st_mtime
                    if mtime != last_snapshot_mtime.get(kernel_name):
                        dest = kernels_dir / f"{kernel_name}.py"
                        shutil.copy2(solution_path, dest)
                        last_snapshot_mtime[kernel_name] = mtime
                        logger.info(f"[SNAPSHOT] {kernel_name} - saved latest kernel ({elapsed:.0f}s)")
                except Exception:
                    pass

            # Check timeout
            if elapsed > args.timeout:
                logger.warning(f"Timeout for {kernel_name} after {elapsed:.0f}s")
                kill_process(proc, stdout_file, stderr_file)
                device_mgr.release(gpu_id)
                del running[kernel_name]

                # Save final kernel if exists
                if solution_path.exists():
                    dest = kernels_dir / f"{kernel_name}.py"
                    shutil.copy2(solution_path, dest)
                    progress.update_kernel(
                        kernel_name,
                        status="completed",
                        end_time=datetime.now(timezone.utc).isoformat(),
                        duration_seconds=elapsed,
                        error="timeout but kernel saved",
                    )
                    logger.info(f"Saved kernel for {kernel_name} (timeout)")
                else:
                    progress.update_kernel(
                        kernel_name,
                        status="timeout",
                        end_time=datetime.now(timezone.utc).isoformat(),
                        duration_seconds=elapsed,
                        error="timeout, no kernel",
                    )
                continue

            # Check if process finished
            exit_code = proc.poll()
            if exit_code is not None:
                device_mgr.release(gpu_id)
                del running[kernel_name]

                # Close file handles
                for fh in (stdout_file, stderr_file):
                    if fh and not fh.closed:
                        try:
                            fh.close()
                        except Exception:
                            pass

                if solution_path.exists():
                    dest = kernels_dir / f"{kernel_name}.py"
                    shutil.copy2(solution_path, dest)
                    status = "completed"
                    error = None if exit_code == 0 else f"exit code {exit_code}"
                    logger.info(f"Completed {kernel_name} (exit={exit_code}, {elapsed:.0f}s)")
                else:
                    status = "failed"
                    error = f"exit code {exit_code}, no kernel"
                    logger.warning(f"Failed {kernel_name} (exit={exit_code}, no kernel)")

                progress.update_kernel(
                    kernel_name,
                    status=status,
                    end_time=datetime.now(timezone.utc).isoformat(),
                    duration_seconds=elapsed,
                    error=error,
                )

                if status == "failed" and attempt + 1 < max_retries:
                    queue.append((kernel_name, kernel_path, attempt + 1))

        # Sleep before next poll
        if running:
            time.sleep(10)

    progress.finalize()
    logger.info(f"All done. Results in: {run_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Run cuda-optimized-skill on KernelGenBench operators",
    )
    parser.add_argument(
        "-d", "--dataset",
        type=str,
        default="KernelGenBench",
        help="Dataset name (default: KernelGenBench)",
    )
    parser.add_argument(
        "-b", "--baseline-run",
        type=str,
        default=None,
        help="Baseline run directory with kernels/ subdir",
    )
    parser.add_argument(
        "--baseline-dir",
        type=str,
        default=None,
        help="Direct path to directory of baseline kernel .py files",
    )
    parser.add_argument(
        "-k", "--kernels",
        type=str,
        default=None,
        help="Comma-separated kernel names to process (default: all)",
    )
    parser.add_argument(
        "-t", "--timeout",
        type=int,
        default=1800,
        help="Timeout per kernel in seconds (default: 1800)",
    )
    parser.add_argument(
        "--device-count",
        type=int,
        default=8,
        help="Number of GPUs to use (default: 8)",
    )
    parser.add_argument(
        "--claude-bin",
        type=str,
        default=None,
        help="Path to claude binary (default: claude)",
    )
    parser.add_argument(
        "--budget",
        type=float,
        default=None,
        help="Budget limit per kernel in USD (default: no limit)",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=1,
        help="Max attempts per kernel (default: 1)",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=5,
        help="Max optimization iterations per kernel (default: 5)",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Resume a previous run by name (skips existing kernels)",
    )

    args = parser.parse_args()

    if not args.baseline_run and not args.baseline_dir and not args.resume:
        print("Error: --baseline-run or --baseline-dir is required")
        sys.exit(1)

    if not CUDA_OPTIMIZED_TEMPLATE.exists():
        print(f"Error: cuda-optimized-skill template not found: {CUDA_OPTIMIZED_TEMPLATE}")
        sys.exit(1)

    run(args)


if __name__ == "__main__":
    main()
