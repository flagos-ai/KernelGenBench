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

"""Run AKO4ALL optimization on kernels from a baseline run.

Takes kernels from a normalcc/normal_cc baseline run and feeds them to
the original AKO4ALL pipeline for iterative optimization.

Usage:
    python run_ako4all.py --baseline-run runs/normal_cc_v2_1_20260318_160435 --iterations 30
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

try:
    import yaml
except ImportError:
    yaml = None

from device_manager import (
    DeviceManager,
    detect_device_type,
    parse_device_count,
    parse_device_ids,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent
AKO4ALL_TEMPLATE = SCRIPT_DIR.parent / "sota_agents" / "AKO4ALL"


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

    # Close file handles
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
        """Record kernel optimization started."""
        self.data["kernels"][kernel_name] = {
            "status": "running",
            "gpu_id": gpu_id,
            "attempt": attempt,
            "start_time": datetime.now(timezone.utc).isoformat(),
            "end_time": None,
            "duration_seconds": None,
            "optimized": False,
            "error": None,
        }
        self._recount()
        self._save()

    def update_kernel(self, kernel_name: str, **kwargs):
        """Update kernel fields."""
        if kernel_name in self.data["kernels"]:
            self.data["kernels"][kernel_name].update(kwargs)
            self._recount()
            self._save()

    def finalize(self):
        """Mark run as complete."""
        self.data["end_time"] = datetime.now(timezone.utc).isoformat()
        self._save()

    def _recount(self):
        """Recount summary statistics."""
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
        """Write progress to disk."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w") as f:
            json.dump(self.data, f, indent=2, ensure_ascii=False)


def setup_ako4all_workspace(
    workspace_dir: Path,
    kernel_path: Path | None,
    iterations: int,
    op_name: str,
    dataset: str,
    attempt: int = 0,
) -> None:
    """Set up AKO4ALL workspace for a kernel.

    Copies the entire AKO4ALL directory as-is, then:
    - Optimization mode (kernel_path provided): places kernel into input/kernel.py
    - Generation mode (kernel_path is None): leaves input/ empty, copies operator
      prompt to context/operator_spec.md

    On retry (attempt > 0), preserves existing workspace (and output files) but
    resets solution/ directory for a fresh start.

    Args:
        workspace_dir: Target workspace directory
        kernel_path: Path to baseline kernel file, or None for generation mode
        iterations: Number of optimization iterations
        op_name: Operator name (e.g., aten__add)
        dataset: Dataset name (e.g., KernelGenBench)
        attempt: Attempt number (0 = first run)
    """
    if attempt > 0 and workspace_dir.exists():
        # Retry: preserve workspace (and output files), just reset solution/
        solution_dir = workspace_dir / "solution"
        if solution_dir.exists():
            shutil.rmtree(solution_dir)
        solution_dir.mkdir(exist_ok=True)
        (solution_dir / ".gitkeep").touch()
        return

    # First attempt: copy entire AKO4ALL directory as workspace
    if workspace_dir.exists():
        shutil.rmtree(workspace_dir)
    shutil.copytree(AKO4ALL_TEMPLATE, workspace_dir)

    if kernel_path is not None:
        # Optimization mode: place baseline kernel into input/
        input_dir = workspace_dir / "input"
        input_dir.mkdir(exist_ok=True)
        # Remove .gitkeep if present
        gitkeep = input_dir / ".gitkeep"
        if gitkeep.exists():
            gitkeep.unlink()
        shutil.copy2(kernel_path, input_dir / "kernel.py")
    else:
        # Generation mode: copy operator prompt to context/
        context_dir = workspace_dir / "context"
        context_dir.mkdir(exist_ok=True)
        prompts_ds = "KernelGenBench" if dataset.startswith("KernelGenBench") else dataset
        prompt_path = SCRIPT_DIR / "prompts" / prompts_ds / f"{op_name}.md"
        if prompt_path.exists():
            shutil.copy2(prompt_path, context_dir / "operator_spec.md")
        else:
            logger.warning(f"Prompt not found: {prompt_path}")

    # Set PROJECT_ROOT so bench.py can find verify_single.py
    env_file = workspace_dir / ".env"
    env_file.write_text(f"PROJECT_ROOT={SCRIPT_DIR.parent}\n")

    # Override any parent CLAUDE.md that may block file modifications
    claude_md = workspace_dir / "CLAUDE.md"
    claude_md.write_text(
        "# CLAUDE.md\n\n"
        "You have full permission to create, modify, and delete any files in this workspace. "
        "Start working immediately — no confirmation needed.\n"
    )

    # Remove HINTS.md for non-NVIDIA chips (hints are NVIDIA-specific)
    device_type = detect_device_type()
    if device_type != "cuda":
        hints_path = workspace_dir / "HINTS.md"
        if hints_path.exists():
            hints_path.unlink()
            logger.info(f"Removed HINTS.md for {device_type} device")

    # AKO4ALL TASK.md requires git (create branch, commit per iteration)
    subprocess.run(["git", "init"], cwd=workspace_dir, capture_output=True)
    subprocess.run(["git", "add", "-A"], cwd=workspace_dir, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "initial: AKO4ALL workspace"],
        cwd=workspace_dir, capture_output=True,
        env={**os.environ, "GIT_AUTHOR_NAME": "ako4all", "GIT_AUTHOR_EMAIL": "ako@bench",
             "GIT_COMMITTER_NAME": "ako4all", "GIT_COMMITTER_EMAIL": "ako@bench"},
    )


def launch_ako4all(
    workspace_dir: Path,
    gpu_id: int,
    iterations: int,
    op_name: str,
    dataset: str,
    attempt: int = 0,
    generate: bool = False,
    claude_bin: str = "claude",
    budget: float = None,
) -> tuple[subprocess.Popen, object, object]:
    """Launch AKO4ALL optimization process.

    Args:
        workspace_dir: AKO4ALL workspace directory
        gpu_id: GPU device ID
        iterations: Number of optimization iterations
        op_name: Operator name
        dataset: Dataset name
        generate: Whether this is generation mode (no baseline kernel)
        claude_bin: Path to claude binary
        budget: Optional budget limit in USD

    Returns:
        (process, stdout_file, stderr_file)
    """
    # Build prompt
    if generate:
        prompt = (
            f"Follow the instructions in TASK.md. Save HINTS.md to memory. "
            f"The operator is {op_name}, dataset is {dataset}. "
            f"Generate and optimize for up to {iterations} iterations. "
            f"Stop early only if all viable approaches are exhausted."
        )
    else:
        prompt = (
            f"Follow the instructions in TASK.md. Save HINTS.md to memory. "
            f"The operator is {op_name}, dataset is {dataset}. "
            f"Optimize for up to {iterations} iterations. "
            f"Stop early only if all viable approaches are exhausted."
        )

    # Prepare output paths (with attempt suffix to preserve previous attempts)
    suffix = f"_attempt{attempt}" if attempt > 0 else ""
    stdout_path = workspace_dir / f"ako4all_output{suffix}.jsonl"
    stderr_path = workspace_dir / f"ako4all{suffix}.log"

    # Environment
    env = os.environ.copy()
    env.pop("CLAUDECODE", None)
    env["IS_SANDBOX"] = "1"
    env["PROJECT_ROOT"] = str(SCRIPT_DIR.parent)

    from device_manager import get_device_env_var
    env[get_device_env_var()] = str(gpu_id)

    # Build command
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

    return proc, stdout_file, stderr_file


def collect_optimized_kernel(workspace_dir: Path) -> str | None:
    """Collect optimized kernel from AKO4ALL workspace.

    Args:
        workspace_dir: AKO4ALL workspace directory

    Returns:
        Optimized kernel code, or None if not found
    """
    solution_path = workspace_dir / "solution" / "kernel.py"
    if solution_path.exists():
        return solution_path.read_text()
    return None


def run(args):
    """Main orchestration loop."""
    # Build task queue based on mode
    queue = deque()

    if args.mode == "optimize":
        # Resolve baseline run path
        baseline_run = Path(args.baseline_run)
        if not baseline_run.is_absolute():
            baseline_run = SCRIPT_DIR / "runs" / args.baseline_run

        if not baseline_run.exists():
            print(f"Error: Baseline run not found: {baseline_run}")
            sys.exit(1)

        # Find baseline kernels
        baseline_kernels_dir = baseline_run / "kernels"
        if not baseline_kernels_dir.exists():
            print(f"Error: No kernels directory in baseline run: {baseline_kernels_dir}")
            sys.exit(1)

        kernel_files = sorted(baseline_kernels_dir.glob("*.py"))
        if not kernel_files:
            print(f"Error: No kernel files found in {baseline_kernels_dir}")
            sys.exit(1)

        logger.info(f"Found {len(kernel_files)} kernels in baseline run")

        # Filter kernels if specified
        if args.kernels:
            filter_names = set(args.kernels.split(","))
            kernel_files = [
                kf for kf in kernel_files
                if kf.stem in filter_names or kf.stem.split("__")[-1] in filter_names
            ]
            logger.info(f"Filtered to {len(kernel_files)} kernels")

        # Build queue: (kernel_name, kernel_path, attempt)
        for kernel_file in kernel_files:
            queue.append((kernel_file.stem, kernel_file, 0))

        run_name = f"ako4all_{baseline_run.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    else:  # generate mode
        # Find prompts (sub-datasets share KernelGenBench prompts directory)
        prompts_dataset = "KernelGenBench" if args.dataset.startswith("KernelGenBench") else args.dataset
        prompts_dir = SCRIPT_DIR / "prompts" / prompts_dataset
        if not prompts_dir.exists():
            print(f"Error: Prompts directory not found: {prompts_dir}")
            sys.exit(1)

        prompt_files = sorted(prompts_dir.glob("*.md"))

        # Filter by sub-dataset prefix (e.g., KernelGenBench-aten -> aten__*)
        if args.dataset.startswith("KernelGenBench-"):
            sub = args.dataset.split("-", 1)[1]
            prefix_map = {"aten": "aten__", "cublas": "cublas__", "vllm": "vllm13__"}
            prefix = prefix_map.get(sub)
            if prefix:
                prompt_files = [pf for pf in prompt_files if pf.name.startswith(prefix)]
            elif sub == "nocublas":
                prompt_files = [pf for pf in prompt_files if not pf.name.startswith("cublas__")]

        if not prompt_files:
            print(f"Error: No prompt files found in {prompts_dir}")
            sys.exit(1)

        logger.info(f"Found {len(prompt_files)} prompts in {args.dataset}")

        # Filter if specified
        if args.kernels:
            filter_names = set(args.kernels.split(","))
            prompt_files = [
                pf for pf in prompt_files
                if pf.stem in filter_names or pf.stem.split("__")[-1] in filter_names
            ]
            logger.info(f"Filtered to {len(prompt_files)} prompts")

        # Build queue: (op_name, None, attempt) - None indicates generation mode
        for prompt_file in prompt_files:
            queue.append((prompt_file.stem, None, 0))

        run_name = f"ako4all_generate_{args.dataset}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Handle resume: reuse existing run directory
    if args.resume:
        resume_dir = SCRIPT_DIR / "runs" / args.resume
        if not resume_dir.exists():
            print(f"Error: Resume run not found: {resume_dir}")
            sys.exit(1)
        run_name = args.resume

    # Create run directory
    run_dir = SCRIPT_DIR / "runs" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    workspaces_dir = run_dir / "workspaces_ako"
    workspaces_dir.mkdir(exist_ok=True)

    optimized_kernels_dir = run_dir / "kernels"
    optimized_kernels_dir.mkdir(exist_ok=True)

    # Skip existing kernels when resuming
    if args.resume:
        force_rerun = set()
        if args.kernels:
            force_rerun = set(args.kernels.split(","))

        existing_kernels = set()
        for f in optimized_kernels_dir.glob("*.py"):
            if f.stem not in force_rerun:
                existing_kernels.add(f.stem)

        before = len(queue)
        queue = deque((k, p, a) for k, p, a in queue if k not in existing_kernels)
        logger.info(f"Resume: skipping {before - len(queue)} existing kernels, {len(queue)} remaining")

    # Save config
    config = {
        "mode": args.mode,
        "dataset": args.dataset,
        "baseline_run": str(args.baseline_run) if args.baseline_run else None,
        "iterations": args.iterations,
        "timeout": args.timeout,
        "claude_bin": args.claude_bin,
        "budget": args.budget,
        "max_retries": args.max_retries,
    }
    with open(run_dir / "config.yaml", "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    # Initialize device manager
    gpu_ids = args.gpu_ids
    if gpu_ids is None and args.device_count is not None:
        gpu_ids = list(range(args.device_count))
    device_mgr = DeviceManager(
        lock_dir="/tmp/agent_bench_gpu_locks",
        gpu_ids=gpu_ids,
    )

    # Initialize progress
    progress = Progress(run_dir / "progress.json")

    logger.info(f"Queue: {len(queue)} kernels to process")

    # Running tasks: {kernel_name: (proc, stdout_file, stderr_file, gpu_id, start_time, workspace_dir)}
    running: dict[str, tuple] = {}

    # Track last snapshot mtime per kernel for incremental saving
    last_snapshot_mtime: dict[str, float] = {}

    # Graceful shutdown
    shutdown_requested = False

    def signal_handler(sig, frame):
        nonlocal shutdown_requested
        if shutdown_requested:
            logger.warning("Force shutdown, exiting immediately")
            os._exit(1)
        shutdown_requested = True
        logger.warning(f"Shutdown requested, killing {len(running)} tasks...")

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    max_retries = args.max_retries

    logger.info(f"Starting: {len(queue)} kernels, {len(device_mgr.gpu_ids)} GPUs, max_retries={max_retries}")

    while (queue or running) and not shutdown_requested:
        # Launch new tasks
        while queue and not shutdown_requested:
            gpu_id = device_mgr.acquire()
            if gpu_id is None:
                break

            kernel_name, kernel_path, attempt = queue.popleft()
            workspace_dir = workspaces_dir / kernel_name

            try:
                # Set up workspace
                setup_ako4all_workspace(
                    workspace_dir=workspace_dir,
                    kernel_path=kernel_path,
                    iterations=args.iterations,
                    op_name=kernel_name,
                    dataset=args.dataset,
                    attempt=attempt,
                )

                # Launch AKO4ALL
                proc, stdout_file, stderr_file = launch_ako4all(
                    workspace_dir=workspace_dir,
                    gpu_id=gpu_id,
                    iterations=args.iterations,
                    op_name=kernel_name,
                    dataset=args.dataset,
                    attempt=attempt,
                    generate=(kernel_path is None),
                    claude_bin=args.claude_bin,
                    budget=args.budget,
                )

                running[kernel_name] = (
                    proc,
                    stdout_file,
                    stderr_file,
                    gpu_id,
                    time.time(),
                    workspace_dir,
                    attempt,
                    kernel_path,
                )
                progress.add_kernel(kernel_name, gpu_id, attempt)
                mode_str = "generation" if kernel_path is None else "optimization"
                retry_str = f" (attempt {attempt + 1})" if attempt > 0 else ""
                logger.info(f"Launched AKO4ALL {mode_str} for {kernel_name} (GPU={gpu_id}){retry_str}")

            except Exception as e:
                logger.error(f"Failed to launch AKO4ALL for {kernel_name}: {e}")
                device_mgr.release(gpu_id)
                if attempt + 1 < max_retries:
                    queue.append((kernel_name, kernel_path, attempt + 1))

        # Check running tasks
        for kernel_name in list(running.keys()):
            proc, stdout_file, stderr_file, gpu_id, start_time, workspace_dir, attempt, kernel_path = running[
                kernel_name
            ]
            elapsed = time.time() - start_time

            # Incremental snapshot: save solution/kernel.py whenever it's updated
            solution_path = workspace_dir / "solution" / "kernel.py"
            if solution_path.exists():
                try:
                    mtime = solution_path.stat().st_mtime
                    if mtime != last_snapshot_mtime.get(kernel_name):
                        optimized_path = optimized_kernels_dir / f"{kernel_name}.py"
                        shutil.copy2(solution_path, optimized_path)
                        last_snapshot_mtime[kernel_name] = mtime
                        logger.info(f"[SNAPSHOT] {kernel_name} - saved latest kernel ({elapsed:.0f}s)")
                except Exception as e:
                    logger.debug(f"Snapshot failed for {kernel_name}: {e}")

            # Check timeout
            if args.timeout and proc.poll() is None and elapsed > args.timeout:
                logger.warning(f"[TIMEOUT] {kernel_name} after {args.timeout}s")
                kill_process(proc, stdout_file, stderr_file)
                device_mgr.release(gpu_id)
                del running[kernel_name]

                has_kernel = (optimized_kernels_dir / f"{kernel_name}.py").exists()
                if has_kernel:
                    logger.info(f"[TIMEOUT+SAVED] {kernel_name} - kernel saved from earlier snapshot")
                    progress.update_kernel(
                        kernel_name,
                        status="completed",
                        duration_seconds=round(elapsed),
                        end_time=datetime.now(timezone.utc).isoformat(),
                        optimized=True,
                        error=f"Timeout after {args.timeout}s (kernel saved)",
                    )
                elif attempt + 1 < max_retries:
                    logger.warning(f"[RETRY] {kernel_name} after timeout (attempt {attempt + 1}/{max_retries})")
                    queue.append((kernel_name, kernel_path, attempt + 1))
                    progress.update_kernel(
                        kernel_name,
                        status="retrying",
                        duration_seconds=round(elapsed),
                        error=f"Timeout after {args.timeout}s, retrying...",
                    )
                else:
                    progress.update_kernel(
                        kernel_name,
                        status="timeout",
                        duration_seconds=round(elapsed),
                        end_time=datetime.now(timezone.utc).isoformat(),
                        error=f"Timeout after {args.timeout}s (max retries reached)",
                    )
                continue

            # Check if completed
            if proc.poll() is not None:
                device_mgr.release(gpu_id)
                del running[kernel_name]

                # Close file handles
                for fh in (stdout_file, stderr_file):
                    try:
                        if not fh.closed:
                            fh.close()
                    except Exception:
                        pass

                # Collect optimized kernel
                optimized_code = collect_optimized_kernel(workspace_dir)

                if optimized_code:
                    # Save optimized kernel
                    optimized_path = optimized_kernels_dir / f"{kernel_name}.py"
                    optimized_path.write_text(optimized_code)

                    logger.info(f"[SUCCESS] {kernel_name} ({elapsed:.0f}s)")
                    progress.update_kernel(
                        kernel_name,
                        status="completed",
                        duration_seconds=round(elapsed),
                        end_time=datetime.now(timezone.utc).isoformat(),
                        optimized=True,
                    )
                else:
                    if attempt + 1 < max_retries:
                        logger.warning(f"[RETRY] {kernel_name} - no kernel found (attempt {attempt + 1}/{max_retries})")
                        queue.append((kernel_name, kernel_path, attempt + 1))
                        progress.update_kernel(
                            kernel_name,
                            status="retrying",
                            error="No optimized kernel found, retrying...",
                        )
                    else:
                        logger.error(f"[FAILED] {kernel_name} - no optimized kernel found (max retries reached)")
                        progress.update_kernel(
                            kernel_name,
                            status="failed",
                            duration_seconds=round(elapsed),
                            end_time=datetime.now(timezone.utc).isoformat(),
                            error="No optimized kernel found in solution/",
                        )

        if running:
            time.sleep(10)  # Poll interval

    # Handle shutdown
    if shutdown_requested:
        for kernel_name, (
            proc,
            stdout_file,
            stderr_file,
            gpu_id,
            start_time,
            workspace_dir,
            attempt,
            kernel_path,
        ) in running.items():
            kill_process(proc, stdout_file, stderr_file)
            device_mgr.release(gpu_id)
            progress.update_kernel(
                kernel_name,
                status="cancelled",
                end_time=datetime.now(timezone.utc).isoformat(),
                duration_seconds=round(time.time() - start_time),
            )

    device_mgr.release_all()
    progress.finalize()

    # Restore terminal
    os.system("stty sane 2>/dev/null")

    # Print summary
    s = progress.data["summary"]
    print(f"\n{'='*60}")
    print(f"AKO4ALL {args.mode} completed: {run_name}")
    print(f"Total: {s['total']}, Completed: {s['completed']}, Failed: {s['failed']}")
    print(f"Results: {run_dir}")
    print(f"Optimized kernels: {optimized_kernels_dir}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(
        description="Run AKO4ALL optimization or generation on kernels"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["optimize", "generate"],
        default="optimize",
        help="Mode: optimize (needs --baseline-run) or generate (from scratch)",
    )
    parser.add_argument(
        "--baseline-run",
        "-b",
        type=str,
        default=None,
        help="Baseline run directory (required for optimize mode)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="KernelGenBench",
        help="Dataset name (default: KernelGenBench)",
    )
    parser.add_argument(
        "--kernels",
        "-k",
        type=str,
        default=None,
        help="Specific kernel(s) to process, comma-separated",
    )
    parser.add_argument(
        "--iterations",
        "-i",
        type=int,
        default=5,
        help="Number of optimization iterations per kernel (default: 5)",
    )
    parser.add_argument(
        "--timeout",
        "-t",
        type=int,
        default=1800,
        help="Timeout per kernel in seconds (default: 1800 = 30 min)",
    )
    parser.add_argument(
        "--claude-bin",
        type=str,
        default="claude",
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
        "--resume",
        type=str,
        default=None,
        help="Resume a previous run by name (skips existing kernels)",
    )

    device_group = parser.add_mutually_exclusive_group()
    device_group.add_argument(
        "--device-count",
        type=parse_device_count,
        default=None,
        help="Number of devices to use (default: auto-detect)",
    )
    device_group.add_argument(
        "--gpu-ids",
        type=parse_device_ids,
        default=None,
        help="Comma-separated physical device IDs to use (for example: 0,2,3)",
    )

    args = parser.parse_args()

    # Validate mode-specific requirements
    if args.mode == "optimize" and not args.baseline_run:
        print("Error: --baseline-run is required for optimize mode")
        sys.exit(1)

    if not AKO4ALL_TEMPLATE.exists():
        print(f"Error: AKO4ALL template not found: {AKO4ALL_TEMPLATE}")
        sys.exit(1)

    run(args)


if __name__ == "__main__":
    main()
