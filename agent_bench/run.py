#!/usr/bin/env python3
"""Batch run agents to generate Triton kernels."""

import argparse
import json
import logging
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from collections import deque

try:
    import yaml
except ImportError:
    yaml = None

from device_manager import DeviceManager
from methods import get_method, list_methods

logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent


def load_dotenv(env_path: str = None):
    """Load .env file into os.environ."""
    if env_path is None:
        env_path = SCRIPT_DIR / ".env"
    if not os.path.exists(env_path):
        return
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, _, val = line.partition("=")
                key, val = key.strip(), val.strip()
                if val and val[0] in ('"', "'") and val[-1] == val[0]:
                    val = val[1:-1]
                if key:
                    os.environ[key] = val


def load_config(config_path: Path) -> dict:
    """Load YAML config file."""
    if yaml is None:
        print("Error: 'pyyaml' required. Install: pip install pyyaml")
        sys.exit(1)
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_ops_from_prompts(prompts_dir: Path, namespace: str) -> list[str]:
    """Load operator names by scanning prompt files in directory.

    Args:
        prompts_dir: Directory containing prompt .md files
        namespace: Operator namespace (e.g., "aten" or "cupy")

    Returns:
        List of full operator names (e.g., ["aten::add", "aten::softmax"])
    """
    ops = []
    for f in sorted(prompts_dir.glob("*.md")):
        op_name = f.stem  # e.g., "softmax" from "softmax.md"
        ops.append(f"{namespace}::{op_name}")
    return ops


def kill_process(handle: dict):
    """Kill a process and its process group."""
    proc = handle.get("proc")
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

    # Close file handles safely
    stdout_file = handle.get("stdout_file")
    stderr_file = handle.get("stderr_file")
    if stdout_file:
        try:
            if not stdout_file.closed:
                stdout_file.close()
        except Exception:
            pass
    if stderr_file:
        try:
            if not stderr_file.closed:
                stderr_file.close()
        except Exception:
            pass


class Progress:
    """Manages progress.json with real-time updates."""

    def __init__(self, path: Path, dataset: str):
        self.path = path
        self.data = {
            "dataset": dataset,
            "start_time": datetime.now(timezone.utc).isoformat(),
            "end_time": None,
            "summary": {
                "total": 0,
                "completed": 0,
                "running": 0,
                "failed": 0,
                "pending": 0,
            },
            "operators": {},
        }
        self._save()

    def add_operator(self, operator: str, gpu_id: int, attempt: int):
        """Record operator task started."""
        self.data["operators"][operator] = {
            "status": "running",
            "gpu_id": gpu_id,
            "attempt": attempt,
            "start_time": datetime.now(timezone.utc).isoformat(),
            "end_time": None,
            "duration_seconds": None,
            "code_generated": False,
            "error": None,
        }
        self._recount()
        self._save()

    def update_operator(self, operator: str, **kwargs):
        """Update operator fields."""
        if operator in self.data["operators"]:
            self.data["operators"][operator].update(kwargs)
            self._recount()
            self._save()

    def finalize(self):
        """Mark run as complete."""
        self.data["end_time"] = datetime.now(timezone.utc).isoformat()
        self._save()

    def _recount(self):
        """Recount summary statistics."""
        ops = self.data["operators"]
        self.data["summary"]["total"] = len(ops)
        self.data["summary"]["completed"] = sum(1 for v in ops.values() if v["status"] == "completed")
        self.data["summary"]["failed"] = sum(1 for v in ops.values() if v["status"] in ("failed", "timeout"))
        self.data["summary"]["running"] = sum(1 for v in ops.values() if v["status"] == "running")
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


def run(args):
    """Main orchestration loop."""
    # Load config
    config_path = args.config or (SCRIPT_DIR / "config.yaml")
    config = load_config(config_path)

    # Store dataset in config for methods to access
    config["dataset"] = args.dataset

    # Override config with command line args for iterative_optimizer
    if args.max_optimize_calls is not None:
        config.setdefault("agent", {})["max_optimize_calls"] = args.max_optimize_calls
    if args.target_speedup is not None:
        config.setdefault("agent", {})["target_speedup"] = args.target_speedup

    # Get method (auto-detect from run config when resuming)
    method_name = args.method
    if args.resume and method_name == "naive_cc":
        # User didn't explicitly specify method, try to detect from run config
        run_config_path = (SCRIPT_DIR / config.get("paths", {}).get("runs", "runs")
                          / args.resume / "config.yaml")
        if run_config_path.exists():
            run_config = load_config(run_config_path)
            saved_method = run_config.get("method")
            if saved_method:
                method_name = saved_method
                logger.info(f"Auto-detected method from run config: {method_name}")
    method = get_method(method_name)
    logger.info(f"Using method: {method.name}")

    # Paths
    prompts_dir = SCRIPT_DIR / config.get("paths", {}).get("prompts", "prompts")
    runs_dir = SCRIPT_DIR / config.get("paths", {}).get("runs", "runs")

    # Dataset
    dataset = args.dataset
    dataset_prompts_dir = prompts_dir / dataset

    # Load operators from prompt files
    namespace = "cupy" if dataset == "cupy" else "aten"
    ops = load_ops_from_prompts(dataset_prompts_dir, namespace)
    if not ops:
        print(f"Error: No prompt files found in {dataset_prompts_dir}")
        sys.exit(1)

    # Filter operators if specified (exact match on operator name)
    if args.op:
        filter_ops = set(args.op.split(","))
        ops = [op for op in ops if op.split("::")[-1] in filter_ops]

    logger.info(f"Loaded {len(ops)} operators for dataset {dataset}")

    # Create run directory
    if args.resume:
        run_name = args.resume
        run_dir = runs_dir / run_name
        if not run_dir.exists():
            print(f"Error: Run directory {run_dir} not found")
            sys.exit(1)
    else:
        run_name = f"{method.name}_{dataset}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        run_dir = runs_dir / run_name
        run_dir.mkdir(parents=True, exist_ok=True)

    kernels_dir = run_dir / "kernels"
    kernels_dir.mkdir(parents=True, exist_ok=True)

    workspaces_dir = run_dir / "workspaces"
    workspaces_dir.mkdir(parents=True, exist_ok=True)

    # Save config snapshot with dataset and method
    config_snapshot_path = run_dir / "config.yaml"
    if not config_snapshot_path.exists():
        config_snapshot = config.copy()
        config_snapshot["dataset"] = dataset
        config_snapshot["method"] = method.name
        with open(config_snapshot_path, "w") as f:
            yaml.dump(config_snapshot, f, default_flow_style=False)

    # Initialize device manager
    device_cfg = config.get("device", {}) or {}
    device_mgr = DeviceManager(
        lock_dir=device_cfg.get("lock_dir", "/tmp/agent_bench_gpu_locks"),
        gpu_ids=device_cfg.get("gpu_ids"),
    )

    # Initialize progress
    progress = Progress(run_dir / "progress.json", dataset)

    # Check existing kernels (for resume)
    existing_kernels = set()
    if args.resume:
        # When --op is specified with --resume, force re-run those operators
        force_rerun = set()
        if args.op:
            force_rerun = set(args.op.split(","))

        for f in kernels_dir.glob("*.py"):
            op_name = f.stem
            if op_name not in force_rerun:
                existing_kernels.add(op_name)
        logger.info(f"Found {len(existing_kernels)} existing kernels (skipping)")
        if force_rerun:
            logger.info(f"Force re-run: {', '.join(force_rerun)}")

    # Build task queue
    queue = deque()
    for full_name in ops:
        op_name = full_name.split("::")[-1]
        if op_name not in existing_kernels:
            queue.append((full_name, op_name, 0))

    logger.info(f"Queue: {len(queue)} operators to process")

    # Agent config
    timeout = method.get_timeout(config)
    max_retries = config.get("agent", {}).get("max_retries", 3)
    poll_interval = config.get("poll_interval", 10)

    # Running tasks: {op_name: (handle, gpu_id, attempt, full_name, start_time)}
    running: dict[str, tuple] = {}

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

    logger.info(f"Starting: {len(queue)} operators, {len(device_mgr.gpu_ids)} GPUs")

    while (queue or running) and not shutdown_requested:
        # Launch new tasks
        while queue and not shutdown_requested:
            gpu_id = device_mgr.acquire()
            if gpu_id is None:
                break

            full_name, op_name, attempt = queue.popleft()
            prompt_path = dataset_prompts_dir / f"{op_name}.md"

            if not prompt_path.exists():
                logger.warning(f"Prompt not found: {prompt_path}")
                device_mgr.release(gpu_id)
                continue

            workspace_dir = workspaces_dir / op_name

            try:
                handle = method.launch(
                    operator=op_name,
                    prompt_path=prompt_path,
                    workspace_dir=workspace_dir,
                    gpu_id=gpu_id,
                    config=config,
                )
                running[op_name] = (handle, gpu_id, attempt, full_name, time.time(), workspace_dir)
                progress.add_operator(op_name, gpu_id, attempt + 1)
                logger.info(f"Launched {method.name} for {op_name} (GPU={gpu_id})")
            except Exception as e:
                logger.error(f"Failed to launch agent for {op_name}: {e}")
                device_mgr.release(gpu_id)
                if attempt + 1 < max_retries:
                    queue.append((full_name, op_name, attempt + 1))

        # Check running tasks
        for op_name in list(running.keys()):
            handle, gpu_id, attempt, full_name, start_time, workspace_dir = running[op_name]
            elapsed = time.time() - start_time
            proc = method.get_process(handle)

            # Check timeout
            if timeout and proc.poll() is None and elapsed > timeout:
                logger.error(f"[TIMEOUT] {op_name} after {timeout}s")
                kill_process(handle)
                device_mgr.release(gpu_id)
                del running[op_name]

                # Retry timeout cases
                if attempt + 1 < max_retries:
                    logger.warning(f"[RETRY] {op_name} after timeout (attempt {attempt + 1})")
                    queue.append((full_name, op_name, attempt + 1))
                    progress.update_operator(
                        op_name,
                        status="retrying",
                        duration_seconds=round(elapsed),
                        error=f"Timeout after {timeout}s, retrying...",
                    )
                else:
                    progress.update_operator(
                        op_name,
                        status="timeout",
                        duration_seconds=round(elapsed),
                        end_time=datetime.now(timezone.utc).isoformat(),
                        error=f"Timeout after {timeout}s (max retries reached)",
                    )
                continue

            # Check if completed
            if proc.poll() is not None:
                device_mgr.release(gpu_id)
                del running[op_name]

                # Finish and extract result
                result = method.finish(
                    operator=op_name,
                    handle=handle,
                    workspace_dir=workspace_dir,
                    config=config,
                )

                if result.code:
                    # Save kernel to unified directory
                    kernel_path = kernels_dir / f"{op_name}.py"
                    kernel_path.write_text(result.code)

                    logger.info(f"[SUCCESS] {op_name} ({elapsed:.0f}s)")
                    progress.update_operator(
                        op_name,
                        status="completed",
                        duration_seconds=round(elapsed),
                        end_time=datetime.now(timezone.utc).isoformat(),
                        code_generated=True,
                    )
                else:
                    # Failed to extract code
                    if attempt + 1 < max_retries:
                        logger.warning(f"[RETRY] {op_name} (attempt {attempt + 1})")
                        queue.append((full_name, op_name, attempt + 1))
                        progress.update_operator(op_name, status="retrying")
                    else:
                        logger.error(f"[FAILED] {op_name} - no code extracted")
                        progress.update_operator(
                            op_name,
                            status="failed",
                            duration_seconds=round(elapsed),
                            end_time=datetime.now(timezone.utc).isoformat(),
                            error="Failed to extract code from output",
                        )

        if running:
            time.sleep(poll_interval)

    # Handle shutdown
    if shutdown_requested:
        for op_name, (handle, gpu_id, attempt, full_name, start_time, workspace_dir) in running.items():
            kill_process(handle)
            device_mgr.release(gpu_id)
            progress.update_operator(
                op_name,
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
    print(f"\n{'='*50}")
    print(f"Run completed: {run_name}")
    print(f"Method: {method.name}")
    print(f"Total: {s['total']}, Completed: {s['completed']}, Failed: {s['failed']}")
    print(f"Results: {run_dir}")
    print(f"{'='*50}")


def main():
    parser = argparse.ArgumentParser(description="Batch run agents to generate Triton kernels")
    parser.add_argument(
        "--dataset", "-d",
        type=str,
        default="v2_1",
        help="Dataset to run (default: v2_1)"
    )
    parser.add_argument(
        "--op", "-o",
        type=str,
        default=None,
        help="Specific operator(s) to run, comma-separated"
    )
    parser.add_argument(
        "--method", "-m",
        type=str,
        default="naive_cc",
        choices=list_methods(),
        help=f"Agent method to use (default: naive_cc, available: {', '.join(list_methods())})"
    )
    parser.add_argument(
        "--resume", "-r",
        type=str,
        default=None,
        help="Resume from existing run directory name"
    )
    parser.add_argument(
        "--config", "-c",
        type=Path,
        default=None,
        help="Path to config.yaml"
    )
    parser.add_argument(
        "--max-optimize-calls",
        type=int,
        default=None,
        help="Max CC calls for iterative_optimizer (default: from config or 10)"
    )
    parser.add_argument(
        "--target-speedup",
        type=float,
        default=None,
        help="Target speedup for iterative_optimizer (default: from config or 1.0)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable debug logging"
    )

    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    load_dotenv()
    run(args)


if __name__ == "__main__":
    main()
