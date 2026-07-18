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

"""
Verifier Server for Operator Verification (Multi-Backend Support)

A HTTP API server for executing operator tests on CUDA devices.
The client passes kernel_code (source content), and the server saves it to a temporary
file before execution.

Architecture:
    - DeviceStatesManager: Manages device states (idle/busy)
    - TasksManager: Manages task queue, assigns tasks to idle devices

Usage:
    python -m sandbox.server.verifier_server --port 8888

API:
    POST /test          - Submit test request (synchronously waits for result)
    GET  /status        - Get server and device status
    GET  /health        - Health check
"""

import os
import sys
import asyncio
import subprocess
import json
import logging
import tempfile
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any
from dataclasses import dataclass
from enum import Enum

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import uvicorn

# Import runtime module for device detection
import torch
class _device: name = "cuda"
device = _device()
class _torch_device_fn:
    @staticmethod
    def device_count(): return torch.cuda.device_count()
torch_device_fn = _torch_device_fn()
def get_visible_devices_env(): return "CUDA_VISIBLE_DEVICES"

# =============================================================================
# Setup
# =============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent.parent
# Default test script - can be overridden via command line
TEST_SCRIPT = SCRIPT_DIR / "test_single_operator.py"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# =============================================================================
# Device States Manager
# =============================================================================


class DeviceState(str, Enum):
    IDLE = "idle"
    BUSY = "busy"


@dataclass
class DeviceInfo:
    device_id: int
    state: DeviceState = DeviceState.IDLE
    current_task: Optional[str] = None


class DeviceStatesManager:
    """Manages device states, ensuring only one test runs per device at a time.

    Supports CUDA backend through the runtime module.
    """

    def __init__(self):
        self._devices: Dict[int, DeviceInfo] = {}
        self._lock = asyncio.Lock()
        self._device_type = device.name  # cuda
        self._init_devices()

    @property
    def device_type(self) -> str:
        """Return the current device type (cuda)."""
        return self._device_type

    def _init_devices(self):
        """Detect device count and initialize device states."""
        device_count = self._detect_device_count()
        logger.info(f"Detected {device_count} {self._device_type.upper()} device(s)")
        for i in range(device_count):
            self._devices[i] = DeviceInfo(device_id=i)

    def _detect_device_count(self) -> int:
        """Detect device count using runtime module.

        Priority:
        1. TEST_DEVICE_COUNT environment variable (if set)
        2. torch_device_fn.device_count() from runtime
        """
        # Check for explicit override via environment variable
        test_device_count = os.environ.get("TEST_DEVICE_COUNT")
        if test_device_count:
            try:
                count = int(test_device_count)
                logger.info(f"Using TEST_DEVICE_COUNT={count}")
                return count
            except ValueError:
                logger.warning(f"Invalid TEST_DEVICE_COUNT value: {test_device_count}")

        # Use runtime module for device detection
        try:
            count = torch_device_fn.device_count()
            if count > 0:
                return count
        except Exception as e:
            logger.warning(f"Failed to detect devices via torch_device_fn: {e}")

        # Default fallback
        logger.warning("Could not detect devices, defaulting to 1")
        return 1

    async def acquire(self) -> Optional[int]:
        """Acquire an idle device. Returns device_id, or None if no device is available."""
        async with self._lock:
            for device_id, info in self._devices.items():
                if info.state == DeviceState.IDLE:
                    info.state = DeviceState.BUSY
                    logger.info(f"Allocated Device {device_id}")
                    return device_id
            return None

    async def release(self, device_id: int):
        """Release a device."""
        async with self._lock:
            if device_id in self._devices:
                self._devices[device_id].state = DeviceState.IDLE
                self._devices[device_id].current_task = None
                logger.info(f"Released Device {device_id}")

    async def set_task(self, device_id: int, task: str):
        """Set current task for a device."""
        async with self._lock:
            if device_id in self._devices:
                self._devices[device_id].current_task = task

    def get_status(self) -> Dict[str, Any]:
        """Get status of all devices."""
        return {
            "device_type": self._device_type,
            "total": len(self._devices),
            "idle": sum(1 for d in self._devices.values() if d.state == DeviceState.IDLE),
            "busy": sum(1 for d in self._devices.values() if d.state == DeviceState.BUSY),
            "devices": {
                i: {"state": d.state.value, "task": d.current_task}
                for i, d in self._devices.items()
            }
        }


# =============================================================================
# Tasks Manager
# =============================================================================


@dataclass
class Task:
    operator_name: str
    kernel_code: str
    test_module: str
    test_set: str
    timeout: int
    output_dir: str


class TasksManager:
    """Manages task queue, assigns tasks to idle devices for execution."""

    def __init__(self, device_manager: DeviceStatesManager, test_script: Path = None):
        self._device_manager = device_manager
        self._test_script = test_script or TEST_SCRIPT

    async def submit_and_wait(self, task: Task) -> Dict[str, Any]:
        """Submit a task and wait for result."""
        # Wait for an idle device
        device_id = None
        while device_id is None:
            device_id = await self._device_manager.acquire()
            if device_id is None:
                logger.info("No idle device available, waiting...")
                await asyncio.sleep(1)

        try:
            await self._device_manager.set_task(device_id, task.operator_name)
            result = await self._run_on_device(task, device_id)
            return result
        finally:
            await self._device_manager.release(device_id)

    async def _run_on_device(self, task: Task, device_id: int) -> Dict[str, Any]:
        """Execute test on a specified device."""
        device_type = self._device_manager.device_type.upper()
        logger.info(f"Executing test on {device_type} {device_id}: {task.operator_name}")

        # Save kernel_code to a temporary file
        temp_file = None
        try:
            # Create temporary file for kernel code
            temp_file = Path(task.output_dir) / f"aten_{task.operator_name}.py"
            temp_file.parent.mkdir(parents=True, exist_ok=True)
            with open(temp_file, 'w', encoding='utf-8') as f:
                f.write(task.kernel_code)

            # Get the appropriate environment variable for device visibility
            visible_env = get_visible_devices_env()

            # Build command with device visibility set
            cmd = f"{visible_env}={device_id} {sys.executable} {self._test_script} " \
                  f"{temp_file} --device-count 1 --timeout {task.timeout} " \
                  f"--output-dir {task.output_dir}"
            if task.test_module:
                cmd += f" --test-module {task.test_module}"
            elif task.test_set:
                cmd += f" --test-set {task.test_set}"

            logger.info(f"Executing command: {cmd}")

            # Execute subprocess asynchronously
            loop = asyncio.get_running_loop()
            returncode, stdout, stderr = await loop.run_in_executor(
                None, self._run_subprocess_shell, cmd, task.timeout + 120
            )

            # Read result file
            result_file = Path(task.output_dir) / f"tle_test_{task.operator_name}" / "log_0" / "result.json"

            if result_file.exists():
                with open(result_file, 'r') as f:
                    results = json.load(f)
                    result = results[0]
                result["device_id"] = device_id
                result["operator"] = task.operator_name
                return result
            else:
                # Result file does not exist, return error
                stderr_text = stderr[:2000] if stderr else ""
                return {
                    "success": False,
                    "operator": task.operator_name,
                    "device_id": device_id,
                    "error": f"Test execution failed (exit code: {returncode})",
                    "traceback": stderr_text
                }

        except Exception as e:
            return {
                "success": False,
                "operator": task.operator_name,
                "device_id": device_id,
                "error": str(e)
            }

    def _run_subprocess_shell(self, cmd: str, timeout: int) -> tuple:
        """Execute a shell command (synchronous, runs in executor)."""
        try:
            proc = subprocess.Popen(
                cmd,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            stdout, stderr = proc.communicate(timeout=timeout)
            return (
                proc.returncode,
                stdout.decode('utf-8', errors='ignore'),
                stderr.decode('utf-8', errors='ignore')
            )
        except subprocess.TimeoutExpired:
            proc.kill()
            stdout, stderr = proc.communicate()
            return (
                -1,
                stdout.decode('utf-8', errors='ignore') if stdout else "",
                "Timeout"
            )


# =============================================================================
# API Models
# =============================================================================


class TestRequest(BaseModel):
    """Test request model."""
    operator_name: str = Field(..., description="Operator name")
    kernel_code: str = Field(..., description="Kernel source code content")
    test_module: str = Field(default="", description="Test module (takes priority over test_set)")
    test_set: str = Field(default="KernelGenBench", description="Test set: KernelGenBench")
    timeout: int = Field(default=300, description="Timeout in seconds")


class TestResponse(BaseModel):
    """Test response model."""
    success: bool
    operator: str
    device_id: Optional[int] = None
    error: Optional[str] = None
    traceback: Optional[str] = None


# =============================================================================
# Server
# =============================================================================


class VerifierServer:
    """HTTP server for operator verification with multi-backend support."""

    def __init__(self, output_dir: str = None, test_script: str = None):
        self.output_dir = Path(output_dir) if output_dir else PROJECT_ROOT / "test_server_results"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.device_manager = DeviceStatesManager()
        self.task_manager = TasksManager(
            self.device_manager,
            test_script=Path(test_script) if test_script else None
        )

        self.app = FastAPI(
            title="Verifier Server",
            description="Operator verification server (CUDA)",
            version="2.0.0"
        )
        self._setup_routes()

    def _setup_routes(self):
        @self.app.get("/health")
        async def health():
            return {
                "status": "healthy",
                "timestamp": datetime.now().isoformat(),
                "device_type": self.device_manager.device_type
            }

        @self.app.get("/status")
        async def status():
            return {
                "device": self.device_manager.get_status(),
                "output_dir": str(self.output_dir)
            }

        @self.app.post("/test", response_model=TestResponse)
        async def submit_test(request: TestRequest):
            task = Task(
                operator_name=request.operator_name,
                kernel_code=request.kernel_code,
                test_module=request.test_module,
                test_set=request.test_set,
                timeout=request.timeout,
                output_dir=str(self.output_dir)
            )

            result = await self.task_manager.submit_and_wait(task)

            return TestResponse(
                success=result.get("success", False),
                operator=result.get("operator", request.operator_name),
                device_id=result.get("device_id"),
                error=result.get("error"),
                traceback=result.get("traceback")
            )

    def run(self, host: str = "0.0.0.0", port: int = 8888):
        uvicorn.run(self.app, host=host, port=port)


# =============================================================================
# Main
# =============================================================================


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Verifier Server (Multi-Backend)")
    parser.add_argument("--host", type=str, default="0.0.0.0",
                        help="Host address to bind to")
    parser.add_argument("--port", type=int, default=8888,
                        help="Port to listen on")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Directory for output files")
    parser.add_argument("--test-script", type=str, default=None,
                        help="Path to the test script to execute")

    args = parser.parse_args()

    server = VerifierServer(output_dir=args.output_dir, test_script=args.test_script)
    device_status = server.device_manager.get_status()
    logger.info(f"Starting server: {args.host}:{args.port}")
    logger.info(f"Device type: {device_status['device_type'].upper()}")
    logger.info(f"Device status: {device_status['total']} total, "
                f"{device_status['idle']} idle, {device_status['busy']} busy")
    server.run(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
