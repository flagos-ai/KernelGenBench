"""Device manager with lock-file based allocation. Supports CUDA, NPU, MUSA."""

import logging
import os
import subprocess
import time

logger = logging.getLogger(__name__)


def detect_device_type() -> str:
    """Detect current device type: 'cuda', 'npu', 'musa', or 'iluvatar'."""
    vendor = os.environ.get("GEMS_VENDOR", "")
    if vendor == "ascend" or os.environ.get("ASCEND_RT_VISIBLE_DEVICES"):
        return "npu"
    if vendor == "mthreads" or os.environ.get("MUSA_VISIBLE_DEVICES"):
        return "musa"
    if vendor == "iluvatar":
        return "iluvatar"
    # Auto-detect Iluvatar GPU via device name
    if not vendor:
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=10)
            if result.returncode == 0 and "Iluvatar" in result.stdout:
                return "iluvatar"
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
    return "cuda"


_VISIBLE_DEVICES_ENV = {
    "cuda": "CUDA_VISIBLE_DEVICES",
    "npu": "ASCEND_RT_VISIBLE_DEVICES",
    "musa": "MUSA_VISIBLE_DEVICES",
    "iluvatar": "CUDA_VISIBLE_DEVICES",
}


def get_device_env_var() -> str:
    """Get the environment variable name for device visibility."""
    return _VISIBLE_DEVICES_ENV.get(detect_device_type(), "CUDA_VISIBLE_DEVICES")


class DeviceManager:
    """Manages device allocation using lock files to prevent conflicts."""

    def __init__(self, lock_dir: str, gpu_ids: list[int] | None = None):
        self.lock_dir = lock_dir
        os.makedirs(lock_dir, exist_ok=True)
        self.device_type = detect_device_type()

        if gpu_ids is not None:
            self.gpu_ids = gpu_ids
        else:
            self.gpu_ids = self._detect_devices()

        logger.info(f"DeviceManager initialized: type={self.device_type}, devices={self.gpu_ids}")

    def _detect_devices(self) -> list[int]:
        """Detect available devices based on device type."""
        if self.device_type == "cuda":
            return self._detect_via_cmd(
                ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"])
        elif self.device_type == "npu":
            return self._detect_npu()
        elif self.device_type == "musa":
            return self._detect_via_cmd(["musa-smi", "-L"])
        return [0]

    def _detect_via_cmd(self, cmd: list[str]) -> list[int]:
        """Detect devices by running a command that outputs device indices."""
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                ids = [int(line.strip()) for line in result.stdout.strip().split("\n")
                       if line.strip() and line.strip().isdigit()]
                if ids:
                    return ids
        except (subprocess.TimeoutExpired, FileNotFoundError, ValueError):
            pass
        logger.warning(f"Failed to detect devices via {cmd[0]}, defaulting to [0]")
        return [0]

    def _detect_npu(self) -> list[int]:
        """Detect Ascend NPU devices via npu-smi."""
        try:
            result = subprocess.run(
                ["npu-smi", "info", "-l"], capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                import re
                ids = [int(m) for m in re.findall(r"NPU ID\s*:\s*(\d+)", result.stdout)]
                if ids:
                    return ids
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
        logger.warning("Failed to detect NPU devices, defaulting to [0]")
        return [0]

    def _lock_path(self, gpu_id: int) -> str:
        return os.path.join(self.lock_dir, f"gpu_{gpu_id}.lock")

    def acquire(self) -> int | None:
        """Acquire a free GPU. Returns gpu_id or None if all busy."""
        for gpu_id in self.gpu_ids:
            lock_path = self._lock_path(gpu_id)
            if os.path.exists(lock_path):
                # Check if the holding process is still alive
                if self._is_lock_stale(lock_path):
                    logger.info(f"Removing stale lock for GPU {gpu_id}")
                    os.remove(lock_path)
                else:
                    continue

            # Acquire the lock
            try:
                with open(lock_path, "w") as f:
                    f.write(f"{os.getpid()}\n{time.time()}\n")
                logger.info(f"Acquired GPU {gpu_id}")
                return gpu_id
            except OSError as e:
                logger.warning(f"Failed to acquire GPU {gpu_id}: {e}")
                continue

        return None

    def release(self, gpu_id: int):
        """Release a GPU lock."""
        lock_path = self._lock_path(gpu_id)
        if os.path.exists(lock_path):
            os.remove(lock_path)
            logger.info(f"Released GPU {gpu_id}")

    def _is_lock_stale(self, lock_path: str) -> bool:
        """Check if a lock file's owning process is dead."""
        try:
            with open(lock_path) as f:
                lines = f.read().strip().split("\n")
                pid = int(lines[0])
            # Check if process is alive
            os.kill(pid, 0)
            return False
        except (OSError, ValueError, IndexError):
            return True

    def release_all(self):
        """Release all locks owned by this process."""
        for gpu_id in self.gpu_ids:
            lock_path = self._lock_path(gpu_id)
            if os.path.exists(lock_path):
                try:
                    with open(lock_path) as f:
                        pid = int(f.read().strip().split("\n")[0])
                    if pid == os.getpid():
                        os.remove(lock_path)
                        logger.info(f"Released GPU {gpu_id}")
                except (OSError, ValueError, IndexError):
                    pass

    def available_count(self) -> int:
        """Return the number of currently available GPUs."""
        count = 0
        for gpu_id in self.gpu_ids:
            lock_path = self._lock_path(gpu_id)
            if not os.path.exists(lock_path) or self._is_lock_stale(lock_path):
                count += 1
        return count
