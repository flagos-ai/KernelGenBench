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

"""Vendor detection for multi-chip support."""
import functools
import os
import subprocess
from enum import Enum


class Vendor(Enum):
    NVIDIA = "nvidia"
    HYGON = "hygon"
    ASCEND = "ascend"
    ILUVATAR = "iluvatar"
    MTHREADS = "mthreads"
    METAX = "metax"


@functools.lru_cache(maxsize=1)
def detect_vendor() -> Vendor:
    """Detect hardware vendor. Priority: env var > system probe > torch device name."""
    # 1. GEMS_VENDOR environment variable (explicit override)
    gems_vendor = os.environ.get("GEMS_VENDOR", "")
    if gems_vendor:
        try:
            return Vendor(gems_vendor)
        except ValueError:
            pass

    # 2. Device-specific env vars
    if os.environ.get("ASCEND_RT_VISIBLE_DEVICES"):
        return Vendor.ASCEND
    if os.environ.get("MUSA_VISIBLE_DEVICES"):
        return Vendor.MTHREADS

    # 3. System command probes
    # MetaX: mx-smi or MACA_PATH
    try:
        r = subprocess.run(
            ["mx-smi"],
            capture_output=True, text=True, timeout=5,
        )
        if r.returncode == 0 and "MetaX" in r.stdout:
            return Vendor.METAX
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        pass

    if os.environ.get("MACA_PATH"):
        return Vendor.METAX

    # Hygon DCU: rocm-smi
    try:
        r = subprocess.run(
            ["rocm-smi", "--showproductname"],
            capture_output=True, text=True, timeout=5,
        )
        if r.returncode == 0 and any(
            k in r.stdout for k in ("Hygon", "DCU", "BW", "C-3000")
        ):
            return Vendor.HYGON
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        pass

    # 4. Torch device name detection
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            if "Hygon" in name or "DCU" in name or name.strip() == "BW":
                return Vendor.HYGON
            if "Iluvatar" in name:
                return Vendor.ILUVATAR
            if "MetaX" in name:
                return Vendor.METAX
        if "metax" in str(torch.__version__):
            return Vendor.METAX
        if hasattr(torch, "npu") and torch.npu.is_available():
            return Vendor.ASCEND
        if hasattr(torch, "musa") and torch.musa.is_available():
            return Vendor.MTHREADS
    except Exception:
        pass

    return Vendor.NVIDIA
