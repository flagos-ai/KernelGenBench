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
KernelGenBench Competition Anti-Cheat Package.

7-layer sandbox for detecting cheating in Triton kernel submissions.

Layers:
  1. File System Isolation   — cache_isolator.py
  2. Environment Variables   — applied in orchestrator
  3. Import Hook Sandbox     — import_hook.py
  4. CUDA Layer Protection   — cuda_protector.py
  5. Random Shapes           — shape_generator.py
  6. Process Isolation       — process_isolator.py (via multiprocessing)
  7. Timing Validation       — timing_validator.py

Usage:
    from sandbox.competition import run_check, CheckConfig

    result = run_check(
        kernel_path="path/to/kernel.py",
        generate_inputs=lambda: (torch.randn(128, 512, device='cuda'),),
    )
    print(result.passed)   # True if no cheating detected
    print(result.reason)   # Explanation
"""

from .orchestrator import run_check, CheckConfig, CheckResult
from . import cache_isolator
from . import import_hook
from . import cuda_protector
from . import shape_generator
from . import timing_validator
from . import process_isolator

__all__ = [
    "run_check",
    "CheckConfig",
    "CheckResult",
]
