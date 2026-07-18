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

"""Base class for agent methods."""

import subprocess
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class MethodResult:
    """Result returned by an agent method."""

    code: str | None = None  # Final kernel code
    passed: bool | None = None  # Whether passed verification (optional)
    speedup: float | None = None  # Performance speedup (optional)
    metadata: dict = field(default_factory=dict)  # Method-specific metadata


class BaseMethod(ABC):
    """Base class for agent methods.

    Each method implements a different strategy for generating Triton kernels.
    Methods are responsible for:
    - Building prompts
    - Launching CC processes
    - Extracting results from CC output
    """

    name: str = "base"

    @abstractmethod
    def launch(
        self,
        operator: str,
        prompt_path: Path,
        workspace_dir: Path,
        gpu_id: int,
        config: dict,
    ) -> Any:
        """Launch the agent process.

        Args:
            operator: Operator name
            prompt_path: Path to the operator prompt file
            workspace_dir: Working directory for this operator
            gpu_id: GPU ID to use
            config: Configuration dict

        Returns:
            Process handle or context needed for finish()
        """
        pass

    @abstractmethod
    def finish(
        self,
        operator: str,
        handle: Any,
        workspace_dir: Path,
        config: dict,
    ) -> MethodResult:
        """Finish processing and extract results.

        Called after the process completes.

        Args:
            operator: Operator name
            handle: Process handle from launch()
            workspace_dir: Working directory
            config: Configuration dict

        Returns:
            MethodResult with extracted code and metadata
        """
        pass

    @abstractmethod
    def get_process(self, handle: Any) -> subprocess.Popen:
        """Get the subprocess.Popen object from handle.

        This is used by run.py to check process status and handle timeouts.

        Args:
            handle: Process handle from launch()

        Returns:
            The subprocess.Popen object
        """
        pass

    def get_timeout(self, config: dict) -> int:
        """Get timeout in seconds."""
        return config.get("agent", {}).get("timeout", 1800)
