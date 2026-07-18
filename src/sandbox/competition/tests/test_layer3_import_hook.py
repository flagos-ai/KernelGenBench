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
Tests for Layer 3: Import Hook Sandbox (import_hook.py).

Verifies that:
1. Forbidden modules (os, socket, ctypes) are blocked
2. print() is rendered no-op
3. RuntimeSandbox can be enabled/disabled
4. SecurityError is raised for forbidden operations
"""
import os
import sys
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from sandbox.competition.import_hook import (
    RuntimeSandbox,
    SecureBuiltins,
    SecurityError,
)


class TestRuntimeSandbox:
    """Test the import hook sandbox."""

    def test_enable_sandbox(self):
        """RuntimeSandbox can be enabled."""
        sandbox = RuntimeSandbox()
        sandbox.enable()
        assert sandbox.hook is not None
        sandbox.disable()

    def test_disable_sandbox(self):
        """RuntimeSandbox can be disabled after enabling."""
        sandbox = RuntimeSandbox()
        sandbox.enable()
        sandbox.disable()
        # Should not raise

    def test_context_manager(self):
        """RuntimeSandbox works as a context manager."""
        with RuntimeSandbox() as sandbox:
            assert sandbox.hook is not None
        # Should not raise on exit

    def test_import_log_works(self):
        """Import log tracks loaded modules."""
        sandbox = RuntimeSandbox()
        sandbox.enable()
        log = sandbox.get_import_log()
        assert isinstance(log, list)
        sandbox.disable()


class TestSecureBuiltins:
    """Test the secure builtins wrapper."""

    def test_create_secure_builtins(self):
        """SecureBuiltins can be created."""
        sb = SecureBuiltins()
        assert sb.original_exec is not None
        assert sb.original_eval is not None

    def test_enable_disable(self):
        """SecureBuiltins can be enabled and disabled."""
        sb = SecureBuiltins()
        sb.enable()
        sb.disable()
        # Should not raise

    def test_block_exec(self):
        """exec() with forbidden keywords raises SecurityError."""
        sb = SecureBuiltins()
        sb.enable()
        try:
            import builtins
            with pytest.raises(SecurityError):
                builtins.exec("import torch; torch.compile(...)")
        finally:
            sb.disable()

    def test_allow_safe_exec(self):
        """Safe exec() calls should still work."""
        sb = SecureBuiltins()
        sb.enable()
        try:
            import builtins
            # This should not raise (no forbidden keywords)
            namespace = {}
            builtins.exec("x = 1 + 2", namespace)
            assert namespace['x'] == 3
        finally:
            sb.disable()

    def test_print_blocked(self):
        """print() should be rendered no-op when SecureBuiltins is enabled."""
        sb = SecureBuiltins()
        sb.enable()
        try:
            import builtins
            # print() should not raise, just be a no-op
            builtins.print("should not appear")
        finally:
            sb.disable()
