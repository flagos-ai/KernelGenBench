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
Tests for Layer 1: Cache Isolation (cache_isolator.py).

Verifies that:
1. HOME directory is isolated to a temp dir
2. Triton/PyTorch cache env vars are set
3. Cleanup removes the isolated directory
"""
import os
import sys
import tempfile
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from sandbox.competition.cache_isolator import CacheIsolator


class TestCacheIsolator:
    """Test the file system isolation layer."""

    def test_create_isolator(self):
        """CacheIsolator can be created."""
        ci = CacheIsolator()
        assert ci.original_home is not None
        assert ci.isolated_home is None

    def test_isolate_changes_home(self):
        """isolate() creates a new HOME directory."""
        ci = CacheIsolator()
        original = os.environ.get('HOME', '')
        result = ci.isolate()

        assert result is not None
        assert os.path.isdir(result)
        assert os.environ['HOME'] != original
        assert os.environ['HOME'] == result

    def test_isolate_sets_env_vars(self):
        """isolate() sets cache-related environment variables."""
        ci = CacheIsolator()
        ci.isolate()

        assert 'TRITON_CACHE_DIR' in os.environ
        assert 'TORCHINDUCTOR_CACHE_DIR' in os.environ
        assert os.environ.get('CUDA_CACHE_DISABLE') == '1'
        assert 'XDG_CACHE_HOME' in os.environ

    def test_cleanup_restores_and_removes(self):
        """cleanup() restores HOME and removes the temp dir."""
        ci = CacheIsolator()
        original = os.environ.get('HOME', '')
        isolated = ci.isolate()
        assert os.path.isdir(isolated)

        ci.cleanup()
        assert os.environ['HOME'] == original
        assert not os.path.isdir(isolated)

    def test_context_manager(self):
        """CacheIsolator works as a context manager."""
        original = os.environ.get('HOME', '')
        with CacheIsolator() as isolated:
            assert os.path.isdir(isolated)
            assert os.environ['HOME'] == isolated
        assert os.environ['HOME'] == original
        assert not os.path.isdir(isolated)


class TestCacheIsolatorIsolation:
    """Test that isolation actually prevents cross-contamination."""

    def test_different_runs_have_different_homes(self):
        """Each isolation creates a unique HOME."""
        ci1 = CacheIsolator()
        ci2 = CacheIsolator()

        home1 = ci1.isolate()
        home2 = ci2.isolate()

        assert home1 != home2

        ci1.cleanup()
        ci2.cleanup()

    def test_no_shared_cache_between_runs(self):
        """Files written in one run should not be visible in another."""
        ci1 = CacheIsolator()
        home1 = ci1.isolate()
        test_file = os.path.join(home1, '.test_cache_file')
        with open(test_file, 'w') as f:
            f.write('cached_data')
        ci1.cleanup()

        # Second run should not see the file
        ci2 = CacheIsolator()
        home2 = ci2.isolate()
        assert not os.path.exists(os.path.join(home2, '.test_cache_file'))
        ci2.cleanup()
