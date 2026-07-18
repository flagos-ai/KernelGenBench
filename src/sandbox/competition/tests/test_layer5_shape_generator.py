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
Tests for Layer 5: Random Shape Generator (shape_generator.py).

Verifies that:
1. BucketedShapeGenerator generates valid shapes
2. Shapes are within expected ranges
3. Noise keeps shapes within reasonable bounds
4. TensorLayoutRandomizer generates valid layouts
"""
import os
import sys
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from sandbox.competition.shape_generator import (
    ShapeBucket,
    BucketedShapeGenerator,
    TensorLayoutRandomizer,
)


class TestShapeBucket:
    """Test the ShapeBucket data class."""

    def test_create_bucket(self):
        bucket = ShapeBucket(base=128, noise_range=(-8, 8), alignment=32)
        assert bucket.base == 128
        assert bucket.noise_range == (-8, 8)
        assert bucket.alignment == 32


class TestBucketedShapeGenerator:
    """Test the bucketed shape generator."""

    def test_create_generator(self):
        gen = BucketedShapeGenerator()
        assert gen is not None

    def test_generate_gemm_small(self):
        """Small shapes are generated correctly."""
        gen = BucketedShapeGenerator()
        for _ in range(50):
            M, N, K = gen.generate_gemm_shape(size_category='small', noise_enabled=True)
            assert 64 <= M <= 1024
            assert 64 <= N <= 1024
            assert 64 <= K <= 1024

    def test_generate_gemm_medium(self):
        """Medium shapes are generated correctly."""
        gen = BucketedShapeGenerator()
        for _ in range(50):
            M, N, K = gen.generate_gemm_shape(size_category='medium', noise_enabled=True)
            assert M >= 512
            assert N >= 512
            assert K >= 512

    def test_generate_gemm_large(self):
        """Large shapes are generated correctly."""
        gen = BucketedShapeGenerator()
        for _ in range(50):
            M, N, K = gen.generate_gemm_shape(size_category='large', noise_enabled=True)
            assert M >= 4096
            assert N >= 4096
            assert K >= 4096

    def test_noise_disabled(self):
        """Noise can be disabled for exact shapes."""
        gen = BucketedShapeGenerator()
        for _ in range(20):
            M, N, K = gen.generate_gemm_shape(size_category='small', noise_enabled=False)
            assert M in [128, 256, 512]
            assert N in [128, 256, 512]
            assert K in [128, 256, 512]

    def test_different_runs_different_shapes(self):
        """Different runs produce different shapes (with noise)."""
        gen = BucketedShapeGenerator()
        shapes = set()
        for _ in range(100):
            shape = gen.generate_gemm_shape(size_category='mixed', noise_enabled=True)
            shapes.add(shape)
        # With noise, we should get at least 10 unique shapes out of 100
        assert len(shapes) >= 10, f"Only got {len(shapes)} unique shapes"

    def test_generate_conv_shape(self):
        """Conv shapes are generated correctly."""
        gen = BucketedShapeGenerator()
        shape = gen.generate_conv_shape()
        assert 'batch' in shape
        assert 'in_channels' in shape
        assert 'out_channels' in shape
        assert 'kernel_size' in shape
        assert shape['batch'] >= 1
        assert shape['in_channels'] > 0

    def test_generate_attention_shape(self):
        """Attention shapes are generated correctly."""
        gen = BucketedShapeGenerator()
        shape = gen.generate_attention_shape()
        assert 'batch' in shape
        assert 'seq_len' in shape
        assert 'num_heads' in shape
        assert 'head_dim' in shape
        assert shape['seq_len'] > 0
        assert shape['num_heads'] > 0


class TestTensorLayoutRandomizer:
    """Test the tensor layout randomizer."""

    def test_create_randomizer(self):
        rand = TensorLayoutRandomizer()
        assert rand is not None

    def test_randomize_contiguous(self):
        """Contiguous tensor stays contiguous or is randomized."""
        import torch
        rand = TensorLayoutRandomizer()
        x = torch.randn(128, 256)
        result = rand.randomize_layout(x)
        assert result.shape == x.shape or result.shape[0] == x.shape[0]

    def test_generate_random_strides(self):
        """Random strides are generated correctly."""
        rand = TensorLayoutRandomizer()
        strides = rand.generate_random_strides((128, 256, 64))
        assert len(strides) == 3

    def test_randomize_contiguity(self):
        """Contiguity randomizer works."""
        import torch
        rand = TensorLayoutRandomizer()
        x = torch.randn(64, 128)
        result = rand.randomize_contiguity(x)
        assert result.shape == x.shape
