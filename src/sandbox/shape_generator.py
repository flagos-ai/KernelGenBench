"""
Bucketed Random Shape Generator for KernelGenBench.

Generates GPU-alignment-friendly random shapes with controlled noise.
Prevents shape-specific kernel caching while maintaining warp alignment.
"""
import random
from typing import Tuple, List, Optional, Dict
from dataclasses import dataclass


@dataclass
class ShapeBucket:
    base: int
    noise_range: Tuple[int, int]  # (min, max) offset
    alignment: int


class BucketedShapeGenerator:
    """Generates random shapes within GPU-friendly buckets.

    Key principle: random enough to prevent shape-specific caching,
    but aligned enough to not break warp/tile optimization paths.
    """

    # Standard shape buckets per size category
    STANDARD_BUCKETS = {
        "small": [
            ShapeBucket(128, (-8, 8), 32),
            ShapeBucket(256, (-16, 16), 32),
            ShapeBucket(512, (-32, 32), 64),
        ],
        "medium": [
            ShapeBucket(1024, (-64, 64), 128),
            ShapeBucket(2048, (-128, 128), 128),
            ShapeBucket(4096, (-256, 256), 256),
        ],
        "large": [
            ShapeBucket(8192, (-512, 512), 256),
            ShapeBucket(16384, (-1024, 1024), 512),
        ],
    }

    # Specialized GEMM shapes
    GEMM_OPTIONS = {
        "M": [128, 256, 512, 1024, 2048, 4096, 8192],
        "N": [128, 256, 512, 1024, 2048, 4096, 8192],
        "K": [128, 256, 512, 1024, 2048, 4096, 8192],
    }

    # Attention shapes
    ATTENTION_OPTIONS = {
        "seq_len": [128, 256, 512, 1024, 2048, 4096],
        "head_dim": [64, 128, 256],
        "num_heads": [8, 12, 16, 32],
        "batch_size": [1, 2, 4, 8, 16, 32],
    }

    # Conv shapes
    CONV_OPTIONS = {
        "spatial": [28, 56, 112, 224],
        "channels": [64, 128, 256, 512, 1024],
        "kernel": [1, 3, 5, 7],
    }

    # Noise offsets per size category
    NOISE_OFFSETS = {
        "small": [-32, -16, 0, 16, 32],
        "medium": [-128, -64, 0, 64, 128],
        "large": [-512, -256, 0, 256, 512],
    }

    def __init__(self, seed: int = None):
        self._rng = random.Random(seed)

    def _apply_noise(self, value: int, alignment: int,
                     size_category: str = "medium") -> int:
        """Apply random noise while maintaining alignment."""
        offsets = self.NOISE_OFFSETS.get(size_category, [-64, -32, 0, 32, 64])
        noise = self._rng.choice(offsets)
        result = value + noise
        # Ensure positivity and alignment
        result = max(alignment, result)
        result = (result // alignment) * alignment
        return result

    def generate_gemm_shape(self,
                            size_category: str = None,
                            noise_enabled: bool = True) -> Tuple[int, int, int]:
        """Generate a (M, N, K) GEMM shape.

        Args:
            size_category: 'small', 'medium', 'large', or None (random)
            noise_enabled: if True, adds random noise within bucket
        """
        if size_category is None:
            size_category = self._rng.choice(["small", "medium", "large"])

        M = self._rng.choice(self.GEMM_OPTIONS["M"])
        N = self._rng.choice(self.GEMM_OPTIONS["N"])
        K = self._rng.choice(self.GEMM_OPTIONS["K"])

        if noise_enabled:
            M = self._apply_noise(M, 64, size_category)
            N = self._apply_noise(N, 64, size_category)
            K = self._apply_noise(K, 64, size_category)

        return (M, N, K)

    def generate_attention_shape(self) -> Dict:
        """Generate an attention operator shape."""
        return {
            "batch": self._rng.choice(self.ATTENTION_OPTIONS["batch_size"]),
            "seq_len": self._rng.choice(self.ATTENTION_OPTIONS["seq_len"]),
            "num_heads": self._rng.choice(self.ATTENTION_OPTIONS["num_heads"]),
            "head_dim": self._rng.choice(self.ATTENTION_OPTIONS["head_dim"]),
        }

    def generate_conv_shape(self) -> Dict:
        """Generate a convolution operator shape."""
        return {
            "batch": self._rng.randint(1, 32),
            "in_channels": self._rng.choice(self.CONV_OPTIONS["channels"]),
            "out_channels": self._rng.choice(self.CONV_OPTIONS["channels"]),
            "spatial": self._rng.choice(self.CONV_OPTIONS["spatial"]),
            "kernel_size": self._rng.choice(self.CONV_OPTIONS["kernel"]),
        }

    def generate_shapes(self, op_type: str, count: int,
                        size_category: str = None) -> List:
        """Generate multiple shapes for a given operator type."""
        shapes = []
        for _ in range(count):
            if op_type == "gemm":
                shapes.append(self.generate_gemm_shape(size_category))
            elif op_type == "attention":
                shapes.append(self.generate_attention_shape())
            elif op_type == "conv":
                shapes.append(self.generate_conv_shape())
        return shapes


class TensorLayoutRandomizer:
    """Randomizes tensor layout (stride/contiguity) to prevent kernel cache hits.

    Triton kernel cache keys include stride patterns. Randomizing these
    ensures kernels get freshly compiled for each test.
    """

    LAYOUTS = ["contiguous", "strided", "transposed"]

    def __init__(self, seed: int = None):
        self._rng = random.Random(seed)

    def randomize_2d(self, tensor) -> "torch.Tensor":
        """Apply random 2D layout to a tensor."""
        import torch
        layout = self._rng.choice(self.LAYOUTS)

        if layout == "contiguous":
            return tensor.contiguous()

        elif layout == "strided":
            M, N = tensor.shape
            stride_0 = N + self._rng.randint(1, 32)
            t = torch.empty_strided((M, N), (stride_0, 1),
                                    dtype=tensor.dtype, device=tensor.device)
            t.copy_(tensor)
            return t

        elif layout == "transposed":
            return tensor.t().contiguous().t()

        return tensor

    def randomize_contiguity(self, tensor, prob: float = 0.3) -> "torch.Tensor":
        """With given probability, create a non-contiguous view."""
        if self._rng.random() < prob and tensor.dim() == 2:
            M, N = tensor.shape
            pad = self._rng.randint(1, 64)
            padded = torch.zeros(M, N + pad, dtype=tensor.dtype,
                                device=tensor.device)
            padded[:, :N].copy_(tensor)
            return padded[:, :N]
        return tensor
