"""
Layer 4: Bucketed Random Shape (replaces fully random primes).
From: triton_competition_anti_cheat_guide.md - Section 6
"""
import random
from typing import List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class ShapeBucket:
    """Shape分桶配置"""
    base: int          # 基准值
    noise_range: Tuple[int, int]  # 噪声范围
    alignment: int     # 对齐要求


class BucketedShapeGenerator:
    """分桶随机Shape生成器"""

    # 标准分桶配置（GPU友好）
    STANDARD_BUCKETS = {
        'small': [
            ShapeBucket(128, (-8, 8), 32),
            ShapeBucket(256, (-16, 16), 32),
            ShapeBucket(512, (-32, 32), 64),
        ],
        'medium': [
            ShapeBucket(1024, (-64, 64), 128),
            ShapeBucket(2048, (-128, 128), 128),
            ShapeBucket(4096, (-256, 256), 256),
        ],
        'large': [
            ShapeBucket(8192, (-512, 512), 256),
            ShapeBucket(16384, (-1024, 1024), 512),
        ],
    }

    # 特殊分桶（针对特定算子）
    GEMM_BUCKETS = {
        'M': [128, 256, 512, 1024, 2048, 4096, 8192],
        'N': [128, 256, 512, 1024, 2048, 4096, 8192],
        'K': [128, 256, 512, 1024, 2048, 4096, 8192],
    }

    def __init__(self, bucket_type: str = 'standard'):
        self.bucket_type = bucket_type

    def generate_gemm_shape(self,
                           size_category: str = 'mixed',
                           noise_enabled: bool = True) -> Tuple[int, int, int]:
        """
        生成GEMM Shape

        Args:
            size_category: 'small', 'medium', 'large', 'mixed'
            noise_enabled: 是否添加噪声
        """
        if size_category == 'mixed':
            # 随机选择大小类别
            size_category = random.choice(['small', 'medium', 'large'])

        # 选择基准值
        if size_category == 'small':
            M = random.choice([128, 256, 512])
            N = random.choice([128, 256, 512])
            K = random.choice([128, 256, 512])
        elif size_category == 'medium':
            M = random.choice([1024, 2048, 4096])
            N = random.choice([1024, 2048, 4096])
            K = random.choice([1024, 2048, 4096])
        else:  # large
            M = random.choice([8192, 16384])
            N = random.choice([8192, 16384])
            K = random.choice([8192, 16384])

        # 添加噪声（保持对齐）
        if noise_enabled:
            noise_m = random.choice([-128, -64, -32, 0, 32, 64, 128])
            noise_n = random.choice([-128, -64, -32, 0, 32, 64, 128])
            noise_k = random.choice([-128, -64, -32, 0, 32, 64, 128])

            M = max(64, M + noise_m)
            N = max(64, N + noise_n)
            K = max(64, K + noise_k)

        return (M, N, K)

    def generate_conv_shape(self,
                           batch_range: Tuple[int, int] = (1, 32),
                           channel_range: Tuple[int, int] = (64, 512),
                           spatial_range: Tuple[int, int] = (28, 224)) -> dict:
        """生成Conv Shape"""
        batch = random.randint(*batch_range)
        in_channels = random.choice([64, 128, 256, 512, 768, 1024])
        out_channels = random.choice([64, 128, 256, 512, 768, 1024])
        height = random.choice([28, 56, 112, 224])
        width = random.choice([28, 56, 112, 224])
        kernel_size = random.choice([1, 3, 5, 7])

        return {
            'batch': batch,
            'in_channels': in_channels,
            'out_channels': out_channels,
            'height': height,
            'width': width,
            'kernel_size': kernel_size,
        }

    def generate_attention_shape(self,
                                batch_range: Tuple[int, int] = (1, 32),
                                seq_len_options: List[int] = None,
                                head_dim_options: List[int] = None) -> dict:
        """生成Attention Shape"""
        if seq_len_options is None:
            seq_len_options = [128, 256, 512, 1024, 2048]
        if head_dim_options is None:
            head_dim_options = [64, 128, 256]

        batch = random.randint(*batch_range)
        seq_len = random.choice(seq_len_options)
        num_heads = random.choice([8, 12, 16, 32])
        head_dim = random.choice(head_dim_options)

        return {
            'batch': batch,
            'seq_len': seq_len,
            'num_heads': num_heads,
            'head_dim': head_dim,
        }


class TensorLayoutRandomizer:
    """张量Layout随机化器"""

    def __init__(self):
        self.layout_options = ['contiguous', 'strided', 'transposed']

    def randomize_layout(self, tensor: "torch.Tensor") -> "torch.Tensor":
        """
        随机化张量Layout

        Triton kernel cache key包含：
        - dtype
        - stride pattern
        - contiguity
        随机化可以防止kernel复用
        """
        import torch
        layout = random.choice(self.layout_options)

        if layout == 'contiguous':
            return tensor.contiguous()

        elif layout == 'strided':
            # 创建非连续stride
            if tensor.dim() == 2:
                M, N = tensor.shape
                # 故意创建非对齐stride
                stride_0 = N + random.randint(1, 32)
                stride_1 = 1
                new_tensor = torch.empty_strided(
                    (M, N), (stride_0, stride_1),
                    dtype=tensor.dtype, device=tensor.device
                )
                new_tensor.copy_(tensor)
                return new_tensor
            else:
                return tensor

        elif layout == 'transposed':
            if tensor.dim() == 2:
                return tensor.t().contiguous().t()
            return tensor

        return tensor

    def randomize_contiguity(self, tensor: "torch.Tensor") -> "torch.Tensor":
        """
        随机化连续性

        概率性添加padding或不连续stride
        """
        import torch
        if random.random() < 0.3:
            # 30%概率创建非连续张量
            if tensor.dim() == 2:
                M, N = tensor.shape
                padded_N = N + random.randint(1, 64)
                padded = torch.zeros(M, padded_N, dtype=tensor.dtype, device=tensor.device)
                padded[:, :N] = tensor
                return padded[:, :N]

        return tensor

    def generate_random_strides(self, shape: Tuple[int, ...]) -> Tuple[int, ...]:
        """生成随机stride"""
        strides = []
        current = 1
        for dim in reversed(shape):
            # 随机添加padding
            padding = random.randint(0, 32)
            strides.insert(0, current)
            current = current * (dim + padding)
        return tuple(strides)
