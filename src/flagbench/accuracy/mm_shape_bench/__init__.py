"""
MM Shape-Specific Benchmark — 实验用两道题

测试两个"本质相同"的 shape 是否生成不同 kernel。
"""

from .test_mm_128x768_768x768 import test_accuracy_mm_128x768_768x768
from .test_mm_128x1024_1024x1024 import test_accuracy_mm_128x1024_1024x1024

__all__ = [
    'test_accuracy_mm_128x768_768x768',
    'test_accuracy_mm_128x1024_1024x1024',
]
