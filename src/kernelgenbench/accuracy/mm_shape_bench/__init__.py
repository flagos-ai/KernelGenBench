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
MM Shape-Specific Benchmark — 50 shapes x 2 dtypes = 100 test cases.

Each test pins a (shape, dtype) and evaluates whether the agent can write
an optimal Triton kernel for that specific matmul configuration.
Shapes sourced from 2907 HF model training traces (aten.mm.default).
"""

from .test_mm_128x768_768x768_f32 import test_accuracy_mm_128x768_768x768_f32
from .test_mm_128x768_768x768_f16 import test_accuracy_mm_128x768_768x768_f16
from .test_mm_128x1024_1024x1024_f32 import test_accuracy_mm_128x1024_1024x1024_f32
from .test_mm_128x1024_1024x1024_f16 import test_accuracy_mm_128x1024_1024x1024_f16
from .test_mm_128x1024_1024x4096_f32 import test_accuracy_mm_128x1024_1024x4096_f32
from .test_mm_128x1024_1024x4096_f16 import test_accuracy_mm_128x1024_1024x4096_f16
from .test_mm_128x4096_4096x4096_f32 import test_accuracy_mm_128x4096_4096x4096_f32
from .test_mm_128x4096_4096x4096_f16 import test_accuracy_mm_128x4096_4096x4096_f16
from .test_mm_128x768_768x3072_f32 import test_accuracy_mm_128x768_768x3072_f32
from .test_mm_128x768_768x3072_f16 import test_accuracy_mm_128x768_768x3072_f16
from .test_mm_128x3072_3072x768_f32 import test_accuracy_mm_128x3072_3072x768_f32
from .test_mm_128x3072_3072x768_f16 import test_accuracy_mm_128x3072_3072x768_f16
from .test_mm_128x4096_4096x1024_f32 import test_accuracy_mm_128x4096_4096x1024_f32
from .test_mm_128x4096_4096x1024_f16 import test_accuracy_mm_128x4096_4096x1024_f16
from .test_mm_128x14336_14336x4096_f32 import test_accuracy_mm_128x14336_14336x4096_f32
from .test_mm_128x14336_14336x4096_f16 import test_accuracy_mm_128x14336_14336x4096_f16
from .test_mm_128x384_384x384_f32 import test_accuracy_mm_128x384_384x384_f32
from .test_mm_128x384_384x384_f16 import test_accuracy_mm_128x384_384x384_f16
from .test_mm_128x2304_2304x768_f32 import test_accuracy_mm_128x2304_2304x768_f32
from .test_mm_128x2304_2304x768_f16 import test_accuracy_mm_128x2304_2304x768_f16
from .test_mm_128x4096_4096x14336_f32 import test_accuracy_mm_128x4096_4096x14336_f32
from .test_mm_128x4096_4096x14336_f16 import test_accuracy_mm_128x4096_4096x14336_f16
from .test_mm_128x768_768x1152_f32 import test_accuracy_mm_128x768_768x1152_f32
from .test_mm_128x768_768x1152_f16 import test_accuracy_mm_128x768_768x1152_f16
from .test_mm_128x2048_2048x2048_f32 import test_accuracy_mm_128x2048_2048x2048_f32
from .test_mm_128x2048_2048x2048_f16 import test_accuracy_mm_128x2048_2048x2048_f16
from .test_mm_128x5120_5120x5120_f32 import test_accuracy_mm_128x5120_5120x5120_f32
from .test_mm_128x5120_5120x5120_f16 import test_accuracy_mm_128x5120_5120x5120_f16
from .test_mm_128x1024_1024x5120_f32 import test_accuracy_mm_128x1024_1024x5120_f32
from .test_mm_128x1024_1024x5120_f16 import test_accuracy_mm_128x1024_1024x5120_f16
from .test_mm_128x3072_3072x1024_f32 import test_accuracy_mm_128x3072_3072x1024_f32
from .test_mm_128x3072_3072x1024_f16 import test_accuracy_mm_128x3072_3072x1024_f16
from .test_mm_128x384_384x1536_f32 import test_accuracy_mm_128x384_384x1536_f32
from .test_mm_128x384_384x1536_f16 import test_accuracy_mm_128x384_384x1536_f16
from .test_mm_128x1536_1536x384_f32 import test_accuracy_mm_128x1536_1536x384_f32
from .test_mm_128x1536_1536x384_f16 import test_accuracy_mm_128x1536_1536x384_f16
from .test_mm_768x512_512x768_f32 import test_accuracy_mm_768x512_512x768_f32
from .test_mm_768x512_512x768_f16 import test_accuracy_mm_768x512_512x768_f16
from .test_mm_128x1024_1024x2624_f32 import test_accuracy_mm_128x1024_1024x2624_f32
from .test_mm_128x1024_1024x2624_f16 import test_accuracy_mm_128x1024_1024x2624_f16
from .test_mm_128x5248_5248x1024_f32 import test_accuracy_mm_128x5248_5248x1024_f32
from .test_mm_128x5248_5248x1024_f16 import test_accuracy_mm_128x5248_5248x1024_f16
from .test_mm_1024x512_512x1024_f32 import test_accuracy_mm_1024x512_512x1024_f32
from .test_mm_1024x512_512x1024_f16 import test_accuracy_mm_1024x512_512x1024_f16
from .test_mm_128x18944_18944x3584_f32 import test_accuracy_mm_128x18944_18944x3584_f32
from .test_mm_128x18944_18944x3584_f16 import test_accuracy_mm_128x18944_18944x3584_f16
from .test_mm_128x3584_3584x3584_f32 import test_accuracy_mm_128x3584_3584x3584_f32
from .test_mm_128x3584_3584x3584_f16 import test_accuracy_mm_128x3584_3584x3584_f16
from .test_mm_128x512_512x3584_f32 import test_accuracy_mm_128x512_512x3584_f32
from .test_mm_128x512_512x3584_f16 import test_accuracy_mm_128x512_512x3584_f16
from .test_mm_128x14336_14336x5120_f32 import test_accuracy_mm_128x14336_14336x5120_f32
from .test_mm_128x14336_14336x5120_f16 import test_accuracy_mm_128x14336_14336x5120_f16
from .test_mm_128x192_192x192_f32 import test_accuracy_mm_128x192_192x192_f32
from .test_mm_128x192_192x192_f16 import test_accuracy_mm_128x192_192x192_f16
from .test_mm_128x13824_13824x5120_f32 import test_accuracy_mm_128x13824_13824x5120_f32
from .test_mm_128x13824_13824x5120_f16 import test_accuracy_mm_128x13824_13824x5120_f16
from .test_mm_128x2560_2560x2560_f32 import test_accuracy_mm_128x2560_2560x2560_f32
from .test_mm_128x2560_2560x2560_f16 import test_accuracy_mm_128x2560_2560x2560_f16
from .test_mm_128x11008_11008x4096_f32 import test_accuracy_mm_128x11008_11008x4096_f32
from .test_mm_128x11008_11008x4096_f16 import test_accuracy_mm_128x11008_11008x4096_f16
from .test_mm_128x256_256x2048_f32 import test_accuracy_mm_128x256_256x2048_f32
from .test_mm_128x256_256x2048_f16 import test_accuracy_mm_128x256_256x2048_f16
from .test_mm_768x128_128x768_f32 import test_accuracy_mm_768x128_128x768_f32
from .test_mm_768x128_128x768_f16 import test_accuracy_mm_768x128_128x768_f16
from .test_mm_128x9216_9216x2304_f32 import test_accuracy_mm_128x9216_9216x2304_f32
from .test_mm_128x9216_9216x2304_f16 import test_accuracy_mm_128x9216_9216x2304_f16
from .test_mm_128x1024_1024x2304_f32 import test_accuracy_mm_128x1024_1024x2304_f32
from .test_mm_128x1024_1024x2304_f16 import test_accuracy_mm_128x1024_1024x2304_f16
from .test_mm_128x5120_5120x4096_f32 import test_accuracy_mm_128x5120_5120x4096_f32
from .test_mm_128x5120_5120x4096_f16 import test_accuracy_mm_128x5120_5120x4096_f16
from .test_mm_128x4096_4096x5120_f32 import test_accuracy_mm_128x4096_4096x5120_f32
from .test_mm_128x4096_4096x5120_f16 import test_accuracy_mm_128x4096_4096x5120_f16
from .test_mm_128x512_512x2048_f32 import test_accuracy_mm_128x512_512x2048_f32
from .test_mm_128x512_512x2048_f16 import test_accuracy_mm_128x512_512x2048_f16
from .test_mm_128x3584_3584x18944_f32 import test_accuracy_mm_128x3584_3584x18944_f32
from .test_mm_128x3584_3584x18944_f16 import test_accuracy_mm_128x3584_3584x18944_f16
from .test_mm_128x5120_5120x14336_f32 import test_accuracy_mm_128x5120_5120x14336_f32
from .test_mm_128x5120_5120x14336_f16 import test_accuracy_mm_128x5120_5120x14336_f16
from .test_mm_128x512_512x4096_f32 import test_accuracy_mm_128x512_512x4096_f32
from .test_mm_128x512_512x4096_f16 import test_accuracy_mm_128x512_512x4096_f16
from .test_mm_128x6144_6144x768_f32 import test_accuracy_mm_128x6144_6144x768_f32
from .test_mm_128x6144_6144x768_f16 import test_accuracy_mm_128x6144_6144x768_f16
from .test_mm_1x768_768x768_f32 import test_accuracy_mm_1x768_768x768_f32
from .test_mm_1x768_768x768_f16 import test_accuracy_mm_1x768_768x768_f16
from .test_mm_768x1_1x768_f32 import test_accuracy_mm_768x1_1x768_f32
from .test_mm_768x1_1x768_f16 import test_accuracy_mm_768x1_1x768_f16
from .test_mm_128x5120_5120x13824_f32 import test_accuracy_mm_128x5120_5120x13824_f32
from .test_mm_128x5120_5120x13824_f16 import test_accuracy_mm_128x5120_5120x13824_f16
from .test_mm_128x1024_1024x3072_f32 import test_accuracy_mm_128x1024_1024x3072_f32
from .test_mm_128x1024_1024x3072_f16 import test_accuracy_mm_128x1024_1024x3072_f16
from .test_mm_128x256_256x768_f32 import test_accuracy_mm_128x256_256x768_f32
from .test_mm_128x256_256x768_f16 import test_accuracy_mm_128x256_256x768_f16
from .test_mm_128x1152_1152x768_f32 import test_accuracy_mm_128x1152_1152x768_f32
from .test_mm_128x1152_1152x768_f16 import test_accuracy_mm_128x1152_1152x768_f16
from .test_mm_128x4096_4096x11008_f32 import test_accuracy_mm_128x4096_4096x11008_f32
from .test_mm_128x4096_4096x11008_f16 import test_accuracy_mm_128x4096_4096x11008_f16
from .test_mm_128x11008_11008x2048_f32 import test_accuracy_mm_128x11008_11008x2048_f32
from .test_mm_128x11008_11008x2048_f16 import test_accuracy_mm_128x11008_11008x2048_f16
from .test_mm_128x1024_1024x2048_f32 import test_accuracy_mm_128x1024_1024x2048_f32
from .test_mm_128x1024_1024x2048_f16 import test_accuracy_mm_128x1024_1024x2048_f16

__all__ = [
    'test_accuracy_mm_128x768_768x768_f32',
    'test_accuracy_mm_128x768_768x768_f16',
    'test_accuracy_mm_128x1024_1024x1024_f32',
    'test_accuracy_mm_128x1024_1024x1024_f16',
    'test_accuracy_mm_128x1024_1024x4096_f32',
    'test_accuracy_mm_128x1024_1024x4096_f16',
    'test_accuracy_mm_128x4096_4096x4096_f32',
    'test_accuracy_mm_128x4096_4096x4096_f16',
    'test_accuracy_mm_128x768_768x3072_f32',
    'test_accuracy_mm_128x768_768x3072_f16',
    'test_accuracy_mm_128x3072_3072x768_f32',
    'test_accuracy_mm_128x3072_3072x768_f16',
    'test_accuracy_mm_128x4096_4096x1024_f32',
    'test_accuracy_mm_128x4096_4096x1024_f16',
    'test_accuracy_mm_128x14336_14336x4096_f32',
    'test_accuracy_mm_128x14336_14336x4096_f16',
    'test_accuracy_mm_128x384_384x384_f32',
    'test_accuracy_mm_128x384_384x384_f16',
    'test_accuracy_mm_128x2304_2304x768_f32',
    'test_accuracy_mm_128x2304_2304x768_f16',
    'test_accuracy_mm_128x4096_4096x14336_f32',
    'test_accuracy_mm_128x4096_4096x14336_f16',
    'test_accuracy_mm_128x768_768x1152_f32',
    'test_accuracy_mm_128x768_768x1152_f16',
    'test_accuracy_mm_128x2048_2048x2048_f32',
    'test_accuracy_mm_128x2048_2048x2048_f16',
    'test_accuracy_mm_128x5120_5120x5120_f32',
    'test_accuracy_mm_128x5120_5120x5120_f16',
    'test_accuracy_mm_128x1024_1024x5120_f32',
    'test_accuracy_mm_128x1024_1024x5120_f16',
    'test_accuracy_mm_128x3072_3072x1024_f32',
    'test_accuracy_mm_128x3072_3072x1024_f16',
    'test_accuracy_mm_128x384_384x1536_f32',
    'test_accuracy_mm_128x384_384x1536_f16',
    'test_accuracy_mm_128x1536_1536x384_f32',
    'test_accuracy_mm_128x1536_1536x384_f16',
    'test_accuracy_mm_768x512_512x768_f32',
    'test_accuracy_mm_768x512_512x768_f16',
    'test_accuracy_mm_128x1024_1024x2624_f32',
    'test_accuracy_mm_128x1024_1024x2624_f16',
    'test_accuracy_mm_128x5248_5248x1024_f32',
    'test_accuracy_mm_128x5248_5248x1024_f16',
    'test_accuracy_mm_1024x512_512x1024_f32',
    'test_accuracy_mm_1024x512_512x1024_f16',
    'test_accuracy_mm_128x18944_18944x3584_f32',
    'test_accuracy_mm_128x18944_18944x3584_f16',
    'test_accuracy_mm_128x3584_3584x3584_f32',
    'test_accuracy_mm_128x3584_3584x3584_f16',
    'test_accuracy_mm_128x512_512x3584_f32',
    'test_accuracy_mm_128x512_512x3584_f16',
    'test_accuracy_mm_128x14336_14336x5120_f32',
    'test_accuracy_mm_128x14336_14336x5120_f16',
    'test_accuracy_mm_128x192_192x192_f32',
    'test_accuracy_mm_128x192_192x192_f16',
    'test_accuracy_mm_128x13824_13824x5120_f32',
    'test_accuracy_mm_128x13824_13824x5120_f16',
    'test_accuracy_mm_128x2560_2560x2560_f32',
    'test_accuracy_mm_128x2560_2560x2560_f16',
    'test_accuracy_mm_128x11008_11008x4096_f32',
    'test_accuracy_mm_128x11008_11008x4096_f16',
    'test_accuracy_mm_128x256_256x2048_f32',
    'test_accuracy_mm_128x256_256x2048_f16',
    'test_accuracy_mm_768x128_128x768_f32',
    'test_accuracy_mm_768x128_128x768_f16',
    'test_accuracy_mm_128x9216_9216x2304_f32',
    'test_accuracy_mm_128x9216_9216x2304_f16',
    'test_accuracy_mm_128x1024_1024x2304_f32',
    'test_accuracy_mm_128x1024_1024x2304_f16',
    'test_accuracy_mm_128x5120_5120x4096_f32',
    'test_accuracy_mm_128x5120_5120x4096_f16',
    'test_accuracy_mm_128x4096_4096x5120_f32',
    'test_accuracy_mm_128x4096_4096x5120_f16',
    'test_accuracy_mm_128x512_512x2048_f32',
    'test_accuracy_mm_128x512_512x2048_f16',
    'test_accuracy_mm_128x3584_3584x18944_f32',
    'test_accuracy_mm_128x3584_3584x18944_f16',
    'test_accuracy_mm_128x5120_5120x14336_f32',
    'test_accuracy_mm_128x5120_5120x14336_f16',
    'test_accuracy_mm_128x512_512x4096_f32',
    'test_accuracy_mm_128x512_512x4096_f16',
    'test_accuracy_mm_128x6144_6144x768_f32',
    'test_accuracy_mm_128x6144_6144x768_f16',
    'test_accuracy_mm_1x768_768x768_f32',
    'test_accuracy_mm_1x768_768x768_f16',
    'test_accuracy_mm_768x1_1x768_f32',
    'test_accuracy_mm_768x1_1x768_f16',
    'test_accuracy_mm_128x5120_5120x13824_f32',
    'test_accuracy_mm_128x5120_5120x13824_f16',
    'test_accuracy_mm_128x1024_1024x3072_f32',
    'test_accuracy_mm_128x1024_1024x3072_f16',
    'test_accuracy_mm_128x256_256x768_f32',
    'test_accuracy_mm_128x256_256x768_f16',
    'test_accuracy_mm_128x1152_1152x768_f32',
    'test_accuracy_mm_128x1152_1152x768_f16',
    'test_accuracy_mm_128x4096_4096x11008_f32',
    'test_accuracy_mm_128x4096_4096x11008_f16',
    'test_accuracy_mm_128x11008_11008x2048_f32',
    'test_accuracy_mm_128x11008_11008x2048_f16',
    'test_accuracy_mm_128x1024_1024x2048_f32',
    'test_accuracy_mm_128x1024_1024x2048_f16',
]
