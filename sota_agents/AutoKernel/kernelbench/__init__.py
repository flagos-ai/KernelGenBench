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
AutoKernel KernelBench Integration.

Bridge between AutoKernel's iterative optimization loop and the KernelBench
benchmark suite (ScalingIntelligence/KernelBench). Enables running 50-300+
refinement experiments per problem instead of one-shot LLM generation.

Components:
    bridge.py   -- Load, cache, and set up KernelBench problems
    bench_kb.py -- Evaluate ModelNew vs Model (correctness + speedup)
    scorer.py   -- Batch scoring across levels, compute fast_p metric
"""

__version__ = "1.0.0"
