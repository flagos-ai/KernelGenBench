<!--
 Copyright 2026 FlagOS Contributors

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 -->

# Capabilities

{term}`KernelGenBench` provides comprehensive capabilities for {term}`Kernel` generation evaluation.

## Multi-Source Evaluation

210 operators from ATen, vLLM, and cuBLAS sources.

→ See [Multi-Source Operators](../features/multi-source.md) for details.

## Multi-Chip Support

6 hardware platforms (NVIDIA, Ascend, MUSA, Hygon, Iluvatar, MetaX) with auto-detection.

→ See [Multi-Chip Support](../features/multi-chip.md) for details.

## Two Evaluation Tracks

| Track | Purpose |
|-------|---------|
| [LLM Track](../operation-guide/llm-track/index.md) | {term}`Pass@K` evaluation |
| [Agent Track](../operation-guide/agent-track/index.md) | Iterative generation |

## Anti-Hack Validation

Three-tier mechanism (AST scan, Ghost replay, Hardware profiling).

→ See [Anti-Hack Architecture](../features/anti-hack.md) for details.

## Evaluation Metrics

Accuracy, speedup, token cost, wall time.

→ See [Evaluation Metrics](../features/metrics.md) for details.
