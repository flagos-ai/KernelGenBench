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
