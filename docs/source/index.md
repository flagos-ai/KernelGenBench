# KernelGenBench Documentation

A benchmark framework for evaluating {term}`LLM` and agent-based {term}`Triton` kernel generation.

```{toctree}
:maxdepth: 2
:caption: Contents

overview/index
getting-started/index
features/index
operation-guide/index
reference/index
development/index
faq/index
glossary/index
```

## Quick Links

| Resource | Description |
|----------|-------------|
| [Getting Started](getting-started/index.md) | Install and run your first benchmark |
| [LLM Track](operation-guide/llm-track/index.md) | {term}`Pass@K` evaluation |
| [Agent Track](operation-guide/agent-track/index.md) | {term}`Agent`-based evaluation |
| [FAQ](faq/index.md) | Common questions |
| [Glossary](glossary/index.md) | Technical terminology |

## About

{term}`KernelGenBench` is a component of [FlagOS](https://flagos.io/) — a unified, open-source AI system software stack.

### Key Features

- **210 operators** across {term}`ATen`, {term}`vLLM`, and {term}`cuBLAS`
- **6 hardware platforms** (NVIDIA, Ascend, MUSA, Hygon, Iluvatar, MetaX)
- **Two evaluation tracks**: LLM Track and {term}`Agent` Track
- **Automatic verification**: accuracy testing with tolerance-based comparison

## Citation

```bibtex
@software{kernelgenbench2026,
  title={KernelGenBench: A Benchmark for LLM and Agent-Based Triton Kernel Generation},
  author={KernelGen Team},
  url={https://github.com/flagos-ai/KernelGenBench},
  year={2026}
}
```
