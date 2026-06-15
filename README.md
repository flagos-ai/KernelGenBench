[<img width="2182" height="602" alt="github+banner-20260130" src=".github/assets/banner-20260130.png" />](https://flagos.io/)

[[中文版](./README.zh-CN.md)|English]

<div align="right">
  <a href="https://www.linkedin.com/company/flagos-community" target="_blank">
    <img src="./docs/assets/Linkedin.png" alt="LinkedIn" width="32" height="32" />
  </a>
  <a href="https://www.youtube.com/@FlagOS_Official" target="_blank">
    <img src="./docs/assets/youtube.png" alt="YouTube" width="32" height="32" />
  </a>
  <a href="https://x.com/FlagOS_Official" target="_blank">
    <img src="./docs/assets/x.png" alt="X" width="32" height="32" />
  </a>
  <a href="https://www.facebook.com/flagosglobalcommunity" target="_blank">
    <img src="./docs/assets/Facebook.png" alt="Facebook" width="32" height="32" />
  </a>
  <a href="https://discord.com/invite/ubqGuFMTNE" target="_blank">
    <img src="./docs/assets/discord.png" alt="Discord" width="32" height="32" />
  </a>
</div>

## Overview

KernelGenBench is a component of [FlagOS](https://flagos.io/) — a unified, open-source AI system software stack that fosters an open technology ecosystem by seamlessly integrating various models, systems, and chips. Following the principle of "develop once, migrate across various chips", FlagOS aims to unlock the full computational potential of hardware, break down barriers between different chip software stacks, and effectively reduce migration costs.

KernelGenBench is a benchmark framework for evaluating LLM and agent-based Triton kernel generation across multiple hardware platforms.

![KernelGenBench Overview](docs/_static/images/figures/overview.png)

## Features

- **210 operators** across three sources: ATen (110), vLLM (50), cuBLAS (50)
- **Multi-chip support**: NVIDIA, Ascend NPU, MUSA, Hygon DCU, Iluvatar, MetaX
- **Two evaluation tracks**: LLM Track (Pass@K) and Agent Track (iterative generation)
- **Multiple agent methods**: Claude Code, OpenCode, AutoKernel, AKO4ALL, cuda-optimized-skill
- **Automatic verification**: accuracy testing with tolerance-based comparison

## Quick Start

```bash
# NVIDIA platform
pip install -r requirements/requirements_nvidia.txt
pip install -e .

# Test single operator
python scripts/generate_kernel_and_verify.py \
    --op-name aten::add \
    --single-test \
    --server-type openai
```

👉 **For detailed setup, see [Getting Started](docs/source/getting-started/index.md).**

## Documentation

📚 **Full documentation**: [docs/source/](docs/source/)

| Section | Description |
|---------|-------------|
| [Overview](docs/source/overview/index.md) | What is KernelGenBench and why use it |
| [Getting Started](docs/source/getting-started/index.md) | Installation for all platforms |
| [LLM Track](docs/source/operation-guide/llm-track/index.md) | Pass@K evaluation guide |
| [Agent Track](docs/source/operation-guide/agent-track/index.md) | Agent-based evaluation guide |
| [Reference](docs/source/reference/index.md) | Datasets, operators, hardware |
| [Development](docs/source/development/index.md) | Contributing and extending |
| [FAQ](docs/source/faq/index.md) | Common questions |

## Related Projects

| Project | Description |
|---------|-------------|
| [awesome-LLM-driven-kernel-generation](https://github.com/flagos-ai/awesome-LLM-driven-kernel-generation) | Survey of AI-driven kernel generation |
| [KernelGen](https://github.com/flagos-ai/KernelGen) | High-performance platform for automated Triton kernel generation |

## Citation

```bibtex
@software{kernelgenbench2026,
  title={KernelGenBench: A Benchmark for LLM and Agent-Based Triton Kernel Generation},
  author={KernelGen Team},
  url={https://github.com/flagos-ai/KernelGenBench},
  year={2026}
}
```

## License

Apache 2.0 License