# Getting Started

This guide helps you quickly set up {term}`KernelGenBench` and run your first evaluation.

## Prerequisites

Before installing {term}`KernelGenBench`, ensure you have:

- Python 3.8 or higher
- {term}`CUDA` 11.0+ (for NVIDIA platforms)
- API credentials for your chosen {term}`LLM` provider

## Installation

### NVIDIA

```bash
git clone https://github.com/flagos-ai/KernelGenBench.git
cd KernelGenBench
pip install -r requirements/requirements_nvidia.txt
pip install -e .
```

> `vllm==0.13.0` will automatically install compatible versions of torch and triton.

### Domestic Chips (Ascend / MUSA / Hygon / Iluvatar / MetaX)

On domestic chips, torch and the chip-specific runtime (e.g., torch_npu, torch_musa) are pre-installed in the vendor container image. Use the vendor-provided Docker image to start a container, then install KernelGenBench inside it:

```bash
# Start the vendor container (example for Ascend NPU)
docker run -it --rm --network host \
    --device=/dev/davinci0 --device=/dev/davinci_manager \
    ascend/pytorch:latest bash

# Inside the container, clone and install
git clone https://github.com/flagos-ai/KernelGenBench.git
cd KernelGenBench
pip install -r requirements/requirements_ascend.txt
pip install -e .

# For other chips, replace the requirements file:
#   Hygon DCU:  requirements/requirements_hygon.txt
#   MUSA:       requirements/requirements_musa.txt
#   Iluvatar:   requirements/requirements_iluvatar.txt
#   MetaX:      requirements/requirements_metax.txt
```

> **Note**: Do NOT install vllm on non-NVIDIA platforms — it is NVIDIA-only.

### Configure API Credentials

Set up your {term}`LLM` provider credentials:

```bash
# Anthropic Claude
export ANTHROPIC_API_KEY=your_key

# OpenAI / OpenAI-compatible
export OPENAI_API_KEY=your_key
export OPENAI_BASE_URL=http://your-endpoint/v1  # Optional, for custom endpoints
```

### Step 3: Install Claude Code CLI (for Agent Track)

If you plan to use the Agent Track, install the Claude Code CLI:

```bash
npm install -g @anthropic-ai/claude-code
```

## Verify Installation

Test that {term}`KernelGenBench` is installed correctly:

```bash
python -c "import kernelgenbench; print('KernelGenBench installed successfully')"
```

## Run Your First Evaluation

### Quick Test (Single Operator)

Test with a single {term}`Operator` to verify everything works:

```bash
python scripts/generate_kernel_and_verify.py \
    --op-name aten::add \
    --single-test \
    --server-type openai \
    --model-name gpt-4o \
    --max-rounds 3
```

### Full Evaluation

Run a complete evaluation on all operators:

```bash
# Full evaluation (210 operators)
python scripts/generate_kernel_and_verify.py \
    --server-type openai \
    --model-name gpt-4o \
    --max-rounds 10

# Non-NVIDIA chips (ATen only, 110 operators)
python scripts/generate_kernel_and_verify.py \
    --dataset {term}`KernelGenBench-aten` \
    --server-type openai \
    --model-name gpt-4o \
    --max-rounds 10
```

## Datasets

| Dataset | Operators | Description |
|---------|-----------|-------------|
| `{term}`KernelGenBench`` | 210 | Full set ({term}`ATen` + {term}`vLLM` + {term}`cuBLAS`) |
| `{term}`KernelGenBench-aten`` | 110 | {term}`ATen` operators only |
| `{term}`KernelGenBench-vllm`` | 50 | {term}`vLLM` operators only (NVIDIA only) |
| `{term}`KernelGenBench-cublas`` | 50 | {term}`cuBLAS` operators only (NVIDIA only) |

```{note}
On non-NVIDIA chips, the default dataset is automatically set to {term}`KernelGenBench-aten` because {term}`vLLM` and {term}`cuBLAS` operators require NVIDIA GPUs.
```

## Hardware Detection

{term}`KernelGenBench` automatically detects your hardware platform:

```bash
# Check detected device
python -c "from kernelgenbench.runtime import get_device_type; print(get_device_type())"
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError: No module named 'kernelgenbench'` | Run `pip install -e .` in the project root |
| {term}`CUDA` out of memory | Reduce `--device-count` or use smaller batch sizes |
| API authentication errors | Verify your API keys are set correctly |
| {term}`vLLM` installation conflicts on non-NVIDIA platforms | Do not install {term}`vLLM`; use vendor container images |

## Next Steps

- [Overview](../overview/index.md) - Learn what {term}`KernelGenBench` is
- [Features](../features/index.md) - Explore all features
- [LLM Track](../operation-guide/llm-track/index.md) - Detailed LLM evaluation
- [Agent Track](../operation-guide/agent-track/index.md) - Detailed Agent evaluation
- [FAQ](../faq/index.md) - Frequently asked questions