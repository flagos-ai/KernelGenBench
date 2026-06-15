# Commands

CLI commands for LLM Track evaluation.

## Basic Usage

### Single Operator Test

```bash
python scripts/generate_kernel_and_verify.py \
    --op-name aten::add \
    --single-test \
    --server-type openai \
    --model-name gpt-4o \
    --max-rounds 3
```

### Full Benchmark

```bash
python scripts/generate_kernel_and_verify.py \
    --server-type openai \
    --model-name gpt-4o \
    --max-rounds 10
```

## Dataset Selection

### Full Dataset (NVIDIA)

```bash
python scripts/generate_kernel_and_verify.py \
    --dataset KernelGenBench \
    --server-type openai \
    --model-name gpt-4o
```

### ATen Only (All Platforms)

```bash
python scripts/generate_kernel_and_verify.py \
    --dataset KernelGenBench-aten \
    --server-type openai \
    --model-name gpt-4o
```

### Specific Operator Sources

```bash
# vLLM operators only
python scripts/generate_kernel_and_verify.py \
    --dataset KernelGenBench-vllm \
    --server-type openai

# cuBLAS operators only
python scripts/generate_kernel_and_verify.py \
    --dataset KernelGenBench-cublas \
    --server-type openai
```

## Server Types

### OpenAI

```bash
python scripts/generate_kernel_and_verify.py \
    --server-type openai \
    --model-name gpt-4o
```

### Anthropic

```bash
python scripts/generate_kernel_and_verify.py \
    --server-type anthropic \
    --model-name claude-opus-4-6
```

## Advanced Options

### Enable Reflection

Enable feedback from previous rounds:

```bash
python scripts/generate_kernel_and_verify.py \
    --server-type openai \
    --model-name gpt-4o \
    --reflection
```

### Resume from Checkpoint

```bash
python scripts/generate_kernel_and_verify.py \
    --resume-from output/pass_at_k/previous_run/
```

### Debug Mode

Test with only 8 operators:

```bash
python scripts/generate_kernel_and_verify.py \
    --debug \
    --server-type openai
```
