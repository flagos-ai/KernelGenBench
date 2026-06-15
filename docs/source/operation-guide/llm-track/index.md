# LLM Track

LLM Track evaluates LLMs on direct kernel generation with Pass@K metric.

```{toctree}
:maxdepth: 1

commands
parameters
examples
```

## What It Tests

Base model capability to generate GPU kernels without execution feedback.

## When to Use

- Evaluating base model code generation
- Comparing different LLM providers
- Quick benchmark with lower cost

## Quick Start

```bash
python scripts/generate_kernel_and_verify.py \
    --op-name aten::add \
    --single-test \
    --server-type openai
```
