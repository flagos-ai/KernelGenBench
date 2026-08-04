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

## API Formats

### OpenAI

```bash
python scripts/generate_kernel_and_verify.py \
    --api-format openai \
    --model-name gpt-4o
```

### Anthropic

```bash
python scripts/generate_kernel_and_verify.py \
    --api-format anthropic \
    --model-name claude-opus-4-6
```

### Compatible Endpoints

No provider registration is required. Select the endpoint's wire protocol and
pass its URL, key, and model directly:

```bash
python scripts/generate_kernel_and_verify.py \
    --api-format <openai|anthropic> \
    --model-name <model-name> \
    --base-url <api-endpoint> \
    --api-key <your-api-key>
```

The same values can be provided with `OPENAI_BASE_URL` /
`OPENAI_API_KEY` or `ANTHROPIC_BASE_URL` / `ANTHROPIC_API_KEY`.

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
