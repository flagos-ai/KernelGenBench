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

# Examples

Common LLM Track use cases.

## Quick Verification

Test that everything is working correctly:

```bash
python scripts/generate_kernel_and_verify.py \
    --op-name aten::add \
    --single-test \
    --server-type openai \
    --model-name gpt-4o \
    --max-rounds 1
```

## Pass@1 Evaluation

Evaluate single-shot generation:

```bash
python scripts/generate_kernel_and_verify.py \
    --server-type openai \
    --model-name gpt-4o \
    --max-rounds 1 \
    --temperature 0
```

## Pass@5 Evaluation

Evaluate best-of-5 generation:

```bash
python scripts/generate_kernel_and_verify.py \
    --server-type openai \
    --model-name gpt-4o \
    --max-rounds 5 \
    --temperature 0.8
```

## Cross-Platform Testing

On non-NVIDIA hardware:

```bash
# Dataset automatically set to KernelGenBench-aten
python scripts/generate_kernel_and_verify.py \
    --server-type openai \
    --model-name gpt-4o
```

## Specific Operator Families

Test all GEMM variants:

```bash
python scripts/generate_kernel_and_verify.py \
    --dataset KernelGenBench-cublas \
    --server-type openai \
    --model-name gpt-4o
```

## Debugging and Iteration

Start with debug mode:

```bash
# Test 8 operators
python scripts/generate_kernel_and_verify.py \
    --debug \
    --server-type openai

# If successful, run full benchmark
python scripts/generate_kernel_and_verify.py \
    --server-type openai
```

## Analyzing Results

```bash
python scripts/analyze/analyze.py output/pass_at_k/<run_dir>/
```

## Expected Results

Based on KernelGenBench experiments (NVIDIA A100):

| Method | Accuracy (210 ops) |
|--------|-------------------|
| Pass@1 (Opus-4.6) | 41% |
| Pass@5 (Opus-4.6) | 57% |
| Pass@1 (GPT-4o) | ~35% |
| Pass@5 (GPT-4o) | ~50% |
