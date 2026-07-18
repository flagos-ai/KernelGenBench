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
