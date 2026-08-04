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

# Parameters

LLM Track command-line parameters.

## Required Parameters

| Parameter | Description |
|-----------|-------------|
| `--api-format` | API protocol: `openai` or `anthropic` |
| `--model-name` | Model identifier |

## Optional Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--op-name` | All | Test a single operator (e.g., `aten::add`) |
| `--single-test` | Off | Randomly select 1 operator for quick testing |
| `--base-url` | SDK default / Env var | API base URL for either compatible protocol |
| `--api-key` | Env var | API key (overrides `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` env var) |
| `--dataset` | Auto | Dataset: `KernelGenBench`, `KernelGenBench-aten`, `KernelGenBench-vllm`, `KernelGenBench-cublas` |
| `--max-rounds` | 10 | Number of Pass@K rounds |
| `--device-count` | 8 | Number of GPUs for verification |
| `--timeout` | 300 | Timeout per operator (seconds) |
| `--temperature` | 0.8 | Sampling temperature |
| `--reflection` | Off | Use errors from previous rounds as feedback |
| `--resume-from` | - | Resume from checkpoint directory |
| `--debug` | Off | Debug mode (8 operators only) |

## Parameter Details

### --op-name

Specify a single operator to test:

```bash
# ATen operator
--op-name aten::add

# vLLM operator
--op-name vllm13::rms_norm

# cuBLAS operator
--op-name cublas::cublasSgemm_v2
```

### --dataset

| Dataset | Operators | Platforms |
|---------|-----------|-----------|
| `KernelGenBench` | 210 | NVIDIA only |
| `KernelGenBench-aten` | 110 | All platforms |
| `KernelGenBench-vllm` | 50 | NVIDIA only |
| `KernelGenBench-cublas` | 50 | NVIDIA only |

### --temperature

| Value | Usage |
|-------|-------|
| 0 | Pass@1 evaluation (deterministic) |
| 0.8 | Pass@5 evaluation (diverse sampling) |

### --max-rounds

Number of independent kernel samples to generate:
- Higher values → better Pass@K coverage
- Higher cost → more API calls

### --base-url

Specify a custom OpenAI-compatible or Anthropic-compatible endpoint:

```bash
--api-format <openai|anthropic> --model-name <model> --base-url <endpoint>
```

### --api-key

Override the default API key from environment variables:

```bash
--api-key <your-key>
```

If not set, the selected protocol reads `OPENAI_API_KEY` or
`ANTHROPIC_API_KEY`. The corresponding base URL can be supplied through
`OPENAI_BASE_URL` or `ANTHROPIC_BASE_URL`. `--server-type` remains
available as a backward-compatible alias for `--api-format`.

## Output

Results saved to `output/pass_at_k/<timestamp>/`:

| File | Description |
|------|-------------|
| `pass_at_k_results.json` | Complete results |
| `kernels/` | Generated kernel files |
| `checkpoints/` | Resume checkpoints |
