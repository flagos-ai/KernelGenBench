# Parameters

LLM Track command-line parameters.

## Required Parameters

| Parameter | Description |
|-----------|-------------|
| `--server-type` | LLM provider: `openai` or `anthropic` |
| `--model-name` | Model identifier |

## Optional Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--op-name` | All | Test a single operator (e.g., `aten::add`) |
| `--single-test` | Off | Randomly select 1 operator for quick testing |
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

## Output

Results saved to `output/pass_at_k/<timestamp>/`:

| File | Description |
|------|-------------|
| `pass_at_k_results.json` | Complete results |
| `kernels/` | Generated kernel files |
| `checkpoints/` | Resume checkpoints |
