# Commands

CLI commands for the Agent Track evaluation.

## Basic Usage

```bash
cd agent_bench

# Single operator
bash test_ops.sh add --device-count 1

# Multiple operators
bash test_ops.sh add,softmax,mul --device-count 4

# All operators
bash test_ops.sh --device-count 8
```

## Method Selection

```bash
# Naive method (single call)
bash test_ops.sh add -m naive_cc --device-count 1

# Normal method (with verification loop)
bash test_ops.sh add -m normal_cc --device-count 1

# OpenCode methods
bash test_ops.sh add -m naive_opencode --device-count 1
bash test_ops.sh add -m normal_opencode --device-count 1
```

## Specialized Agents

```bash
# AutoKernel
bash test_autokernel.sh add --device-count 1

# AKO4ALL
bash test_ako4all.sh add --device-count 1

# CUDA Optimized Skill
bash test_cuda_optimized_skill.sh add --device-count 1
```

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `[operators]` | All | Comma-separated operator names |
| `-d, --dataset` | `KernelGenBench` | Dataset to use |
| `-m, --method` | `normal_cc` | Agent method |
| `--device-count` | 8 | Number of GPUs |
| `--timeout` | 600 | Timeout per operator (seconds) |
| `--skip-gen` | Off | Skip prompt generation |
| `--skip-verify` | Off | Skip verification |
| `-v, --verbose` | Off | Enable verbose output |

## Output

Results saved to `agent_bench/runs/<run_name>/`:

| File | Description |
|------|-------------|
| `progress.json` | Real-time progress tracking |
| `kernels/` | Generated kernel files |
| `results.json` | Verification results |
| `logs/` | Execution logs |

## Monitoring Progress

```bash
# Watch progress file
cat agent_bench/runs/<run_name>/progress.json
```

## Analyzing Results

```bash
python scripts/analyze/analyze.py agent_bench/runs/<run_dir>/
```
