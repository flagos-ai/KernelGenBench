# Cost Analysis

Understanding token costs for agent-based kernel generation.

## Token Consumption

Agent methods consume more tokens than direct LLM sampling.

| Method | Tokens per Success |
|--------|-------------------|
| Pass@5 | ~50K |
| Claude Code (normal) | ~500K |
| AKO4ALL | ~5.19M |

## Cost Factors

### Iterative Debugging

Agents may perform multiple iterations:
- Each iteration generates new code
- Execution feedback increases context
- Error messages increase prompt size

### Model Selection

| Model | Relative Cost |
|-------|---------------|
| GPT-4o | Medium |
| Opus-4.6 | High |
| Qwen3.5 | Low |
| GLM-5.0 | Medium |

### Operator Complexity

| Operator Type | Average Iterations |
|---------------|-------------------|
| ATen (Simple) | 2-5 |
| ATen (Complex) | 5-10 |
| vLLM | 10-20 |
| cuBLAS | 10-30 |

## Cost Estimation

### Quick Estimation

```bash
# First run in debug mode (8 operators)
bash test_ops.sh --debug --device-count 1

# Check token usage
cat agent_bench/runs/<run_name>/results.json | grep tokens
```

### Extrapolation

```
Full run cost ≈ (debug tokens / 8) × 210
```

## Cost Optimization

### Reduce Operators

```bash
# Test only specific operators
bash test_ops.sh add,softmax,mul --device-count 1
```

### Use Cheaper Methods

```bash
# naive_cc uses fewer tokens than normal_cc
bash test_ops.sh add -m naive_cc --device-count 1
```

### Set Timeout

```bash
# Limit time per operator
bash test_ops.sh add --timeout 300 --device-count 1
```

## Budget Planning

Based on KernelGenBench experiments:

| Scale | Estimated Tokens | Estimated Cost (Opus) |
|-------|------------------|----------------------|
| Debug (8 operators) | ~5M | ~$50 |
| ATen (110 operators) | ~500M | ~$5,000 |
| Full (210 operators) | ~1B | ~$10,000 |
| Full AKO4ALL | ~5B | ~$50,000 |

```{warning}
Large-scale agent evaluation can consume billions of tokens. Be sure to test with debug mode first and plan your budget accordingly.
```