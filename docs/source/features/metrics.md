# Evaluation Metrics

{term}`KernelGenBench` measures kernel generation capabilities from multiple dimensions: correctness, performance, and cost efficiency.

## Accuracy

### Definition

The percentage of operators where at least one generated kernel passes the following checks:
- All test cases (combined shapes, data types, layouts)
- Three-layer anti-cheating checks

### Clean Pass Rate

A kernel is counted as successful only when:
1. All `ki` test cases pass numerical verification
2. All anti-cheating layers pass

### Test Case Generation

Combinatorial test suites are created through the Cartesian product of:
- Core semantic parameters (dimensions, transpose)
- Shape variations
- Data types
- Memory layouts

## Speedup

### Calculation

Two-level geometric mean:

1. First level: Geometric mean across `ki` test cases → Operator-level {term}`Speedup` `Si`
2. Second level: Geometric mean across all operators → Global speedup

### Formula

```
Si = (∏ speedup_test_j)^(1/ki)
Global = (∏ Si)^(1/n)
```

### Interpretation

| Speedup | Meaning |
|---------|---------|
| > 1.0× | Generated kernel is faster than baseline |
| = 1.0× | Comparable performance |
| < 1.0× | Generated kernel is slower than baseline |

### Baseline

| Source | Baseline |
|--------|----------|
| {term}`ATen` | PyTorch native C++ |
| {term}`vLLM` | {term}`vLLM` {term}`CUDA` kernels |
| {term}`cuBLAS` | Native {term}`cuBLAS` library |

## Token Cost

### Metrics

| Metric | Description |
|--------|-------------|
| Total Tokens | Sum of all tokens consumed |
| Tokens per Success | Total tokens ÷ number of passed operators |

### Importance

Token consumption directly affects:
- API costs
- Evaluation time
- Feasibility of large-scale runs

## Runtime

### Definition

Cumulative solving time per operator, independent of parallelism.

### Usage

Compare efficiency of different generation methods, independent of parallel execution.

## Summary Table

| Metric | Type | Description |
|--------|------|-------------|
| Accuracy | Correctness | Percentage of operators passing all tests |
| {term}`Speedup` | Performance | Geometric mean relative to baseline |
| Token Cost | Efficiency | Number of tokens consumed |
| Runtime | Efficiency | Time consumed |
