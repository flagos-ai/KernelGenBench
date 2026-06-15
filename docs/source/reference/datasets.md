# Datasets

{term}`KernelGenBench` provides four dataset variants for different evaluation scenarios.

## Dataset Overview

| Dataset | Operators | Sources | Platforms |
|---------|-----------|---------|-----------|
| `{term}`KernelGenBench`` | 210 | {term}`ATen` + {term}`vLLM` + {term}`cuBLAS` | NVIDIA only |
| `{term}`KernelGenBench-aten`` | 110 | {term}`ATen` only | All platforms |
| `{term}`KernelGenBench-vllm`` | 50 | {term}`vLLM` only | NVIDIA only |
| `{term}`KernelGenBench-cublas`` | 50 | {term}`cuBLAS` only | NVIDIA only |

## KernelGenBench (Full)

The complete benchmark for NVIDIA platforms.

### Composition

| Source | Count | Description |
|--------|-------|-------------|
| {term}`ATen` | 110 | PyTorch operators |
| {term}`vLLM` | 50 | Inference kernels |
| {term}`cuBLAS` | 50 | BLAS routines |

### Use Case

- Comprehensive evaluation
- NVIDIA hardware only
- Full capability assessment

## KernelGenBench-aten

Cross-platform operator set.

### Composition

- 110 PyTorch {term}`ATen` operators
- No external dependencies

### Use Case

- Multi-chip evaluation
- Cross-platform testing
- Portable benchmark

## KernelGenBench-vllm

{term}`LLM` inference kernel benchmark.

### Composition

- 50 {term}`vLLM` operators
- Attention mechanisms ({term}`PagedAttention`)
- KV cache management
- Quantization kernels

### Use Case

- Inference optimization
- {term}`vLLM` replacement testing
- Complex kernel evaluation

## KernelGenBench-cublas

Linear algebra benchmark.

### Composition

- 50 {term}`cuBLAS` operators
- {term}`GEMM` variants
- Multiple precisions (S/D/C/Z/H)
- Batched and strided versions

### Use Case

- Performance ceiling testing
- BLAS replacement
- High-precision kernels

## Dataset Selection

```bash
# Specify dataset
python scripts/generate_kernel_and_verify.py \
    --dataset {term}`KernelGenBench-aten` \
    --server-type openai
```

On non-NVIDIA platforms, `{term}`KernelGenBench-aten`` is automatically selected.
