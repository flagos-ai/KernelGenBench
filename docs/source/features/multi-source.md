# Multi-Source Operators

{term}`KernelGenBench` evaluates kernel generation for 210 operators from three different sources, each representing different complexity levels and real-world application scenarios.

## Overview

| Source | Operator Count | Description |
|--------|----------------|-------------|
| {term}`ATen` | 110 | PyTorch native operators |
| {term}`vLLM` | 50 | LLM inference kernels |
| {term}`cuBLAS` | 50 | Closed-source library reimplementation |

## ATen Operators

PyTorch {term}`ATen` operators are core computational building blocks used in deep learning frameworks.

### Selection Criteria

- Selected the top 50 most frequently used operators from 2,907 open-source model training traces
- Evenly sampled 60 long-tail operators
- Total: 110 operators selected from 900+ ATen APIs

### Examples

`softmax`, `matmul`, `embedding`, `cumsum`, `add.Tensor`

### Prompt Construction

- Dynamic extraction of FunctionSchema
- Official PyTorch docstrings
- All overload variants as independent problems

### Baseline

PyTorch native C++ implementation

## vLLM Operators

Production-grade {term}`LLM` inference kernels from {term}`vLLM` (v0.13.0).

### Coverage

- Attention mechanism ({term}`PagedAttention` v1)
- KV cache management
- Mixed precision quantization (FP8/AWQ)

### Challenges

Complex memory layout management and algorithmic logic make functional correctness highly challenging.

### Goal

Verify the ability to generate practical inference acceleration kernels.

## cuBLAS Operators

Closed-source library reimplementation targeting {term}`cuBLAS` (v12.4).

### Selection Strategy

- Selected top 10 most frequently called routines via profiling traces
- Extended across different precisions (S/D/C/Z/H) and batching modes
- Strategic sampling of diverse BLAS routines

### API Variants

A single {term}`GEMM` family yields 14 independent problems:

| Precision | Standard | StridedBatched | Batched | 64-bit Index |
|-----------|----------|----------------|---------|--------------|
| Float32 | cublasSgemm | ✓ | ✓ | ✓ |
| Float64 | — | ✓ | ✓ | ✓ |
| Complex64 | ✓ | ✓ | — | ✓ |
| Complex128 | — | ✓ | ✓ | — |
| Float16 | — | ✓ | ✓ | — |

### Challenges

Matching decades of expert hand-tuned performance is extremely difficult.

### Baseline

Direct loading of `libcublas.so` via `ctypes.cdll`, bypassing high-level wrappers.