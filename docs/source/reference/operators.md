# Operators

Reference list of operator sources in {term}`KernelGenBench`.

## ATen Operators (110)

### Selection Criteria

- Top 50 most frequently used operators from 2,907 model training traces
- 60 long-tail operators uniformly sampled

### Examples

| Category | Operators |
|----------|-----------|
| Arithmetic | `add`, `div`, `floor_divide`, `cos`, `sin` |
| Reduction | `sum`, `mean`, `argmax`, `amin` |
| Linear | `matmul`, `linear`, `bmm` |
| Normalization | `_softmax`, `softmax` |
| Activation | `hardsigmoid`, `prelu`, `heaviside` |
| Shape Operations | `expand`, `repeat`, `view` |

## vLLM Operators (50)

### Categories

| Category | Description |
|----------|-------------|
| Attention | {term}`PagedAttention` variants |
| KV Cache | Cache management kernels |
| Quantization | FP8, AWQ kernels |
| Normalization | RMS normalization, fused normalization |

### Examples

| Operator | Description |
|----------|-------------|
| `paged_attention_v1` | Memory-efficient attention |
| `fused_add_rms_norm` | Fused normalization |
| `rotary_embedding` | Positional encoding |

## cuBLAS Operators (50)

### GEMM Family (14 Variants)

| Precision | Standard | StridedBatched | Batched |
|-----------|----------|----------------|---------|
| Float32 | `cublasSgemm_v2` | `cublasSgemmStridedBatched` | `cublasSgemmBatched_64` |
| Float64 | — | `cublasDgemmStridedBatched` | `cublasDgemmBatched` |
| Complex64 | `cublasCgemm_v2` | `cublasCgemmStridedBatched` | — |
| Complex128 | — | `cublasZgemmStridedBatched` | `cublasZgemmBatched` |
| Float16 | — | `cublasHgemmStridedBatched` | `cublasHgemmBatched` |

### Other BLAS Routines

- GEMV (matrix-vector multiplication)
- SYRK (symmetric rank-k update)
- TRSM (triangular solve)
- And other routines across multiple precisions

## Operator Naming Convention

| Source | Prefix | Example |
|--------|--------|---------|
| {term}`ATen` | `aten::` | `aten::add.Tensor` |
| {term}`vLLM` | `vllm13::` | `vllm13::rms_norm` |
| {term}`cuBLAS` | `cublas::` | `cublas::cublasSgemm_v2` |
