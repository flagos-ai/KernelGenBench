# Progress Report: Kaldi K1 CUDA Kernels Python Interface

## Date: 2026-01-16

## Summary

Successfully created a working Python interface for **3 Kaldi K1 CUDA kernels**:
- `copy_low_upp` - Copy lower triangle to upper triangle
- `copy_upp_low` - Copy upper triangle to lower triangle
- `add_mat` - Matrix addition: dst = alpha * src + dst

### Test Results
```
Test 1: copy_low_upp  ✓ PASSED
Test 2: copy_upp_low  ✓ PASSED  
Test 3: add_mat       ✓ PASSED
```

## Approach

We use PyTorch's JIT compilation (`torch.utils.cpp_extension.load_inline`) to:
1. Extract CUDA kernel source code from Kaldi repository
2. Generate C++ adapters with pybind11 bindings
3. Compile to `.so` module
4. Provide clean Python interface

## Tools Created

| Tool | Purpose | Location |
|------|---------|----------|
| `extract_3kernels_cuda.py` | Extract CUDA source | cache/extracted_cuda/unified_3kernels.cu |
| `generate_unified_wrapper.py` | Generate C++ adapters | cache/generated_adapters/unified_kaldi_adapter.cpp |
| `compile_and_test_unified.py` | Compile and test | cache/cuda_jit_unified/kaldi_k1_unified.so |
| `classify_kernels.py` | Classify 169 kernels | cache/kernel_classification.json |

## Kernel Classification

Analyzed all 169 kernels from `kernel_list_k1.py`:

| Phase | Category | Count | Description |
|-------|----------|-------|-------------|
| 1 | standard_dim3_matrixdim | 90 | Standard 2D grid pattern |
| 2 | vector_int_grid | 26 | 1D vector operations |
| 3 | sizet_grid | 4 | size_t grid pattern |
| 4 | standard_dim3_other | 7 | Other dim3 patterns |
| 5 | cudadecoder | 24 | Decoder-specific |
| 6 | cudafeat | 7 | Feature extraction |
| 7 | no_grid_special | 6 | Special cases |
| 8 | cublas | 2 | BLAS wrappers |

**Total Phase 1 (Standard): 127 kernels**

## Usage Example

```python
from script.cuda_baseline.compile_and_test_unified import compile_unified_kernels
import torch

# Load compiled module
kaldi = compile_unified_kernels()

# Use kernels
A = torch.randn(64, 64, device='cuda', dtype=torch.float32)
kaldi.copy_low_upp(A)  # Copy lower to upper triangle

dst = torch.randn(64, 64, device='cuda', dtype=torch.float32)
src = torch.randn(64, 64, device='cuda', dtype=torch.float32)
kaldi.add_mat(dst, src, 2.0)  # dst = 2.0 * src + dst
```

## Challenges Encountered

1. **Complex header dependencies**: Kaldi's `cu-kernels.cu` depends on:
   - `cu-common.h`
   - `kaldi-error.h`
   - `kaldi-types.h`
   - OpenFst headers

2. **Template-based implementation**: Kaldi uses C++ templates (`Real`, `OtherReal`)
   requiring type dispatch in Python adapters

3. **Grid/block calculation**: Kaldi calculates grid dimensions in C++,
   but PyTorch tensors don't expose this information directly

## Solution for Expansion

To add more kernels, follow this pattern:

### 1. Extract kernel source from Kaldi (cu-kernels.cu)
```cuda
template<typename Real>
__global__
static void _kernel_name(Real* param1, MatrixDim d) {
  // Kernel implementation
}

extern "C" {
void cudaF_kernel_name(dim3 Gr, dim3 Bl, float* param1, MatrixDim d) {
  _kernel_name<<<Gr, Bl>>>(param1, d);
}
void cudaD_kernel_name(dim3 Gr, dim3 Bl, double* param1, MatrixDim d) {
  _kernel_name<<<Gr, Bl>>>(param1, d);
}
}
```

### 2. Add adapter function in C++ adapter
```cpp
void kernel_name(torch::Tensor param1) {
  MatrixDim d;
  d.rows = param1.size(0);
  d.cols = param1.size(1);
  d.stride = param1.stride(0);
  
  dim3 Gr(n_blocks(d.cols, 16), n_blocks(d.rows, 16));
  dim3 Bl(16, 16);
  
  if (param1.scalar_type() == torch::kFloat32) {
    cudaF_kernel_name(Gr, Bl, param1.data_ptr<float>(), d);
  } else if (param1.scalar_type() == torch::kFloat64) {
    cudaD_kernel_name(Gr, Bl, param1.data_ptr<double>(), d);
  }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("kernel_name", &kernel_name, "Description");
}
```

## Next Steps

### Option 1: Manual Expansion (Recommended)
Manually add 10-20 most commonly used kernels:
1. scale
2. set_const
3. exp
4. log
5. sigmoid
6. tanh
7. copy_cols
8. copy_rows
9. add_cols
10. mul_cols_vec

### Option 2: Automated Extraction
Create regex-based extraction for each kernel template

### Option 3: Direct Compilation
Install OpenFst and compile entire cu-kernels.cu at once

## Recommendation

Start with **Option 1** for the next 10 kernels. This approach:
- Is most reliable
- Gives us control over each kernel's implementation
- Allows testing each kernel individually
- Avoids complex header dependency issues

## Status

✓ **Phase 1: Proof of Concept Complete**
- 3 kernels working and tested
- Toolchain established
- Classification complete
- Ready for Phase 2 expansion

→ **Phase 2: Expand to 10-20 kernels** (In Progress)
