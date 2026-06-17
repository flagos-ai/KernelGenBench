# Glossary

This section defines technical terminology used throughout the KernelGenBench documentation.

{.glossary}
Agent
: A coding agent that autonomously generates, executes, and iterates on code based on feedback. In KernelGenBench, agents like Claude Code and OpenCode can debug and optimize kernels through execution-driven reinforcement.

{.glossary}
ATen
: PyTorch's native tensor library, providing fundamental operations for deep learning. KernelGenBench includes 110 ATen operators derived from real model training traces.

{.glossary}
CUDA
: NVIDIA's proprietary parallel computing platform and programming model for GPU acceleration. CUDA is deeply tied to NVIDIA hardware architecture.

{.glossary}
cuBLAS
: NVIDIA's closed-source Basic Linear Algebra Subprograms library, highly optimized for NVIDIA GPUs. KernelGenBench includes 50 cuBLAS operators representing extreme performance challenges.

{.glossary}
GEMM
: General Matrix Multiplication, a fundamental linear algebra operation. cuBLAS includes numerous GEMM variants across different precisions and batching modes.

{.glossary}
Kernel
: A function that executes on a GPU, written in CUDA or Triton. Kernels directly determine computational performance and must be optimized for specific hardware.

{.glossary}
KernelGenBench
: A comprehensive benchmark framework for evaluating LLM and agent-based Triton kernel generation across multiple hardware platforms. Part of the FlagOS ecosystem.

{.glossary}
KernelGenBench-aten
: A dataset subset containing 110 PyTorch ATen operators, used for cross-platform evaluation on all supported hardware.

{.glossary}
KernelGenBench-cublas
: A dataset subset containing 50 cuBLAS operators, available only on NVIDIA platforms due to library dependencies.

{.glossary}
KernelGenBench-nocublas
: A dataset subset containing 160 operators (ATen + vLLM), used for NVIDIA evaluation without cuBLAS dependency.

{.glossary}
KernelGenBench-MS
: The Multi-Source sub-benchmark evaluating 210 operators from three sources (ATen, vLLM, cuBLAS) on NVIDIA hardware.

{.glossary}
KernelGenBench-MC
: The Multi-Chip sub-benchmark evaluating 110 ATen operators across six hardware platforms to measure performance portability.

{.glossary}
KernelGenBench-vllm
: A dataset subset containing 50 vLLM operators, available only on NVIDIA platforms.

{.glossary}
LLM
: Large Language Model, an AI model trained on vast amounts of text data. In KernelGenBench, LLMs are evaluated on their ability to generate GPU kernels.

{.glossary}
Operator
: A reusable computational unit in deep learning frameworks. Operators define "what" to compute (e.g., `torch.add`), while kernels define "how" to execute on hardware.

{.glossary}
Pass@K
: An evaluation metric measuring whether at least one correct solution exists among K generated samples. Pass@1 tests single-generation capability; Pass@5 allows multiple attempts.

{.glossary}
PagedAttention
: A memory-efficient attention mechanism used in vLLM for LLM inference. Part of the vLLM operator subset in KernelGenBench.

{.glossary}
Speedup
: Performance improvement ratio of generated kernel versus baseline implementation. Calculated as geometric mean across test cases and operators.

{.glossary}
Triton
: An open-source programming language for GPU kernels that abstracts low-level details while maintaining high performance. Triton code is portable across different GPU architectures.

{.glossary}
vLLM
: A high-throughput LLM inference engine with custom CUDA kernels. KernelGenBench includes 50 vLLM operators representing production inference workloads.

---

## Acronyms

| Acronym | Full Name |
|---------|-----------|
| AST | Abstract Syntax Tree |
| ATen | A Tensor Library |
| BLAS | Basic Linear Algebra Subprograms |
| CUDA | Compute Unified Device Architecture |
| DCU | Data Center Accelerator |
| GEMM | General Matrix Multiplication |
| GPU | Graphics Processing Unit |
| LLM | Large Language Model |
| MUSA | Moore Threads Unified System Architecture |
| NPU | Neural Processing Unit |

---

## Hardware Platforms

| Platform | Vendor | Description |
|----------|--------|-------------|
| NVIDIA | NVIDIA | A100 GPUs, primary evaluation baseline |
| Ascend | Huawei | Neural Processing Units |
| MUSA | Moore Threads | GPU architecture |
| Hygon | Hygon | Data Center Accelerators |
| Iluvatar | Iluvatar | AI accelerators |
| MetaX | MUXI | GPU accelerators |
