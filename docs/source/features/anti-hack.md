# Anti-Hack Architecture

{term}`KernelGenBench` employs a three-tier anti-hack mechanism to prevent benchmark evasion and ensure generated kernels actually perform computation.

## Overview

The anti-hack architecture guards against "cheating" behaviors where generated code might:
- Call pre-existing APIs instead of implementing computation
- Bypass {term}`Triton` compilation
- Use hidden caching mechanisms

## L1: AST Static Scan

### Purpose

Block blacklisted API calls before execution.

### Method

Parse the generated abstract syntax tree (AST) to detect and block:

| Blocked Pattern | Reason |
|-----------------|--------|
| `torch.ops.aten.*` | Direct PyTorch API calls |
| `import vllm` | Using {term}`vLLM` implementation |
| `ctypes` | Dynamic library loading |
| `__import__` | Dynamic imports |

### Implementation

```python
# Blocked calls are detected via AST parsing
# Any attempt to call blacklisted APIs results in immediate rejection
```

## L2: Ghost Replay

### Purpose

Verify that the {term}`Triton` kernel is actually executed, not bypassed.

### Method

1. Execute kernel normally, capture outputs
2. Replace `@triton.jit` decorated function with no-op in memory
3. Re-execute with same inputs
4. Compare outputs

### Logic

- If outputs are **identical**, the {term}`Triton` kernel was never invoked → **Cheating detected**
- If outputs **differ**, the kernel was actually executed → **Valid**

## L3: Hardware Profiling

### Purpose

Confirm {term}`Triton`-specific execution at the hardware level.

### Method

Use `torch.profiler` to verify {term}`Triton`-specific signatures exist in low-level trace logs.

### Availability

| Platform | L3 Support |
|----------|------------|
| NVIDIA | ✓ |
| Non-NVIDIA | ✗ |

Non-NVIDIA platforms rely on L1 and L2 due to absence of equivalent profiling tools.

## Validation Flow

```
Generated Kernel
      │
      ▼
┌─────────────┐
│ L1: AST Scan│─── Fail ──► Reject
└─────────────┘
      │ Pass
      ▼
┌─────────────┐
│ L2: Ghost   │─── Fail ──► Reject
│    Replay   │
└─────────────┘
      │ Pass
      ▼
┌─────────────┐
│ L3: Profile │─── Fail ──► Reject
│  (NVIDIA)   │
└─────────────┘
      │ Pass
      ▼
   Accept
```

## Why Anti-Hack Matters

Without anti-hack measures, models could:
- Achieve high "accuracy" without actual computation
- Mask poor kernel generation capability
- Invalidate benchmark results

{term}`KernelGenBench` ensures evaluations reflect true kernel generation ability.
