<!--
 Copyright 2026 FlagOS Contributors

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 -->

# Anti-Hack Architecture

{term}`KernelGenBench` employs a three-tier anti-hack mechanism to prevent benchmark evasion and ensure generated kernels actually perform computation.

## Overview

The anti-hack architecture guards against "cheating" behaviors where generated code might:
- Call pre-existing APIs instead of implementing computation
- Bypass {term}`Triton` compilation
- Use hidden caching mechanisms

## L1: AST Static Scan

### Purpose

Enforce a whitelist-based approach: most `torch.*` API calls are forbidden.
Only tensor creation, dtype helpers, and constants are allowed.

### Method

Parse the generated abstract syntax tree (AST) to detect and block:

**Whitelist (allowed torch APIs):**
`torch.empty`, `torch.zeros`, `torch.randn`, `torch.range`, `torch.float16`, etc.

**Detected patterns (blocked):**

| Blocked Pattern | Reason |
|-----------------|--------|
| `torch.*()` not in whitelist | Prevents using torch.sum/mean/mm/reductions |
| `print()` | Prevents input sniffing from test harness |
| `.data_ptr()` / `.storage()` | Prevents raw memory access |
| Module-level `_cache = {}` | Prevents inter-iteration result caching |
| `import vllm` | Using pre-existing implementations |
| `exec()` / `eval()` | Dynamic code execution |
| Import alias / `getattr()` bypass | Catches obfuscation attempts |

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
