# Multi-Chip Support

{term}`KernelGenBench` supports six hardware platforms with automatic device detection and unified execution pipeline.

## Supported Platforms

| Platform | Description | Notes |
|----------|-------------|-------|
| NVIDIA | A100 GPUs | Primary baseline |
| Ascend NPU | Huawei AI accelerators | — |
| MUSA | Moore Threads GPUs | — |
| Hygon DCU | Hygon data center accelerators | — |
| Iluvatar | Iluvatar AI chips | — |
| MetaX | MUXI accelerators | — |

## Auto-Detection

Device type is automatically detected at runtime:

```bash
# Check detected device
python -c "from runtime import get_device_type; print(get_device_type())"
```

## Unified Commands

All platforms use the same commands — the framework handles device differences automatically:

```bash
# Same command works on all platforms
python scripts/generate_kernel_and_verify.py \
    --server-type openai \
    --model-name gpt-4o
```

## Platform-Specific Behavior

### Dataset Selection

| Platform | Default Dataset |
|----------|-----------------|
| NVIDIA | {term}`KernelGenBench` (210 operators) |
| Others | {term}`KernelGenBench-aten` (110 operators) |

On non-NVIDIA platforms, {term}`vLLM` and {term}`cuBLAS` operators are unavailable.

### Anti-Hack Layers

| Layer | NVIDIA | Non-NVIDIA |
|-------|--------|------------|
| L1: AST Static Scan | ✓ | ✓ |
| L2: Ghost Replay | ✓ | ✓ |
| L3: Hardware Profiling | ✓ | ✗ |

L3 profiling is NVIDIA-only due to tool availability.

### Tolerance Settings

Numerical tolerances are automatically adjusted per platform to account for different floating-point implementations.

## Cross-Platform Challenges

### Compiler Maturity

Non-NVIDIA platforms have:
- Less mature {term}`Triton` compilers
- Incomplete backend support
- Different memory models

### Performance Impact

- Non-NVIDIA platforms require ~2× more tokens and time
- Cross-platform degradation can be severe
- Platform-specific optimizations needed

## Hardware-Specific Templates

The framework injects platform-specific code templates:

- Import statements
- Runtime configurations
- Memory constraints
- Device-specific constants
