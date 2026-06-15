# Hardware Platforms

Details about supported hardware platforms.

## Supported Platforms

| Platform | Vendor | Type | Notes |
|----------|--------|------|-------|
| NVIDIA | NVIDIA | GPU | Primary baseline |
| Ascend NPU | Huawei | NPU | — |
| MUSA | Moore Threads | GPU | — |
| Hygon DCU | Hygon | DCU | — |
| Iluvatar | Iluvatar | AI Chip | — |
| MetaX | MUXI | GPU | — |

## Platform Detection

```bash
# Check detected device
python -c "from kernelgenbench.runtime import get_device_type; print(get_device_type())"
```

## Platform-Specific Features

### NVIDIA

| Feature | Support |
|---------|---------|
| All datasets | ✓ |
| L3 Anti-hack | ✓ |
| {term}`vLLM` operators | ✓ |
| {term}`cuBLAS` operators | ✓ |

### Non-NVIDIA

| Feature | Support |
|---------|---------|
| {term}`ATen` dataset only | ✓ |
| L1 + L2 Anti-hack | ✓ |
| L3 Anti-hack | ✗ |
| {term}`vLLM` operators | ✗ |
| {term}`cuBLAS` operators | ✗ |

## Installation Requirements

### NVIDIA

```bash
pip install -r requirements/requirements_nvidia.txt
pip install -e .
```

### Ascend NPU

```bash
pip install -r requirements/requirements_ascend.txt
pip install -e .
# Use vendor container image
```

### MUSA

```bash
pip install -r requirements/requirements_musa.txt
pip install -e .
# Use vendor container image
```

### Hygon DCU

```bash
pip install -r requirements/requirements_hygon.txt
pip install -e .
# Use vendor container image
```

### Iluvatar

```bash
pip install -r requirements/requirements_iluvatar.txt
pip install -e .
# Use vendor container image
```

### MetaX

```bash
pip install -r requirements/requirements_metax.txt
pip install -e .
# Use vendor container image
```

## Cross-Platform Considerations

### Compiler Maturity

Non-NVIDIA platforms typically have:
- Less mature {term}`Triton` compilers
- Incomplete backend support
- Different memory models

### Performance Impact

- 2× more tokens and time on average
- May require platform-specific optimizations
- Results vary by {term}`Operator` complexity

### Tolerance Settings

Numerical tolerances are automatically adjusted per platform.
