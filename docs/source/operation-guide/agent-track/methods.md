# Methods

Available agent methods in KernelGenBench.

## Overview

| Method | Description | Model Support |
|--------|-------------|---------------|
| `naive_cc` | Single Claude Code call | Opus, GLM, Qwen, MiniMax |
| `normal_cc` | Claude Code + self-verification loop | Opus, GLM, Qwen, MiniMax |
| `naive_opencode` | Single OpenCode call | Opus, GLM, Qwen, MiniMax |
| `normal_opencode` | OpenCode + self-verification loop | Opus, GLM, Qwen, MiniMax |
| AutoKernel | Automated kernel optimization | GLM, Qwen |
| AKO4ALL | Kernel optimization for all operators | Opus |
| cuda-optimized-skill | CUDA optimization with strategy memory | Opus |

## Basic Agent Frameworks

### Claude Code

| Variant | Description |
|---------|-------------|
| `naive_cc` | Single generation process |
| `normal_cc` | Generation + self-verification + iterative repair |

**Best for**: High accuracy requirements

### OpenCode

| Variant | Description |
|---------|-------------|
| `naive_opencode` | Single generation process |
| `normal_opencode` | Generation + self-verification loop |

**Best for**: Alternative agent framework

## Kernel-Specific Agents

### AutoKernel

Automated kernel optimization pipeline:
- Focus on performance optimization
- Support for multiple models (GLM-5.0, Qwen3.5)
- Highest speedup results

### AKO4ALL

Kernel optimization for all operators:
- Uses Opus-4.6
- Comprehensive operator coverage
- Good balance of accuracy and speedup

### cuda-optimized-skill

CUDA optimization with strategy memory:
- Leverages historical optimization strategies
- Specialized for CUDA kernel optimization
- Strategy-based approach for improved performance

## Choosing a Method

| Goal | Recommended Method |
|------|-------------------|
| High accuracy | `normal_cc` (Claude Code) |
| High speedup | AutoKernel (Qwen) |
| Balanced performance | AKO4ALL |
| Cost-effective | `naive_cc` |