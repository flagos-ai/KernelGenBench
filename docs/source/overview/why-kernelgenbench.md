# Why KernelGenBench?

Understanding the significance of KernelGenBench requires first understanding the challenges of GPU kernel development and the limitations of existing solutions.

## Challenges of Kernel Development

GPU kernel development is a highly specialized and labor-intensive task. Writing efficient kernels requires:

- Deep understanding of low-level programming
- Hardware architecture expertise
- Performance optimization skills
- Cross-platform compatibility knowledge

The rise of LLM and agent-based frameworks offers a promising path for automatic kernel generation, but evaluating their effectiveness requires rigorous benchmarks.

## Limitations of Existing Solutions

Existing benchmarks face the following limitations:

| Limitation | Description |
|------------|-------------|
| Single-source constraint | Only tests standardized PyTorch operators |
| Single-hardware lock-in | Limited to the NVIDIA ecosystem |
| Limited verification | Only focuses on functional correctness |
| No cost tracking | Ignores token consumption and time costs |

### Single-source Constraint

KernelBench and TritonBench pioneered execution-based evaluation, but their exclusive focus on standardized PyTorch operators allows kernel-specific agents to achieve nearly 100% accuracy, creating an illusion that the problem has been solved.

### Single-hardware Lock-in

The vast majority of kernel benchmarks are strictly limited to the NVIDIA ecosystem. No existing benchmark system has measured the performance portability gap across heterogeneous hardware platforms.

## How KernelGenBench Helps

KernelGenBench addresses these gaps through:

1. **Multi-source Evaluation** - Tests operators from ATen, vLLM, and cuBLAS
2. **Multi-chip Support** - Evaluates on 6 hardware platforms
3. **Rigorous Verification** - Three-layer anti-cheating mechanism
4. **Cost Tracking** - Measures token consumption and actual runtime

## Use Cases

### For Individual Developers

Use KernelGenBench as a "quality inspector" for kernel development. The three-layer anti-cheating mechanism ensures your generated kernels truly work in production environments.

### For Chip Vendors

KernelGenBench serves as a "detector" for heterogeneous adaptation. Identify performance bottlenecks and compiler compatibility issues across different chips.

### For Enterprise Teams

Use KernelGenBench as a "decision calculator" for automation costs. Based on large-scale experiments (over 15 billion tokens), understand the token and time costs of automatic kernel generation.