# Frequently Asked Questions

This section answers common questions about KernelGenBench.

## Installation

### Q: Which Python version do I need?

**A**: Python 3.8 or higher is required. Python 3.10+ is recommended for best compatibility.

### Q: Can I use KernelGenBench on CPU-only machines?

**A**: No, KernelGenBench requires GPU hardware for kernel verification. The generated Triton kernels must be executed on actual GPU hardware.

### Q: Why does vLLM installation fail on non-NVIDIA platforms?

**A**: vLLM is designed for NVIDIA GPUs. On non-NVIDIA platforms, torch and triton are pre-installed in vendor container images. Do NOT install vLLM on these platforms — the ATen dataset is automatically used instead.

### Q: How do I install Claude Code CLI for the Agent Track?

**A**: Run `npm install -g @anthropic-ai/claude-code`. You'll also need an Anthropic API key set via `export ANTHROPIC_API_KEY=your_key`.

## Usage

### Q: What's the difference between LLM Track and Agent Track?

**A**:
- **LLM Track**: Tests direct kernel generation without execution feedback. Lower cost, suitable for comparing base model capabilities.
- **Agent Track**: Tests iterative generation with execution feedback. Higher cost but better results, suitable for production-ready kernel generation.

### Q: How do I test a single operator?

**A**: Use the `--op-name` parameter:

```bash
# LLM Track
python scripts/generate_kernel_and_verify.py --op-name aten::add --single-test

# Agent Track
cd agent_bench && bash test_ops.sh add --device-count 1
```

### Q: Which dataset should I use?

**A**:
- **NVIDIA GPUs**: Use `KernelGenBench` (210 operators) for full evaluation
- **Non-NVIDIA platforms**: Use `KernelGenBench-aten` (110 operators), which is auto-selected
- **Specific focus**: Use `KernelGenBench-vllm` for inference kernels or `KernelGenBench-cublas` for linear algebra

### Q: How long does a full benchmark take?

**A**:
- LLM Track (Pass@5, 210 operators): ~6-12 hours depending on model and hardware
- Agent Track (Claude Code, 210 operators): ~24-48 hours depending on operator complexity

### Q: How can I reduce evaluation time?

**A**:
1. Use `--debug` mode (only 8 operators) for testing
2. Increase `--device-count` for parallel verification
3. Use smaller datasets (`KernelGenBench-aten` instead of full)
4. Use LLM Track instead of Agent Track

## Results

### Q: What does accuracy mean?

**A**: Accuracy is the percentage of operators where at least one generated kernel passes all test cases and anti-hack checks.

### Q: What does speedup mean?

**A**: Speedup is the geometric mean of (generated kernel time / baseline time). A speedup > 1.0× means the generated kernel is faster than the baseline.

### Q: Why is my speedup less than 1.0×?

**A**: Generated kernels may not always outperform optimized baselines. This is expected, especially for:
- cuBLAS operators (highly optimized over decades)
- Complex vLLM operators
- Operators on immature non-NVIDIA platforms

### Q: Where are my results saved?

**A**:
- LLM Track: `output/pass_at_k/<timestamp>/`
- Agent Track: `agent_bench/runs/<timestamp>/`

## Errors

### Q: Why do I get "CUDA out of memory" errors?

**A**: Reduce `--device-count` or use smaller batch sizes. Some operators require significant GPU memory.

### Q: Why do generated kernels fail verification?

**A**: Common reasons:
1. Numerical precision mismatch (tolerance too strict)
2. Edge cases not handled in kernel logic
3. Memory access violations
4. Shape/dtype mismatches

### Q: Why does the anti-hack check fail?

**A**: The generated kernel may be calling blacklisted APIs instead of implementing the actual computation. Check the kernel code for:
- Direct calls to `torch.ops.aten.*`
- Imports of `vllm` or `ctypes`
- Any bypass of Triton computation

## Platform-Specific

### Q: How do I run on Ascend NPU?

**A**: Install Ascend dependencies and run in the vendor container image:

```bash
pip install -r requirements/requirements_ascend.txt
pip install -e .
# Framework will auto-detect Ascend hardware
```

### Q: Why is accuracy lower on non-NVIDIA platforms?

**A**: Non-NVIDIA platforms have:
- Less mature Triton compilers
- Incomplete backend support
- Different memory models
- Different performance characteristics

This is expected and demonstrates the cross-platform portability challenge.

### Q: Can I use custom hardware?

**A**: Yes, you can extend KernelGenBench for new platforms by:
1. Adding device detection in `src/runtime/`
2. Creating platform-specific templates in `agent_bench/templates/`
3. Adding platform-specific tolerances

## Cost

### Q: How much does evaluation cost?

**A**: Costs depend on:
- Method (Pass@1 < Pass@5 < Claude Code < AKO4ALL)
- Number of operators
- Model choice

For reference, the full KernelGenBench evaluation consumed 15+ billion tokens.

### Q: How can I estimate costs before running?

**A**: Start with `--debug` mode (8 operators) to measure token consumption per operator, then extrapolate.

## Contributing

### Q: How do I add a new operator?

**A**: See [CONTRIBUTING.md](https://github.com/flagos-ai/KernelGenBench/blob/main/CONTRIBUTING.md) for detailed instructions on adding test cases.

### Q: How do I add a new agent method?

**A**: Create a new directory in `agent_bench/methods/` following the existing structure.
