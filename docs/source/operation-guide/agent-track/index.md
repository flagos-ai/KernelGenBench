# Agent Track

Agent Track evaluates coding agents that iteratively generate, verify, and optimize kernels.

```{toctree}
:maxdepth: 1

setup
methods
commands
cost-analysis
```

## What It Tests

Autonomous debugging and optimization capability with execution feedback.

## When to Use

- Testing agent frameworks (Claude Code, OpenCode)
- Evaluating kernel-specialized agents (AutoKernel, AKO4ALL)
- Production-ready kernel generation

## Quick Start

```bash
cd agent_bench
bash test_ops.sh add --device-count 1
```
