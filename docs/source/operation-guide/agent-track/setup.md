# Setup

Configure your environment for the Agent Track evaluation.

## Option A: Single Environment (Recommended)

Install Claude Code CLI into the same environment with torch/vllm:

```bash
# In your KernelGenBench environment
npm install -g @anthropic-ai/claude-code
cp agent_bench/config.example.yaml agent_bench/config.yaml
```

Edit `config.yaml`:

```yaml
paths:
  python: /path/to/your/python  # Python with torch + vllm + kernelgenbench
```

## Option B: Separate Environments

If Claude Code is installed in a different environment:

```bash
cp agent_bench/config.example.yaml agent_bench/config.yaml
```

Edit `config.yaml`:

```yaml
paths:
  python: /path/to/envs/kernelgenbench/bin/python
```

When running, export PATH:

```bash
export PATH="/path/to/claude_tool/bin:$PATH"
cd agent_bench && bash test_ops.sh add --device-count 1
```

## Configuration Fields

| Field | Description |
|-------|-------------|
| `paths.python` | Python interpreter with torch + vllm + kernelgenbench |
| `agent.bin` | Path to agent CLI executable (default: `claude`) |

## API Credentials

Ensure your API keys are set:

```bash
# Anthropic Claude
export ANTHROPIC_API_KEY=your_key

# OpenAI / OpenAI-compatible
export OPENAI_API_KEY=your_key
```

## Verify Setup

```bash
cd agent_bench

# Quick test with single operator
bash test_ops.sh add --device-count 1
```
