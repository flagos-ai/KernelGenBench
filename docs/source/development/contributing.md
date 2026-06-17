# Contributing

How to contribute to KernelGenBench.

## Ways to Contribute

| Area | Description |
|------|-------------|
| New operators | Expand the benchmark with new test cases |
| New chip backends | Extend support to additional hardware |
| New agents | Integrate coding tools like Codex, Trae, Cursor |
| New methods | Contribute specialized optimization pipelines |

## Getting Started

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## Development Setup

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/KernelGenBench.git
cd KernelGenBench

# Install in development mode
pip install -r requirements/requirements_nvidia.txt
pip install -e .

# Run tests
python -m pytest tests/
```

## Code Style

- Follow PEP 8 for Python code
- Use type hints where appropriate
- Add docstrings to public functions

## Pull Request Guidelines

1. **One feature per PR** - Keep changes focused
2. **Add tests** - Ensure new code is tested
3. **Update docs** - Keep documentation current
4. **Link issues** - Reference any related issues

## Directory Structure

```
src/kernelgenbench/    # Core package
├── accuracy/          # Accuracy tests
├── dataset/           # Dataset definitions
└── ...

agent_bench/           # Agent framework
├── methods/           # Agent methods
└── templates/         # Prompt templates
```

## Code Review

All submissions require review by maintainers.

## License

By contributing, you agree that your contributions will be licensed under the Apache 2.0 License.
