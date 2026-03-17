"""Agent methods for kernel generation.

Each method is a self-contained module in its own directory:

    methods/
    ├── __init__.py          # This file - registry and factory
    ├── base.py              # BaseMethod and MethodResult
    ├── naive_cc/            # Single-call CC method
    │   ├── __init__.py
    │   └── method.py
    ├── normal_cc/           # CC with self-verification loop
    │   ├── __init__.py
    │   └── method.py
    └── iterative_optimizer/ # Multi-round optimization method
        ├── __init__.py
        ├── method.py
        ├── worker.py
        ├── templates/
        └── tools/
"""

from .base import BaseMethod, MethodResult
from .naive_cc import NaiveCCMethod
from .normal_cc import NormalCCMethod
from .iterative_optimizer import IterativeOptimizerMethod

# Registry of available methods
_METHODS = {
    "naive_cc": NaiveCCMethod,
    "normal_cc": NormalCCMethod,
    "iterative_optimizer": IterativeOptimizerMethod,
}


def get_method(name: str) -> BaseMethod:
    """Get a method instance by name.

    Args:
        name: Method name (e.g., "naive_cc", "iterative_optimizer")

    Returns:
        Instance of the method

    Raises:
        ValueError: If method name is not found
    """
    if name not in _METHODS:
        available = ", ".join(_METHODS.keys())
        raise ValueError(f"Unknown method: {name}. Available: {available}")
    return _METHODS[name]()


def list_methods() -> list[str]:
    """List available method names."""
    return list(_METHODS.keys())


__all__ = [
    "BaseMethod",
    "MethodResult",
    "get_method",
    "list_methods",
    "NaiveCCMethod",
    "NormalCCMethod",
    "IterativeOptimizerMethod",
]
