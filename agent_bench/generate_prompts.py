#!/usr/bin/env python3
"""Generate prompt files for each operator in a dataset."""

import argparse
import inspect
import sys
from pathlib import Path

# Add project root to path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from flagbench.dataset import (
    V2_OPERATORS,
    V2_1_OPERATORS,
    CUPY_OPERATORS,
    get_kernelgenbench_operators,
)
from flagbench.dataset.kernel_list import (
    flatten_operator_dict, DynamicImplInfo,
    get_aten_operators, get_vllm_operators, get_cublas_operators,
    get_kernelgenbench_nocublas_operators,
)
from flagbench.dataset.dataloader import TorchOpsLoader
from runtime import get_visible_devices_env, get_device_constraints

# Import device detection for template selection
sys.path.insert(0, str(SCRIPT_DIR))
from device_manager import detect_device_type

# Device type to template subdirectory mapping
_DEVICE_TEMPLATE_DIR = {
    "npu": "ascend",
    "musa": "musa",
}


# Dataset name to operators mapping
DATASET_OPERATORS = {
    "v2": V2_OPERATORS,
    "v2_1": V2_1_OPERATORS,
    "cupy": CUPY_OPERATORS,
    "KernelGenBench": None,  # Special handling: 210 ops = 110 aten + 50 vllm13 + 50 cublas
    "KernelGenBench-aten": None,  # 110 aten ops
    "KernelGenBench-vllm": None,  # 50 vllm13 ops
    "KernelGenBench-cublas": None,  # 50 cublas ops
    "KernelGenBench-nocublas": None,  # 160 ops = 110 aten + 50 vllm13 (for non-NVIDIA chips)
}


def _get_kernelgenbench_flat_ops(dataset: str = "KernelGenBench") -> dict:
    """Get KernelGenBench operators as flat dict {full_name: func}."""
    if dataset == "KernelGenBench-aten":
        return get_aten_operators()
    elif dataset == "KernelGenBench-vllm":
        return get_vllm_operators()
    elif dataset == "KernelGenBench-cublas":
        return get_cublas_operators()
    elif dataset == "KernelGenBench-nocublas":
        return get_kernelgenbench_nocublas_operators()
    else:
        return get_kernelgenbench_operators()


def get_namespace_from_op(full_name: str) -> str:
    """Extract namespace from full operator name like 'aten::add' -> 'aten'."""
    if "::" in full_name:
        return full_name.split("::")[0]
    return "aten"


def get_baseline_operator_info(op_name: str, namespace: str, baseline_func=None) -> dict:
    """Get operator info from baseline function (for vllm13/cublas ops without torch.ops schema).

    Args:
        op_name: Operator name
        namespace: Namespace (vllm13, cublas)
        baseline_func: The baseline function object

    Returns:
        dict with keys: signatures, impl_info, input_args, baseline_code
    """
    info = {
        "signatures": "",
        "impl_info": f"- `{op_name}`",
        "input_args": "",
        "baseline_code": "",
    }

    if baseline_func is not None:
        try:
            sig = inspect.signature(baseline_func)
            info["signatures"] = f"- `{op_name}`: `{op_name}{sig}`"
            # Format input args
            params = []
            for pname, param in sig.parameters.items():
                if param.annotation != inspect.Parameter.empty:
                    params.append(f"  - `{pname}`: {param.annotation}")
                else:
                    params.append(f"  - `{pname}`")
            info["input_args"] = "\n".join(params) if params else f"(See {namespace} documentation)"

            # Get docstring if available
            doc = inspect.getdoc(baseline_func)
            if doc:
                info["input_args"] += f"\n\n**Description**: {doc}"

            # Get baseline source code
            try:
                info["baseline_code"] = inspect.getsource(baseline_func)
            except (OSError, TypeError):
                info["baseline_code"] = f"# Source code not available for {op_name}"
        except Exception:
            info["signatures"] = f"- `{op_name}`: (See {namespace} documentation)"
            info["input_args"] = f"(See {namespace} documentation for parameter details)"
    else:
        info["signatures"] = f"- `{op_name}`: (See {namespace} documentation)"
        info["input_args"] = f"(See {namespace} documentation for parameter details)"

    return info


# Template name mapping by namespace
TEMPLATE_BY_NAMESPACE = {
    "aten": "triton_kernel_aten.md",
    "vllm13": "triton_kernel_vllm.md",
    "cublas": "triton_kernel_cublas.md",
    "cupy": "triton_kernel_cublas.md",  # cupy uses same template as cublas
}


def load_template(template_path: Path) -> str:
    """Load prompt template from file."""
    with open(template_path, "r") as f:
        return f.read()


# Global loaders (reuse to avoid repeated initialization)
_torch_ops_loader = None
_impl_info_loader = None


def get_loaders():
    """Get or create global loaders."""
    global _torch_ops_loader, _impl_info_loader
    if _torch_ops_loader is None:
        _torch_ops_loader = TorchOpsLoader(to_str=True)
    if _impl_info_loader is None:
        _impl_info_loader = DynamicImplInfo()
    return _torch_ops_loader, _impl_info_loader


def get_operator_info(op_name: str, namespace: str = "aten") -> dict:
    """Get operator signatures and implementation info.

    Args:
        op_name: Operator name (e.g., "add", "softmax")
        namespace: Namespace (e.g., "aten", "cupy")

    Returns:
        dict with keys: signatures, impl_info, input_args
    """
    loader, impl_info_loader = get_loaders()

    info = {
        "signatures": "",
        "impl_info": "",
        "input_args": "",
    }

    try:
        # Get operator schemas
        api_info = loader.get_operator(namespace, op_name)

        # Format signatures
        sig_lines = []
        for overload_name, schema in api_info.schemas.items():
            full_name = f"{op_name}.{overload_name}" if overload_name else op_name
            sig_lines.append(f"- `{full_name}`: {schema}")
        info["signatures"] = "\n".join(sig_lines)

        # Get impl info (which overloads to implement)
        impl = impl_info_loader.get(op_name, namespace=namespace)
        if impl:
            impl_lines = [f"- `{name}` (autograd: {auto.name})" for name, auto in impl]
            info["impl_info"] = "\n".join(impl_lines)
        else:
            # If not in static IMPL_INFO, use all schemas
            impl_lines = [f"- `{name}`" for name in api_info.schemas.keys()]
            info["impl_info"] = "\n".join(impl_lines) if impl_lines else "- `default`"

        # Extract input args from first schema
        if api_info.schemas:
            first_schema = next(iter(api_info.schemas.values()))
            info["input_args"] = f"```\n{first_schema}\n```"

    except Exception as e:
        print(f"Warning: Could not get info for {namespace}::{op_name}: {e}")
        info["signatures"] = f"(Could not load schema for {op_name})"
        info["impl_info"] = f"- `{op_name}`"
        info["input_args"] = "(See PyTorch documentation)"

    return info


def render_prompt(template: str, operator: str, full_name: str, op_info: dict,
                  gpu_id: int = 0, python_path: str = "python") -> str:
    """Render prompt template with operator info."""
    prompt = template.replace("{{OPERATOR}}", operator)
    prompt = prompt.replace("{{FULL_NAME}}", full_name)
    prompt = prompt.replace("{{GPU_ID}}", str(gpu_id))
    prompt = prompt.replace("{{PYTHON_PATH}}", python_path)
    prompt = prompt.replace("{{OP_SIGNATURES}}", op_info["signatures"])
    prompt = prompt.replace("{{IMPL_INFO}}", op_info.get("impl_info", ""))
    prompt = prompt.replace("{{INPUT_ARGS}}", op_info["input_args"])
    prompt = prompt.replace("{{BASELINE_CODE}}", op_info.get("baseline_code", ""))
    prompt = prompt.replace("{{REFERENCE_CODE}}", "")  # Optional, can be added later
    # Device-agnostic: inject device env var name
    prompt = prompt.replace("{{DEVICE_ENV}}", get_visible_devices_env())
    return prompt


def generate_prompts_for_dataset(
    dataset: str,
    output_dir: Path,
    template_dir: Path,
    operators: list[str] | None = None,
    force: bool = False,
    python_path: str = "python",
) -> None:
    """Generate prompt files for all operators in a dataset.

    Args:
        dataset: Dataset name (v2, v2_1, cupy, KernelGenBench)
        output_dir: Output directory for prompt files
        template_dir: Directory containing prompt templates
        operators: Optional list of specific operators to generate
        force: Overwrite existing files
        python_path: Python interpreter path for prompts
    """
    if dataset not in DATASET_OPERATORS:
        raise ValueError(f"Unknown dataset: {dataset}. Available: {list(DATASET_OPERATORS.keys())}")

    # Pre-load all templates (keyed by namespace)
    # Device-specific templates take priority over generic ones
    device_type = detect_device_type()
    device_subdir = _DEVICE_TEMPLATE_DIR.get(device_type)
    device_template_dir = template_dir / device_subdir if device_subdir else None

    templates = {}
    for ns, tmpl_name in TEMPLATE_BY_NAMESPACE.items():
        # Try device-specific template first
        if device_template_dir:
            device_tmpl_path = device_template_dir / tmpl_name
            if device_tmpl_path.exists():
                templates[ns] = load_template(device_tmpl_path)
                continue
        # Fallback to generic template
        tmpl_path = template_dir / tmpl_name
        if tmpl_path.exists():
            templates[ns] = load_template(tmpl_path)
    # Fallback: legacy single template
    legacy_path = template_dir / "triton_kernel.md"
    if legacy_path.exists():
        legacy_template = load_template(legacy_path)
    else:
        legacy_template = None

    # Get operators dict and flatten
    is_kgb = dataset.startswith("KernelGenBench")
    if is_kgb:
        flat_ops = _get_kernelgenbench_flat_ops(dataset)
    elif dataset == "cupy":
        flat_ops = DATASET_OPERATORS[dataset]
    else:
        flat_ops = flatten_operator_dict(DATASET_OPERATORS[dataset], "aten")

    # Sub-datasets share KernelGenBench prompts directory
    prompts_dataset = "KernelGenBench" if is_kgb else dataset

    # Create output directory
    dataset_output_dir = output_dir / prompts_dataset
    dataset_output_dir.mkdir(parents=True, exist_ok=True)

    # Filter operators if specified (exact match on operator name)
    if operators:
        flat_ops = {k: v for k, v in flat_ops.items()
                   if k.split("::")[-1] in operators}

    # Generate ops_list.txt
    ops_list_path = dataset_output_dir / "ops_list.txt"
    with open(ops_list_path, "w") as f:
        for full_name in sorted(flat_ops.keys()):
            f.write(f"{full_name}\n")
    print(f"Generated {ops_list_path} ({len(flat_ops)} operators)")

    # Generate prompt for each operator
    generated = 0
    skipped = 0

    for full_name in sorted(flat_ops.keys()):
        op_name = full_name.split("::")[-1]
        namespace = get_namespace_from_op(full_name)

        # Output file path: use full_name with :: replaced to avoid collisions
        # e.g. "aten::add" -> "aten__add.md", "vllm13::rms_norm" -> "vllm13__rms_norm.md"
        if is_kgb:
            safe_name = full_name.replace("::", "__")
            prompt_path = dataset_output_dir / f"{safe_name}.md"
        else:
            prompt_path = dataset_output_dir / f"{op_name}.md"

        # Skip if exists and not force
        if prompt_path.exists() and not force:
            skipped += 1
            continue

        # Select template by namespace
        template = templates.get(namespace) or legacy_template
        if template is None:
            print(f"Warning: No template found for namespace '{namespace}', skipping {full_name}")
            continue

        # Get operator info based on namespace
        if namespace in ("vllm13", "cublas", "cupy"):
            baseline_func = flat_ops.get(full_name)
            op_info = get_baseline_operator_info(op_name, namespace, baseline_func)
        else:
            op_info = get_operator_info(op_name, namespace)

        # Render prompt
        prompt = render_prompt(
            template=template,
            operator=op_name,
            full_name=full_name,
            op_info=op_info,
            python_path=python_path,
        )

        # Write prompt file
        with open(prompt_path, "w") as f:
            f.write(prompt)

        generated += 1

    print(f"Generated {generated} prompts, skipped {skipped} existing")


def main():
    parser = argparse.ArgumentParser(description="Generate prompt files for operator datasets")
    parser.add_argument(
        "--dataset", "-d",
        type=str,
        choices=list(DATASET_OPERATORS.keys()) + ["all"],
        default="v2_1",
        help="Dataset to generate prompts for (default: v2_1)"
    )
    parser.add_argument(
        "--op", "-o",
        type=str,
        default=None,
        help="Specific operator(s) to generate, comma-separated (e.g., add,softmax)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=SCRIPT_DIR / "prompts",
        help="Output directory for prompts (default: prompts/)"
    )
    parser.add_argument(
        "--template-dir",
        type=Path,
        default=SCRIPT_DIR / "templates",
        help="Directory containing prompt templates (default: templates/)"
    )
    parser.add_argument(
        "--force", "-f",
        action="store_true",
        help="Overwrite existing prompt files"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Generate prompts for all datasets"
    )
    parser.add_argument(
        "--python-path",
        type=str,
        default="python",
        help="Python interpreter path for prompts (default: python)"
    )

    args = parser.parse_args()

    # Parse operators
    operators = args.op.split(",") if args.op else None

    # Determine datasets to process
    if args.all or args.dataset == "all":
        datasets = list(DATASET_OPERATORS.keys())
    else:
        datasets = [args.dataset]

    # Generate prompts
    for dataset in datasets:
        print(f"\n=== Generating prompts for {dataset} ===")
        try:
            generate_prompts_for_dataset(
                dataset=dataset,
                output_dir=args.output_dir,
                template_dir=args.template_dir,
                operators=operators,
                force=args.force,
                python_path=args.python_path,
            )
        except Exception as e:
            print(f"Error generating prompts for {dataset}: {e}")
            if len(datasets) == 1:
                raise


if __name__ == "__main__":
    main()
