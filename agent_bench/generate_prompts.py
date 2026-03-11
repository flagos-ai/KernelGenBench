#!/usr/bin/env python3
"""Generate prompt files for each operator in a dataset."""

import argparse
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
)
from flagbench.dataset.kernel_list import flatten_operator_dict, DynamicImplInfo
from flagbench.dataset.dataloader import TorchOpsLoader


# Dataset name to operators mapping
DATASET_OPERATORS = {
    "v2": V2_OPERATORS,
    "v2_1": V2_1_OPERATORS,
    "cupy": CUPY_OPERATORS,
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
    prompt = prompt.replace("{{IMPL_INFO}}", op_info["impl_info"])
    prompt = prompt.replace("{{INPUT_ARGS}}", op_info["input_args"])
    prompt = prompt.replace("{{REFERENCE_CODE}}", "")  # Optional, can be added later
    return prompt


def generate_prompts_for_dataset(
    dataset: str,
    output_dir: Path,
    template_path: Path,
    operators: list[str] | None = None,
    force: bool = False,
    python_path: str = "python",
) -> None:
    """Generate prompt files for all operators in a dataset.

    Args:
        dataset: Dataset name (v2, v2_1, cupy)
        output_dir: Output directory for prompt files
        template_path: Path to prompt template
        operators: Optional list of specific operators to generate
        force: Overwrite existing files
        python_path: Python interpreter path for prompts
    """
    if dataset not in DATASET_OPERATORS:
        raise ValueError(f"Unknown dataset: {dataset}. Available: {list(DATASET_OPERATORS.keys())}")

    # Load template
    template = load_template(template_path)

    # Get operators dict
    ops_dict = DATASET_OPERATORS[dataset]

    # Determine namespace and flatten operators
    if dataset == "cupy":
        namespace = "cupy"
        # cupy operators are already in flat format (cupy::xxx -> func)
        flat_ops = ops_dict
    else:
        namespace = "aten"
        flat_ops = flatten_operator_dict(ops_dict, namespace)

    # For cupy, we need to handle the signature lookup differently
    # since cupy operators don't have torch.ops schemas
    is_cupy = (dataset == "cupy")

    # Create output directory
    dataset_output_dir = output_dir / dataset
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
        # Extract operator name
        op_name = full_name.split("::")[-1]

        # Output file path
        prompt_path = dataset_output_dir / f"{op_name}.md"

        # Skip if exists and not force
        if prompt_path.exists() and not force:
            skipped += 1
            continue

        # Get operator info
        if is_cupy:
            # cupy operators don't have torch.ops schemas, use placeholder
            op_info = {
                "signatures": f"- `{op_name}`: (See cuBLAS documentation)",
                "impl_info": f"- `{op_name}`",
                "input_args": "(See cuBLAS documentation for parameter details)",
            }
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
        "--template",
        type=Path,
        default=SCRIPT_DIR / "templates" / "triton_kernel.md",
        help="Path to prompt template"
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
                template_path=args.template,
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
