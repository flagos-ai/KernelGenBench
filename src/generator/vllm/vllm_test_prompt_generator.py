import json
from pathlib import Path
from typing import Any, Dict

DEFAULT_SIGNATURES_PATH = Path(__file__).with_name("vllm_ops_signatures_split.json")


def load_vllm_ops_signatures(path: Path = DEFAULT_SIGNATURES_PATH) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def get_vllm_op_schema(op_name: str, path: Path = DEFAULT_SIGNATURES_PATH) -> Dict[str, Any]:
    data = load_vllm_ops_signatures(path)
    if op_name not in data:
        raise KeyError(f"Operator '{op_name}' not found in {path}")
    return data[op_name]


def build_vllm_testfunc_prompt(op_name: str, path: Path = DEFAULT_SIGNATURES_PATH) -> str:
    """
    Build a test-function prompt for a vLLM custom op.
    """
    data = load_vllm_ops_signatures(path)
    schema = get_vllm_op_schema(op_name, path)

    signature = schema.get("signature")
    input_params = schema.get("input_parameters")
    output_params = schema.get("output_parameters")
    vllm_api = schema.get("vllm_api", f"_custom_ops.{op_name}(...)")
    doc = schema.get("doc")

    prompt = "You are a test function generation expert. Generate a test function for vLLM operator.\n\n"

    prompt += "## Operator Information\n"
    prompt += f"- Operator: {op_name}\n"
    prompt += f"- Function name: test_accuracy_{op_name}\n"
    prompt += f"- Label: @label(\"{op_name}\")\n"
    prompt += f"- Signature: {signature}\n"
    prompt += f"- vLLM API: from vllm import _custom_ops; {vllm_api}\n"
    prompt += f"- Triton API: flagbench.triton.{op_name}(...)\n\n"

    if doc:
        prompt += f"## Documentation\n{doc}\n\n"

    prompt += "## Requirements\n\n"

    prompt += "### 1. Structure (MUST follow exactly)\n"
    prompt += "```python\n"
    prompt += "import flagbench\n"
    prompt += "from sandbox.config import DEVICE as device\n"
    prompt += "from sandbox.verifier.test_parametrize import parametrize, label\n"
    prompt += "from sandbox.utils.accuracy_utils import gems_assert_close as assert_close\n"
    prompt += "from sandbox.utils.accuracy_utils import CustomBenchmarkResult\n"
    prompt += "import torch\n"
    prompt += "import triton\n\n"
    prompt += f"@label(\"{op_name}\")\n"
    prompt += "@parametrize(\"param1\", [val1, val2, ...])\n"
    prompt += "@parametrize(\"param2\", [val1, val2, ...])\n"
    prompt += f"def test_accuracy_{op_name}(param1, param2, ...):\n"
    prompt += "    # ===== Accuracy Test =====\n"
    prompt += "    # Create inputs\n"
    prompt += f"    # Call baseline: flagbench.baseline.{op_name}(...)\n"
    prompt += f"    # Call triton:   flagbench.triton.{op_name}(...)\n"
    prompt += "    # Compare: assert_close(act_out, ref_out, dtype)\n\n"
    prompt += "    # ===== Performance Test =====\n"
    prompt += "    # Skip small sizes\n"
    prompt += "    # Benchmark with triton.testing.do_bench\n"
    prompt += "    # Return CustomBenchmarkResult\n"
    prompt += "```\n\n"

    prompt += "### 2. Test Parameters\n"
    prompt += "Create @parametrize decorators ONLY for parameters that appear in the operator signature:\n"
    prompt += f"- Operator signature: {input_params}\n"
    prompt += "- **CRITICAL: Do NOT invent parameters that are not in the signature above.** "
    prompt += "Every @parametrize parameter MUST correspond to an actual argument of the operator. "
    prompt += "Do NOT add auxiliary variables like num_experts, seed, stride, block_size, split_factor, etc. "
    prompt += "unless they are actual parameters in the operator signature.\n"
    prompt += "- Tensor shape params: cover small (1, 32), medium (128, 512), large (1024, 4096), very large (5333, 8192). "
    prompt += "If the operator requires column counts to be aligned (e.g., multiple of 8), use aligned values instead of 497.\n"
    prompt += "- Integer/float params from signature: 3-4 representative values covering typical usage\n"
    prompt += "- Data types: @parametrize(\"dtype\", [...]) with dtypes the operator actually supports\n"
    prompt += "- Boolean/optional params from signature: include True/False or None cases\n"
    prompt += "- Goal: 30-150 parameter combinations total. Quality over quantity.\n\n"

    prompt += "### 3. Implementation Pattern\n\n"

    if output_params == "None":
        prompt += "**In-place operation (output is None):**\n"
        prompt += "```python\n"
        prompt += "# Create input tensors\n"
        prompt += "input1 = torch.randn(..., device='cuda')\n\n"
        prompt += "# Create output tensors for baseline\n"
        prompt += "ref_out1 = torch.empty(..., device='cuda')\n"
        prompt += "ref_out2 = torch.empty(..., device='cuda')\n\n"
        prompt += "# Clone for triton\n"
        prompt += "act_out1 = ref_out1.clone()\n"
        prompt += "act_out2 = ref_out2.clone()\n\n"
        prompt += "# Call baseline (modifies ref_out tensors in-place)\n"
        prompt += f"flagbench.baseline.{op_name}(input1, ..., ref_out1, ref_out2, ...)\n\n"
        prompt += "# Call triton (modifies act_out tensors in-place)\n"
        prompt += f"flagbench.triton.{op_name}(input1, ..., act_out1, act_out2, ...)\n\n"
        prompt += "# Compare mutated tensors\n"
        prompt += "assert_close(act_out1, ref_out1, dtype)\n"
        prompt += "assert_close(act_out2, ref_out2, dtype)\n"
        prompt += "```\n\n"
    else:
        prompt += "**Return value operation:**\n"
        prompt += "```python\n"
        prompt += "# Create inputs\n"
        prompt += "input1 = torch.randn(..., device='cuda')\n\n"
        prompt += "# Call baseline\n"
        prompt += f"ref_out = flagbench.baseline.{op_name}(input1, ...)\n\n"
        prompt += "# Call triton\n"
        prompt += f"act_out = flagbench.triton.{op_name}(input1, ...)\n\n"
        prompt += "# Compare\n"
        prompt += "assert_close(act_out, ref_out, dtype)\n"
        prompt += "```\n\n"

    prompt += "### 4. Performance Test (in the SAME function, after accuracy test)\n\n"
    prompt += "After the accuracy comparison, add a performance benchmark section:\n"
    prompt += "```python\n"
    prompt += "    # ===== Performance Test =====\n"
    prompt += "    # Skip small sizes for performance test\n"
    prompt += "    if <main_size_param> < <threshold>:\n"
    prompt += "        return None\n\n"
    prompt += "    # Prepare fresh data for benchmarking\n"
    prompt += "    # ... create new input/output tensors ...\n\n"
    prompt += "    # Benchmark baseline\n"
    prompt += f"    ms_baseline = triton.testing.do_bench(\n"
    prompt += f"        lambda: flagbench.baseline.{op_name}(...), warmup=25, rep=100)\n\n"
    prompt += "    # Benchmark triton\n"
    prompt += f"    ms_triton = triton.testing.do_bench(\n"
    prompt += f"        lambda: flagbench.triton.{op_name}(...), warmup=25, rep=100)\n\n"
    prompt += "    speedup = ms_baseline / ms_triton\n"
    prompt += "    return CustomBenchmarkResult(\n"
    prompt += "        ref_time=ms_baseline, res_time=ms_triton, speedup=speedup)\n"
    prompt += "```\n"
    prompt += "IMPORTANT: For in-place ops, the lambda must create fresh output tensors each call.\n\n"

    prompt += "## CRITICAL Requirements\n\n"
    prompt += "1. **Code block**: Use ```python (NO space between ``` and python)\n"
    prompt += f"2. **Label**: MUST be @label(\"{op_name}\")\n"
    prompt += f"3. **Function name**: MUST be test_accuracy_{op_name}\n"
    prompt += "4. **ONE function only**: All test cases in one function with multiple @parametrize\n"
    prompt += "5. **Imports**: import flagbench; from sandbox.config import DEVICE as device; from sandbox.verifier.test_parametrize import parametrize, label; from sandbox.utils.accuracy_utils import gems_assert_close as assert_close; from sandbox.utils.accuracy_utils import CustomBenchmarkResult; import torch; import triton\n"
    prompt += f"6. **Baseline calls**: Use flagbench.baseline.{op_name}(...) NOT vllm._custom_ops\n"
    prompt += f"7. **Triton calls**: Use flagbench.triton.{op_name}(...) NOT direct imports\n"
    prompt += "8. **Device**: All tensors on 'cuda'\n"
    prompt += "9. **Performance**: Use triton.testing.do_bench for timing, return CustomBenchmarkResult for large sizes, return None for small sizes\n"
    prompt += "10. **Output**: Only the test function code in ONE ```python block\n"
    prompt += "11. **Parameters**: NEVER create @parametrize for variables not in the operator signature. No auxiliary variables like seed, num_experts (unless in signature), stride, block_size, etc.\n\n"

    prompt += "## Example (reference style)\n"
    prompt += "```python\n"
    prompt += "import flagbench\n"
    prompt += "from sandbox.config import DEVICE as device\n"
    prompt += "from sandbox.verifier.test_parametrize import parametrize, label\n"
    prompt += "from sandbox.utils.accuracy_utils import gems_assert_close as assert_close\n"
    prompt += "from sandbox.utils.accuracy_utils import CustomBenchmarkResult\n"
    prompt += "import torch\n"
    prompt += "import triton\n\n"
    prompt += f"@label(\"{op_name}\")\n"
    prompt += "@parametrize(\"size\", [1, 32, 71, 256, 1024, 4096, 5333])\n"
    prompt += "@parametrize(\"param\", [8, 16, 32, 64])\n"
    prompt += "@parametrize(\"dtype\", [torch.float32])\n"
    prompt += f"def test_accuracy_{op_name}(size, param, dtype):\n"
    prompt += "    x = torch.randn(size, device='cuda', dtype=dtype)\n"
    prompt += f"    ref_out = flagbench.baseline.{op_name}(x, param)\n"
    prompt += f"    act_out = flagbench.triton.{op_name}(x, param)\n"
    prompt += "    assert_close(act_out, ref_out, dtype)\n\n"
    prompt += "    # ===== Performance Test =====\n"
    prompt += "    if size < 1024:\n"
    prompt += "        return None\n\n"
    prompt += "    ms_baseline = triton.testing.do_bench(\n"
    prompt += f"        lambda: flagbench.baseline.{op_name}(x.clone(), param),\n"
    prompt += "        warmup=25, rep=100)\n"
    prompt += "    ms_triton = triton.testing.do_bench(\n"
    prompt += f"        lambda: flagbench.triton.{op_name}(x.clone(), param),\n"
    prompt += "        warmup=25, rep=100)\n\n"
    prompt += "    speedup = ms_baseline / ms_triton\n"
    prompt += "    return CustomBenchmarkResult(\n"
    prompt += "        ref_time=ms_baseline, res_time=ms_triton, speedup=speedup)\n"
    prompt += "```\n"

    return prompt
