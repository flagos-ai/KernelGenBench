"""
Isolated single-case runner for vllm13 ops.
Runs one test case in a fresh Python process to avoid CUDA state pollution.

Usage:
    python -m sandbox.verifier.isolated_runner \
        --test-file <path> --func-name <name> --combo-json <json> --result-file <path>
"""
import argparse
import importlib.util
import json
import sys
import traceback


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-file", required=True)
    parser.add_argument("--func-name", required=True)
    parser.add_argument("--combo-json", required=True)
    parser.add_argument("--result-file", required=True)
    args = parser.parse_args()

    # Load the test module from file
    module_name = args.func_name + "_module"
    spec = importlib.util.spec_from_file_location(module_name, args.test_file)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)

    func = getattr(mod, args.func_name)
    combo = json.loads(args.combo_json)

    # Convert list values back to tuples for parametrize compatibility
    combo = {k: tuple(v) if isinstance(v, list) else v for k, v in combo.items()}

    result = {"status": "ok", "traceback": None}
    try:
        ret = func(**combo)
        # We don't pass back the return value (CustomBenchmarkResult etc.)
        # since it's not JSON-serializable. The parent only needs pass/fail.
        result["status"] = "ok"
    except Exception:
        result["status"] = "error"
        result["traceback"] = traceback.format_exc()

    with open(args.result_file, "w") as f:
        json.dump(result, f)


if __name__ == "__main__":
    main()
