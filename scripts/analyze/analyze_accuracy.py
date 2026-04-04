#!/usr/bin/env python3
"""分析 antihack 报告文件，输出 anti-hack 后的正确率（总/aten/vllm13/cublas）。

用法:
    python analyze_accuracy.py <antihack_report.json 或 results目录>
"""
import json
import sys
from pathlib import Path

TOTAL = {"aten": 110, "vllm13": 50, "cublas": 50, "total": 210}


def classify(op_name: str) -> str:
    if op_name.startswith("aten::"):
        return "aten"
    elif op_name.startswith("vllm13::"):
        return "vllm13"
    elif op_name.startswith("cublas::"):
        return "cublas"
    return "unknown"


def _add_prefix_from_kernels(run_dir: Path, op_names: list) -> list:
    """从 kernels 目录的文件名推断算子前缀。如 aten__add.py -> aten::add"""
    kernels_dir = run_dir / "kernels"
    if not kernels_dir.exists():
        return op_names
    # 建立映射: bare_name -> prefixed_name
    mapping = {}
    for f in kernels_dir.iterdir():
        if not f.name.endswith(".py"):
            continue
        stem = f.stem  # e.g. aten__add_ or cublas__cublasSgemm_v2
        for prefix in ("aten_", "vllm13_", "cublas_"):
            if stem.startswith(prefix):
                # aten__add -> aten::add, aten___softmax -> aten::_softmax
                bare = stem[len(prefix):]  # _add or __softmax
                if bare.startswith("_"):
                    bare = bare[1:]  # add or _softmax
                ns = prefix.rstrip("_")
                mapping[bare] = f"{ns}::{bare}"
                break
    result = []
    for op in op_names:
        if op in mapping:
            result.append(mapping[op])
        else:
            result.append(op)
    return result


def load_clean_passed(path: Path) -> list:
    """从各种格式的 antihack/results 文件中提取 clean passed 算子列表。"""
    with open(path) as f:
        data = json.load(f)

    # 格式0: passed_operators + hacked_operators 直接给出（cc antihack_report）
    if "passed_operators" in data:
        hacked_names = set()
        for h in data.get("hacked_operators", []):
            if isinstance(h, str):
                hacked_names.add(h)
            elif isinstance(h, dict):
                hacked_names.add(h.get("op_name", ""))
        return [op for op in data["passed_operators"] if op not in hacked_names]

    # 格式1: clean_passed_operators 直接给出（opencode antihack_report）
    if "clean_passed_operators" in data:
        return data["clean_passed_operators"]

    # 格式2: pass_at_k_results_antihack.json — operators dict with is_hack
    if "operators" in data and isinstance(data["operators"], dict):
        ops = data["operators"]
        first_val = next(iter(ops.values()), None)
        # agent_bench results.json: {status: passed/failed}
        if isinstance(first_val, dict) and "status" in first_val:
            hacked = set()
            # 尝试从同目录 antihack 文件加载 hacked list
            for ah_name in ["antihack_report.json", "antihack_l1_report.json"]:
                ah_path = path.parent / ah_name
                if ah_path.exists() and ah_path != path:
                    with open(ah_path) as f2:
                        ah = json.load(f2)
                    for h in ah.get("hacked_operators", []):
                        if isinstance(h, str):
                            hacked.add(h)
                        elif isinstance(h, dict):
                            hacked.add(h.get("op_name", ""))
            passed = [k for k, v in ops.items() if v.get("status") == "passed" and k not in hacked]
            # 如果算子名没有前缀，从 kernels 目录推断
            if passed and classify(passed[0]) == "unknown":
                passed = _add_prefix_from_kernels(path.parent, passed)
            return passed
        # pass_at_k format: {is_hack, passed, ...}
        if isinstance(first_val, dict) and "is_hack" in first_val:
            return [k for k, v in ops.items() if v.get("passed") and not v.get("is_hack")]

    # 格式3: hacked_operators 是列表 + verify_passed 数（cc antihack_report）
    if "hacked_operators" in data and "verify_passed" in data:
        hacked_names = set()
        for h in data.get("hacked_operators", []):
            if isinstance(h, str):
                hacked_names.add(h)
            elif isinstance(h, dict):
                hacked_names.add(h.get("op_name", ""))
        # 需要从同目录找 passed 列表
        for fname in ["results.json", "progress.json"]:
            rpath = path.parent / fname
            if rpath.exists():
                with open(rpath) as f2:
                    rd = json.load(f2)
                ops = rd.get("operators", rd)
                if isinstance(ops, dict):
                    passed = [k for k, v in ops.items() if isinstance(v, dict) and v.get("status") == "passed" and k not in hacked_names]
                    if passed:
                        return passed
        # fallback: 只有数字没有列表
        print(f"Warning: 只有汇总数字，无法按类型分类", file=sys.stderr)
        return []

    # 格式4: hack_details dict
    if "hack_details" in data:
        hacked = set()
        for op, info in data.get("hack_details", {}).items():
            if isinstance(info, dict) and info.get("hacked"):
                hacked.add(op)
        # 找 passed
        for fname in ["results.json", "progress.json"]:
            rpath = path.parent / fname
            if rpath.exists():
                with open(rpath) as f2:
                    rd = json.load(f2)
                ops = rd.get("operators", rd)
                if isinstance(ops, dict):
                    passed = [k for k, v in ops.items() if isinstance(v, dict) and v.get("status") == "passed" and k not in hacked]
                    if passed:
                        return passed

    print(f"Error: 无法识别文件格式: {list(data.keys())[:10]}", file=sys.stderr)
    return []


def main():
    if len(sys.argv) < 2:
        print(f"用法: python {sys.argv[0]} <antihack_report.json 或 results目录>")
        sys.exit(1)

    path = Path(sys.argv[1])
    if path.is_dir():
        # 自动查找 antihack 文件
        for name in ["antihack_report.json", "antihack_l1_report.json", "pass_at_k_results_antihack.json", "results.json"]:
            if (path / name).exists():
                path = path / name
                break
        else:
            print(f"Error: 目录中未找到 antihack/results 文件", file=sys.stderr)
            sys.exit(1)

    print(f"文件: {path}")
    clean_passed = load_clean_passed(path)

    if not clean_passed:
        print("未找到 clean passed 算子列表")
        sys.exit(1)

    counts = {"aten": 0, "vllm13": 0, "cublas": 0, "unknown": 0}
    for op in clean_passed:
        cat = classify(op)
        counts[cat] += 1

    total_clean = counts["aten"] + counts["vllm13"] + counts["cublas"]

    print(f"\nAnti-hack 后正确率:")
    print(f"  总计:   {total_clean}/{TOTAL['total']} ({total_clean/TOTAL['total']*100:.1f}%)")
    print(f"  aten:   {counts['aten']}/{TOTAL['aten']} ({counts['aten']/TOTAL['aten']*100:.1f}%)")
    print(f"  vllm13: {counts['vllm13']}/{TOTAL['vllm13']} ({counts['vllm13']/TOTAL['vllm13']*100:.1f}%)")
    print(f"  cublas: {counts['cublas']}/{TOTAL['cublas']} ({counts['cublas']/TOTAL['cublas']*100:.1f}%)")
    if counts["unknown"]:
        print(f"  unknown: {counts['unknown']}")

    # Token 效率分析
    run_dir = path.parent
    token_path = run_dir / "token_analysis.json"
    if token_path.exists():
        with open(token_path) as f:
            token_data = json.load(f)
        total_tokens = token_data.get("total", {}).get("total", 0)
        if total_tokens and total_clean > 0:
            avg = total_tokens / total_clean
            print(f"\nToken 效率:")
            print(f"  总 token: {total_tokens:,}")
            print(f"  平均每个正确算子: {avg:,.0f} tokens")
    else:
        print(f"\n(未找到 token_analysis.json)")


if __name__ == "__main__":
    main()
