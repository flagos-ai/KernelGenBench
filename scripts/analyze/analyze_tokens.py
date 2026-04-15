#!/usr/bin/env python3
"""分析 agent 测评的 token 用量。

用法:
    python analyze_tokens.py <results目录>
"""
import json
import sys
from pathlib import Path


def _count_tokens_oc(filepath: Path) -> int:
    """从 oc_output.json (JSONL) 统计 token 总量。取最后一条 step_finish 的 tokens.total。"""
    total = 0
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if obj.get("type") == "step_finish":
                tokens = obj.get("part", {}).get("tokens", {})
                t = tokens.get("total", 0)
                if t:
                    total = t  # 取最后一条（累计值）
    return total


def _count_tokens_cc(filepath: Path) -> int:
    """从 cc_output.jsonl (JSONL) 统计 token 总量。累加所有 assistant 消息的 usage。"""
    total = 0
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if obj.get("type") == "assistant":
                usage = obj.get("message", {}).get("usage", {})
                total += usage.get("input_tokens", 0)
                total += usage.get("output_tokens", 0)
                total += usage.get("cache_creation_input_tokens", 0)
                total += usage.get("cache_read_input_tokens", 0)
    return total


def scan_workspace_tokens(run_dir: Path) -> dict:
    """扫描 workspaces 目录，统计每个算子的 token 用量。

    支持带 attempt 后缀的文件名（如 cc_output_attempt0.jsonl）和旧格式（cc_output.jsonl）。
    多次 retry 的 token 会累加。

    Returns:
        {"per_op": {op_name: tokens}, "total": int, "agent_type": "oc"|"cc"|"unknown"}
    """
    ws_dir = run_dir / "workspaces"
    if not ws_dir.exists():
        return {"per_op": {}, "total": 0, "agent_type": "unknown"}

    per_op = {}
    agent_type = "unknown"
    for op_dir in sorted(ws_dir.iterdir()):
        if not op_dir.is_dir():
            continue

        op_tokens = 0

        # Look for CC output files (new format with attempt suffix + legacy)
        cc_files = sorted(op_dir.glob("cc_output_attempt*.jsonl")) + list(op_dir.glob("cc_output.jsonl"))
        if cc_files:
            agent_type = "cc"
            for f in cc_files:
                op_tokens += _count_tokens_cc(f)

        # Look for OC output files (new format with attempt suffix + legacy)
        oc_files = sorted(op_dir.glob("oc_output_attempt*.json")) + list(op_dir.glob("oc_output.json"))
        if oc_files:
            agent_type = "oc"
            for f in oc_files:
                op_tokens += _count_tokens_oc(f)

        if op_tokens > 0:
            per_op[op_dir.name] = op_tokens

    return {"per_op": per_op, "total": sum(per_op.values()), "agent_type": agent_type}


def main():
    if len(sys.argv) < 2:
        print(f"用法: python {sys.argv[0]} <results目录>")
        sys.exit(1)

    run_dir = Path(sys.argv[1])
    if not run_dir.is_dir():
        print(f"Error: {run_dir} 不是目录", file=sys.stderr)
        sys.exit(1)

    token_info = scan_workspace_tokens(run_dir)
    total_tokens = token_info["total"]
    if not total_tokens:
        print(f"(未找到 workspaces 中的 token 数据)")
        return

    agent_label = {"oc": "OpenCode", "cc": "Claude Code"}.get(token_info["agent_type"], "unknown")
    num_ops = len(token_info["per_op"])
    print(f"\nToken 用量 ({agent_label}):")
    print(f"  总 token: {total_tokens:,}")
    print(f"  算子数:   {num_ops}")
    if num_ops > 0:
        print(f"  平均每个算子: {total_tokens / num_ops:,.0f} tokens")

    # 按 token 用量排序输出每个算子
    print(f"\n{'算子':<50} {'Tokens':>12}")
    print("-" * 64)
    for op, tokens in sorted(token_info["per_op"].items(), key=lambda x: x[1], reverse=True):
        print(f"  {op:<48} {tokens:>12,}")


if __name__ == "__main__":
    main()
