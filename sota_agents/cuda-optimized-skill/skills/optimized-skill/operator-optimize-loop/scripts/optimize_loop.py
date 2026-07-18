#!/usr/bin/env python3

# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Evaluate one operator iteration inside an optimization loop.

This script does the repeatable mechanics for each version:
1. Snapshot the current operator implementation into a run directory.
2. Run correctness validation + benchmark via benchmark.py.
3. Generate targeted and full Nsight Compute reports when the backend supports NCU.
4. Import summary/details text from generated .ncu-rep files.
5. Update a run manifest and final summary so Claude can compare iterations.

Code generation and optimization decisions are intentionally left to the skill.
The skill edits or creates the next operator version, then invokes this script again
for the next iteration in the same run directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


TARGETED_SECTIONS = [
    "LaunchStats",
    "Occupancy",
    "SpeedOfLight",
    "MemoryWorkloadAnalysis",
    "SchedulerStats",
]

BACKEND_REFERENCE_DOCS = {
    "cuda": [
        "skills/optimized-skill/reference/cuda/optim.md",
        "skills/optimized-skill/reference/cuda/memory-optim.md",
        "skills/optimized-skill/reference/cuda/compute-optim.md",
        "skills/optimized-skill/reference/cuda/sync-optim.md",
    ],
    "cutlass": [
        "skills/optimized-skill/reference/cutlass/cutlass-optim.md",
    ],
    "triton": [
        "skills/optimized-skill/reference/triton/triton-optim.md",
    ],
}

STRATEGY_TAGS_HEADER = "## Strategy tags"
DEFAULT_SCOPE_TOKEN = "na"


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")



def shell_join(parts: list[str]) -> str:
    return " ".join(shlex.quote(str(part)) for part in parts)



def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")



def read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))



def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")



def valid_report_exists(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 0



def run_command(cmd: list[str], stdout_path: Path, stderr_path: Path) -> subprocess.CompletedProcess:
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="ignore",
        )
    except OSError as exc:
        result = subprocess.CompletedProcess(cmd, 127, "", str(exc))
    write_text(stdout_path, result.stdout)
    write_text(stderr_path, result.stderr)
    return result



def trim_output(text: str, max_lines: int = 20) -> str:
    lines = [line.rstrip() for line in text.splitlines() if line.strip()]
    if len(lines) <= max_lines:
        return "\n".join(lines)
    return "\n".join(lines[:max_lines] + ["..."])



def add_requirement(
    requirements: list[dict[str, Any]],
    errors: list[str],
    name: str,
    ok: bool,
    detail: str,
    *,
    required: bool = True,
) -> None:
    requirements.append(
        {
            "name": name,
            "ok": ok,
            "detail": detail,
            "required": required,
        }
    )
    if required and not ok:
        errors.append(f"{name}: {detail}")



def run_probe(cmd: list[str]) -> dict[str, Any]:
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="ignore",
        )
    except OSError as exc:
        return {
            "command": shell_join(cmd),
            "returncode": 127,
            "stdout": "",
            "stderr": str(exc),
        }
    return {
        "command": shell_join(cmd),
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }



def candidate_has_path(candidate: str) -> bool:
    return any(sep in candidate for sep in ("\\", "/"))



def find_cuda_roots() -> list[Path]:
    roots: list[Path] = []
    for env_name in ("CUDA_PATH", "CUDA_HOME", "CUDA_ROOT"):
        value = os.environ.get(env_name)
        if value:
            roots.append(Path(value))
    return roots



def find_ncu_roots() -> list[Path]:
    roots: list[Path] = []
    program_files = os.environ.get("ProgramFiles")
    if program_files:
        nvidia_dir = Path(program_files) / "NVIDIA Corporation"
        if nvidia_dir.exists():
            roots.extend(sorted(nvidia_dir.glob("Nsight Compute*")))
    return roots



def resolve_executable(candidate: str, tool_name: str) -> str:
    candidate = candidate.strip().strip('"')
    direct = Path(candidate).expanduser()
    if direct.exists():
        return str(direct.resolve())

    resolved = shutil.which(candidate)
    if resolved:
        return resolved

    if candidate_has_path(candidate):
        return ""

    extra_names = [candidate]
    if os.name == "nt" and not Path(candidate).suffix:
        extra_names.extend([f"{candidate}.exe", f"{candidate}.bat", f"{candidate}.cmd"])

    search_roots: list[Path] = []
    if tool_name == "nvcc":
        search_roots.extend(root / "bin" for root in find_cuda_roots())
    elif tool_name == "ncu":
        search_roots.extend(find_ncu_roots())

    for root in search_roots:
        for name in extra_names:
            probe = root / name
            if probe.exists():
                return str(probe.resolve())
    return ""



def probe_executable(candidate: str, tool_name: str, version_args: list[str]) -> dict[str, Any]:
    resolved = resolve_executable(candidate, tool_name)
    info: dict[str, Any] = {
        "requested": candidate,
        "resolved": resolved,
        "exists": bool(resolved),
        "version_command": "",
        "version_returncode": None,
        "version_output": "",
    }
    if not resolved:
        return info

    probe = run_probe([resolved, *version_args])
    output = (probe["stdout"] or probe["stderr"]).strip()
    info["version_command"] = probe["command"]
    info["version_returncode"] = probe["returncode"]
    info["version_output"] = trim_output(output)
    return info



def probe_nvidia_smi() -> dict[str, Any]:
    resolved = shutil.which("nvidia-smi")
    info: dict[str, Any] = {
        "exists": bool(resolved),
        "resolved": resolved or "",
        "query_command": "",
        "returncode": None,
        "query_output": "",
        "gpus": [],
    }
    if not resolved:
        return info

    primary = run_probe(
        [resolved, "--query-gpu=name,compute_cap,driver_version", "--format=csv,noheader"]
    )
    probe = primary
    if primary["returncode"] != 0 or not primary["stdout"].strip():
        fallback = run_probe([resolved, "--query-gpu=name,driver_version", "--format=csv,noheader"])
        if fallback["returncode"] == 0 and fallback["stdout"].strip():
            probe = fallback

    info["query_command"] = probe["command"]
    info["returncode"] = probe["returncode"]
    info["query_output"] = trim_output((probe["stdout"] or probe["stderr"]).strip())

    if probe["returncode"] == 0:
        for line in probe["stdout"].splitlines():
            parts = [part.strip() for part in line.split(",")]
            if len(parts) >= 3:
                info["gpus"].append(
                    {
                        "name": parts[0],
                        "compute_capability": parts[1],
                        "driver_version": parts[2],
                    }
                )
            elif len(parts) >= 2:
                info["gpus"].append(
                    {
                        "name": parts[0],
                        "compute_capability": "",
                        "driver_version": parts[1],
                    }
                )
    return info



def probe_torch_cuda(gpu_index: int) -> dict[str, Any]:
    info: dict[str, Any] = {
        "importable": False,
        "version": "",
        "cuda_version": "",
        "cuda_available": False,
        "device_count": 0,
        "selected_gpu_index": gpu_index,
        "selected_gpu_name": "",
        "selected_gpu_compute_capability": "",
        "selected_sm": "",
        "error": "",
    }
    try:
        import torch  # type: ignore
    except Exception as exc:
        info["error"] = str(exc)
        return info

    info["importable"] = True
    info["version"] = getattr(torch, "__version__", "")
    info["cuda_version"] = getattr(torch.version, "cuda", "") or ""

    try:
        info["cuda_available"] = bool(torch.cuda.is_available())
        if info["cuda_available"]:
            info["device_count"] = int(torch.cuda.device_count())
            if 0 <= gpu_index < info["device_count"]:
                info["selected_gpu_name"] = torch.cuda.get_device_name(gpu_index)
                major, minor = torch.cuda.get_device_capability(gpu_index)
                info["selected_gpu_compute_capability"] = f"{major}.{minor}"
                info["selected_sm"] = f"sm_{major}{minor}"
    except Exception as exc:
        info["error"] = str(exc)
    return info



def infer_backend(solution_file: Path, backend: str) -> str:
    if backend != "auto":
        return backend
    if solution_file.suffix.lower() == ".py":
        return "triton"
    return "cuda"



def backend_supports_ncu(backend: str) -> bool:
    return backend in {"cuda", "cutlass", "triton"}



def collect_preflight(
    args: argparse.Namespace,
    benchmark_script: Path,
    solution_file: Path,
    ref_file: Path | None,
    backend: str,
) -> dict[str, Any]:
    warnings: list[str] = []
    errors: list[str] = []
    requirements: list[dict[str, Any]] = []

    preflight: dict[str, Any] = {
        "checked_at": now_iso(),
        "ready": False,
        "backend": backend,
        "python_executable": sys.executable,
        "python_version": sys.version.splitlines()[0],
        "selected_gpu_index": args.gpu,
        "env_vars": {
            "CUDA_PATH": os.environ.get("CUDA_PATH", ""),
            "CUDA_HOME": os.environ.get("CUDA_HOME", ""),
            "CUDA_ROOT": os.environ.get("CUDA_ROOT", ""),
        },
        "requirements": requirements,
        "warnings": warnings,
        "errors": errors,
    }

    add_requirement(requirements, errors, "solution file", solution_file.exists(), str(solution_file))
    expected_suffix_ok = solution_file.suffix.lower() == ".py" if backend == "triton" else solution_file.suffix.lower() == ".cu"
    add_requirement(
        requirements,
        errors,
        "solution suffix",
        expected_suffix_ok,
        solution_file.suffix or "(no suffix)",
    )
    add_requirement(requirements, errors, "benchmark.py", benchmark_script.exists(), str(benchmark_script))
    add_requirement(
        requirements,
        errors,
        "reference file",
        ref_file is None or ref_file.exists(),
        "not provided" if ref_file is None else str(ref_file),
    )

    torch_info = probe_torch_cuda(args.gpu)
    preflight["torch"] = torch_info
    add_requirement(
        requirements,
        errors,
        "PyTorch import",
        torch_info["importable"],
        torch_info["version"] if torch_info["importable"] else (torch_info["error"] or "torch import failed"),
    )
    add_requirement(
        requirements,
        errors,
        "CUDA runtime",
        torch_info["cuda_available"],
        f"torch CUDA {torch_info['cuda_version']}" if torch_info["cuda_available"] else (torch_info["error"] or "torch.cuda.is_available() returned false"),
    )
    selected_gpu_ok = torch_info["cuda_available"] and 0 <= args.gpu < int(torch_info["device_count"])
    add_requirement(
        requirements,
        errors,
        f"GPU index {args.gpu}",
        selected_gpu_ok,
        f"{torch_info['selected_gpu_name']} ({torch_info['selected_sm']})" if selected_gpu_ok else f"available device count: {torch_info['device_count']}",
    )

    nvidia_smi_info = probe_nvidia_smi()
    preflight["nvidia_smi"] = nvidia_smi_info
    if not nvidia_smi_info["exists"]:
        warnings.append("nvidia-smi not found; GPU model falls back to PyTorch detection.")
    elif nvidia_smi_info.get("returncode") not in (None, 0):
        warnings.append("nvidia-smi is present but GPU query failed.")

    gpu_info: dict[str, Any] = {
        "name": torch_info.get("selected_gpu_name", ""),
        "compute_capability": torch_info.get("selected_gpu_compute_capability", ""),
        "sm": torch_info.get("selected_sm", ""),
        "driver_version": "",
        "source": "torch" if torch_info.get("selected_gpu_name") else "",
    }
    if nvidia_smi_info.get("gpus") and args.gpu < len(nvidia_smi_info["gpus"]):
        smi_gpu = nvidia_smi_info["gpus"][args.gpu]
        if smi_gpu.get("name"):
            gpu_info["name"] = smi_gpu["name"]
            gpu_info["source"] = "nvidia-smi"
        if smi_gpu.get("compute_capability"):
            gpu_info["compute_capability"] = smi_gpu["compute_capability"]
            if not gpu_info["sm"] and "." in smi_gpu["compute_capability"]:
                major, minor = smi_gpu["compute_capability"].split(".", 1)
                gpu_info["sm"] = f"sm_{major}{minor}"
        if smi_gpu.get("driver_version"):
            gpu_info["driver_version"] = smi_gpu["driver_version"]
    preflight["gpu"] = gpu_info

    if backend in {"cuda", "cutlass"}:
        nvcc_info = probe_executable(args.nvcc_bin, "nvcc", ["--version"])
        preflight["nvcc"] = nvcc_info
        add_requirement(
            requirements,
            errors,
            "nvcc executable",
            nvcc_info["exists"],
            nvcc_info["resolved"] or f"cannot resolve {args.nvcc_bin}",
        )
        if nvcc_info["exists"] and nvcc_info.get("version_returncode") not in (None, 0):
            warnings.append("nvcc exists but `--version` did not exit cleanly.")
    else:
        preflight["nvcc"] = {
            "requested": args.nvcc_bin,
            "resolved": "",
            "exists": False,
            "version_command": "",
            "version_returncode": None,
            "version_output": "not required for triton backend",
        }

    if backend_supports_ncu(backend):
        ncu_info = probe_executable(args.ncu_bin, "ncu", ["--version"])
        preflight["ncu"] = ncu_info
        add_requirement(
            requirements,
            errors,
            "ncu executable",
            ncu_info["exists"],
            ncu_info["resolved"] or f"cannot resolve {args.ncu_bin}",
        )
        if ncu_info["exists"] and ncu_info.get("version_returncode") not in (None, 0):
            warnings.append("ncu exists but `--version` did not exit cleanly.")
    else:
        preflight["ncu"] = {
            "requested": args.ncu_bin,
            "resolved": "",
            "exists": False,
            "version_command": "",
            "version_returncode": None,
            "version_output": "not required for this backend",
        }

    if args.arch and gpu_info.get("sm") and args.arch != gpu_info["sm"]:
        warnings.append(f"--arch={args.arch} does not match selected GPU capability {gpu_info['sm']}.")

    preflight["ready"] = not errors
    return preflight



def sanitize_token(text: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "_", text.strip())
    return cleaned.strip("_")[:64] or DEFAULT_SCOPE_TOKEN


def build_scope_key(backend: str, source_file: Path, ref_file: Path | None, dims_args: list[str], arch: str) -> str:
    source_token = sanitize_token(source_file.stem)
    ref_token = sanitize_token(ref_file.stem) if ref_file else "no_ref"
    dims_token = hashlib.sha1("|".join(sorted(dims_args)).encode("utf-8")).hexdigest()[:12]
    arch_token = sanitize_token(arch) if arch else "auto_arch"
    return f"{backend}__{source_token}__{ref_token}__{arch_token}__{dims_token}"


def default_strategy_memory(global_file: Path, scope_key: str) -> dict[str, Any]:
    return {
        "version": 1,
        "scope_key": scope_key,
        "fingerprint_algo": "sha1-16",
        "current_run": {
            "seen_order": [],
            "positive": {},
            "negative": {},
            "rejected": {},
        },
        "global_sync": {
            "enabled": True,
            "global_file": str(global_file),
            "loaded_at": "",
            "updated_at": "",
        },
    }


def load_global_strategy_memory(global_file: Path) -> dict[str, Any]:
    payload = read_json(global_file, None)
    if payload is None:
        return {
            "version": 1,
            "updated_at": "",
            "scopes": {},
        }
    payload.setdefault("version", 1)
    payload.setdefault("updated_at", "")
    payload.setdefault("scopes", {})
    return payload


def save_global_strategy_memory(global_file: Path, payload: dict[str, Any]) -> None:
    payload["updated_at"] = now_iso()
    write_json(global_file, payload)


def ensure_strategy_memory(manifest: dict[str, Any], scope_key: str, global_file: Path) -> dict[str, Any]:
    strategy_memory = manifest.get("strategy_memory")
    if not isinstance(strategy_memory, dict):
        strategy_memory = default_strategy_memory(global_file, scope_key)
        manifest["strategy_memory"] = strategy_memory

    strategy_memory.setdefault("version", 1)
    strategy_memory["scope_key"] = scope_key
    strategy_memory.setdefault("fingerprint_algo", "sha1-16")

    current_run = strategy_memory.setdefault("current_run", {})
    current_run.setdefault("seen_order", [])
    current_run.setdefault("positive", {})
    current_run.setdefault("negative", {})
    current_run.setdefault("rejected", {})

    global_sync = strategy_memory.setdefault("global_sync", {})
    global_sync.setdefault("enabled", True)
    global_sync["global_file"] = str(global_file)
    global_sync.setdefault("loaded_at", "")
    global_sync.setdefault("updated_at", "")

    return strategy_memory


def normalize_strategy_tags(tags: list[str]) -> list[str]:
    normalized = []
    for tag in tags:
        clean = re.sub(r"\s+", "_", tag.strip().lower())
        clean = re.sub(r"[^a-z0-9_\-]", "", clean)
        if clean:
            normalized.append(clean)
    return sorted(set(normalized))


def extract_strategy_tags(proposal_path: Path) -> list[str]:
    if not proposal_path.exists():
        return []
    content = proposal_path.read_text(encoding="utf-8")
    lines = content.splitlines()
    tags: list[str] = []
    in_section = False
    for line in lines:
        stripped = line.strip()
        if stripped.lower() == STRATEGY_TAGS_HEADER.lower():
            in_section = True
            continue
        if in_section and stripped.startswith("## "):
            break
        if in_section and stripped.startswith("-"):
            tags.append(stripped.lstrip("-").strip())
    return normalize_strategy_tags(tags)


def build_strategy_fingerprint(backend: str, tags: list[str]) -> str:
    canonical = {
        "backend": backend,
        "tags": tags,
    }
    raw = json.dumps(canonical, sort_keys=True, ensure_ascii=False)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def get_kernel_median_ms(record: dict[str, Any]) -> float | None:
    bench = record.get("benchmark_result") or {}
    kernel = bench.get("kernel") or {}
    median = kernel.get("median_ms")
    if median is None:
        return None
    try:
        return float(median)
    except (TypeError, ValueError):
        return None


def classify_strategy_outcome(record: dict[str, Any], previous_record: dict[str, Any] | None) -> tuple[str, str]:
    bench = record.get("benchmark_result") or {}
    correctness = bench.get("correctness") or {}

    if record.get("benchmark_rc") != 0:
        return ("rejected", "benchmark_failed")
    if bench.get("has_reference") and correctness.get("passed") is False:
        return ("rejected", "correctness_failed")
    if record.get("targeted_ncu_rc") not in (None, 0):
        return ("rejected", "targeted_ncu_failed")
    if record.get("full_ncu_rc") not in (None, 0):
        return ("rejected", "full_ncu_failed")
    if record.get("ncu_expected") and not record.get("full_report_exists"):
        return ("rejected", "ncu_incomplete")

    current_median = get_kernel_median_ms(record)
    if previous_record is None:
        return ("positive", "baseline_seed")
    if current_median is None:
        return ("rejected", "no_current_median")

    previous_median = get_kernel_median_ms(previous_record)
    if previous_median is None:
        return ("rejected", "no_previous_median")

    if current_median < previous_median:
        return ("positive", "faster_than_previous")
    return ("negative", "slower_or_equal_to_previous")


def update_memory_bucket(bucket: dict[str, Any], fingerprint: str, tags: list[str], iteration: int, reason: str, outcome: str, record: dict[str, Any], previous_record: dict[str, Any] | None) -> None:
    current_median = get_kernel_median_ms(record)
    previous_median = get_kernel_median_ms(previous_record) if previous_record else None

    item = bucket.get(fingerprint)
    if item is None:
        item = {
            "tags": tags,
            "first_iteration": iteration,
            "last_iteration": iteration,
            "count": 0,
            "last_outcome": outcome,
            "last_reason": reason,
            "evidence": {
                "baseline_iteration": previous_record.get("iteration") if previous_record else None,
                "baseline_median_ms": previous_median,
                "current_median_ms": current_median,
            },
        }
        bucket[fingerprint] = item

    item["last_iteration"] = iteration
    item["count"] = int(item.get("count", 0)) + 1
    item["last_outcome"] = outcome
    item["last_reason"] = reason
    item["tags"] = tags
    item["evidence"] = {
        "baseline_iteration": previous_record.get("iteration") if previous_record else None,
        "baseline_median_ms": previous_median,
        "current_median_ms": current_median,
    }


def merge_strategy_constraints(run_memory: dict[str, Any], global_scope: dict[str, Any]) -> dict[str, list[str]]:
    run_data = run_memory.get("current_run") or {}
    blocked = set((run_data.get("negative") or {}).keys())
    blocked.update((run_data.get("rejected") or {}).keys())

    preferred = set((run_data.get("positive") or {}).keys())

    blocked.update((global_scope.get("negative") or {}).keys())
    blocked.update((global_scope.get("rejected") or {}).keys())
    preferred.update((global_scope.get("positive") or {}).keys())

    return {
        "blocked": sorted(blocked),
        "preferred": sorted(preferred),
    }


def render_preflight_markdown(preflight: dict[str, Any]) -> str:
    lines = [
        "# Operator Optimization Loop Preflight",
        "",
        "## Status",
        f"- ready: {'yes' if preflight.get('ready') else 'no'}",
        f"- checked at: {preflight.get('checked_at', '')}",
        f"- backend: {preflight.get('backend', 'unknown')}",
        f"- python: {preflight.get('python_executable', '')}",
        f"- python version: {preflight.get('python_version', '')}",
        f"- selected gpu index: {preflight.get('selected_gpu_index')}",
        "",
        "## Required environment",
        "",
        "| Requirement | Status | Detail |",
        "| --- | --- | --- |",
    ]

    for item in preflight.get("requirements", []):
        status = "ok" if item.get("ok") else "missing"
        detail = str(item.get("detail", "")).replace("\n", "<br>")
        lines.append(f"| {item.get('name')} | {status} | {detail} |")

    gpu = preflight.get("gpu") or {}
    torch_info = preflight.get("torch") or {}
    nvidia_smi = preflight.get("nvidia_smi") or {}
    nvcc = preflight.get("nvcc") or {}
    ncu = preflight.get("ncu") or {}

    lines.extend(
        [
            "",
            "## GPU",
            f"- model: {gpu.get('name') or 'unknown'}",
            f"- compute capability: {gpu.get('compute_capability') or 'unknown'}",
            f"- sm: {gpu.get('sm') or 'unknown'}",
            f"- driver version: {gpu.get('driver_version') or 'unknown'}",
            f"- source: {gpu.get('source') or 'unknown'}",
            f"- torch: {torch_info.get('version') or 'not importable'}",
            f"- torch cuda: {torch_info.get('cuda_version') or 'unknown'}",
            f"- device count: {torch_info.get('device_count')}",
            f"- nvidia-smi: {nvidia_smi.get('resolved') or 'not found'}",
            "",
            "## Tools",
            f"- nvcc requested: {nvcc.get('requested', '')}",
            f"- nvcc resolved: {nvcc.get('resolved') or 'not required'}",
            f"- nvcc version: {nvcc.get('version_output') or 'n/a'}",
            f"- ncu requested: {ncu.get('requested', '')}",
            f"- ncu resolved: {ncu.get('resolved') or 'not required'}",
            f"- ncu version: {ncu.get('version_output') or 'n/a'}",
            "",
            "## Environment variables",
            f"- CUDA_PATH: {preflight.get('env_vars', {}).get('CUDA_PATH') or '(unset)'}",
            f"- CUDA_HOME: {preflight.get('env_vars', {}).get('CUDA_HOME') or '(unset)'}",
            f"- CUDA_ROOT: {preflight.get('env_vars', {}).get('CUDA_ROOT') or '(unset)'}",
            "",
            "## Errors",
        ]
    )

    if preflight.get("errors"):
        lines.extend(f"- {item}" for item in preflight["errors"])
    else:
        lines.append("- none")

    lines.extend(["", "## Warnings"])
    if preflight.get("warnings"):
        lines.extend(f"- {item}" for item in preflight["warnings"])
    else:
        lines.append("- none")

    return "\n".join(lines)



def ensure_run_dir(solution_file: Path, run_dir_arg: str) -> Path:
    if run_dir_arg:
        run_dir = Path(run_dir_arg).resolve()
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = (solution_file.resolve().parent / "optimize_runs" / f"run_{ts}").resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir



def load_manifest(manifest_path: Path, args: argparse.Namespace, run_dir: Path, solution_file: Path, backend: str) -> dict[str, Any]:
    manifest = read_json(manifest_path, None)
    if manifest is None:
        manifest = {
            "created_at": now_iso(),
            "updated_at": now_iso(),
            "run_dir": str(run_dir),
            "backend": backend,
            "source_file": str(solution_file.resolve()),
            "source_cu_file": str(solution_file.resolve()),
            "reference_file": str(Path(args.ref).resolve()) if args.ref else "",
            "max_iterations": args.max_iterations,
            "warmup": args.warmup,
            "repeat": args.repeat,
            "gpu": args.gpu,
            "arch": args.arch,
            "ptr_size": args.ptr_size,
            "seed": args.seed,
            "dims_args": list(args.dim_args),
            "reference_docs": BACKEND_REFERENCE_DOCS.get(backend, []),
            "ncu_supported": backend_supports_ncu(backend),
            "preflight": {},
            "iterations": [],
            "best_iteration": None,
            "best_kernel_path": "",
        }
    else:
        manifest["backend"] = backend
        manifest["source_file"] = str(solution_file.resolve())
        manifest["source_cu_file"] = str(solution_file.resolve())
        manifest["reference_docs"] = BACKEND_REFERENCE_DOCS.get(backend, [])
        manifest["ncu_supported"] = backend_supports_ncu(backend)
    return manifest



def pick_iteration_index(manifest: dict[str, Any], requested_iteration: int) -> int:
    if requested_iteration >= 0:
        return requested_iteration
    return len(manifest.get("iterations", []))



def build_benchmark_cmd(args: argparse.Namespace, benchmark_script: Path, snapshot_file: Path, benchmark_json: Path, backend: str) -> list[str]:
    cmd = [
        sys.executable,
        str(benchmark_script),
        str(snapshot_file),
        f"--backend={backend}",
        f"--warmup={args.warmup}",
        f"--repeat={args.repeat}",
        f"--gpu={args.gpu}",
        f"--seed={args.seed}",
        f"--atol={args.atol}",
        f"--rtol={args.rtol}",
        f"--json-out={benchmark_json}",
    ]
    if backend in {"cuda", "cutlass"}:
        cmd.append(f"--nvcc-bin={args.nvcc_bin}")
    if args.ref:
        cmd.append(f"--ref={Path(args.ref).resolve()}")
    if args.arch:
        cmd.append(f"--arch={args.arch}")
    if args.ptr_size > 0:
        cmd.append(f"--ptr-size={args.ptr_size}")
    cmd.extend(args.dim_args)
    return cmd



def build_targeted_ncu_cmd(args: argparse.Namespace, bench_cmd: list[str], out_prefix: Path) -> list[str]:
    cmd = [
        args.ncu_bin,
        "--target-processes",
        "all",
        "--profile-from-start",
        "on",
        "--launch-skip",
        str(args.launch_skip),
        "--launch-count",
        str(args.launch_count),
    ]
    if args.kernel_name_regex:
        cmd.extend(["--kernel-name-base", "demangled", "-k", f"regex:{args.kernel_name_regex}"])
    for section in TARGETED_SECTIONS:
        cmd.extend(["--section", section])
    cmd.extend(["-o", str(out_prefix), "-f"])
    cmd.extend(bench_cmd)
    return cmd



def build_full_ncu_cmd(args: argparse.Namespace, bench_cmd: list[str], out_prefix: Path) -> list[str]:
    cmd = [
        args.ncu_bin,
        "--target-processes",
        "all",
        "--profile-from-start",
        "on",
        "--launch-skip",
        str(args.launch_skip),
        "--launch-count",
        str(args.launch_count),
        "--set",
        "full",
    ]
    if args.kernel_name_regex:
        cmd.extend(["--kernel-name-base", "demangled", "-k", f"regex:{args.kernel_name_regex}"])
    cmd.extend(["-o", str(out_prefix), "-f"])
    cmd.extend(bench_cmd)
    return cmd



def import_ncu_report(args: argparse.Namespace, rep_path: Path, summary_txt: Path, details_txt: Path) -> dict[str, Any]:
    summary_cmd = [args.ncu_bin, "--import", str(rep_path), "--print-summary", "per-kernel"]
    details_cmd = [args.ncu_bin, "--import", str(rep_path), "--page", "details"]

    summary_res = run_command(summary_cmd, summary_txt, summary_txt.with_suffix(".stderr.txt"))
    details_res = run_command(details_cmd, details_txt, details_txt.with_suffix(".stderr.txt"))

    return {
        "summary_command": shell_join(summary_cmd),
        "summary_txt": str(summary_txt),
        "summary_rc": summary_res.returncode,
        "details_command": shell_join(details_cmd),
        "details_txt": str(details_txt),
        "details_rc": details_res.returncode,
    }



def choose_best_iteration(iterations: list[dict[str, Any]]) -> dict[str, Any] | None:
    candidates = []
    for item in iterations:
        bench = item.get("benchmark_result") or {}
        kernel = bench.get("kernel") or {}
        correctness = bench.get("correctness") or {}
        ncu_required = item.get("ncu_expected", False)
        if ncu_required and not item.get("full_report_exists"):
            continue
        if bench.get("has_reference") and not correctness.get("passed"):
            continue
        if kernel.get("median_ms") is None or kernel.get("average_ms") is None:
            continue
        candidates.append(item)
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda item: (
            item["benchmark_result"]["kernel"]["median_ms"],
            item["benchmark_result"]["kernel"]["average_ms"],
            item["iteration"],
        ),
    )



def render_iteration_markdown(record: dict[str, Any]) -> str:
    bench = record.get("benchmark_result") or {}
    correctness = bench.get("correctness") or {}
    kernel = bench.get("kernel") or {}
    reference = bench.get("reference") or {}
    strategy = record.get("strategy") or {}
    constraints = strategy.get("constraints") or {}
    lines = [
        f"# Iteration v{record['iteration']}",
        "",
        "## Status",
        f"- backend: {record.get('backend', 'unknown')}",
        f"- benchmark rc: {record['benchmark_rc']}",
        f"- targeted ncu rc: {record.get('targeted_ncu_rc')}",
        f"- full ncu rc: {record.get('full_ncu_rc')}",
        f"- snapshot file: {record['snapshot_file']}",
        f"- correctness checked: {correctness.get('checked')}",
        f"- correctness passed: {correctness.get('passed')}",
        "",
        "## Strategy memory",
        f"- tags: {', '.join(strategy.get('tags') or []) or 'none'}",
        f"- fingerprint: {strategy.get('fingerprint') or 'none'}",
        f"- outcome: {strategy.get('outcome') or 'pending'}",
        f"- reason: {strategy.get('reason') or 'not_available'}",
        f"- blocked fingerprints: {', '.join(constraints.get('blocked') or []) or 'none'}",
        f"- preferred fingerprints: {', '.join(constraints.get('preferred') or []) or 'none'}",
        "",
        "## Commands",
        f"- benchmark: `{record['benchmark_command']}`",
        f"- targeted ncu: `{record.get('targeted_ncu_command', '')}`",
        f"- full ncu: `{record.get('full_ncu_command', '')}`",
        "",
        "## Benchmark",
        f"- kernel average ms: {kernel.get('average_ms')}",
        f"- kernel median ms: {kernel.get('median_ms')}",
        f"- kernel min ms: {kernel.get('min_ms')}",
        f"- kernel max ms: {kernel.get('max_ms')}",
        f"- speedup vs reference: {bench.get('speedup_vs_reference')}",
        f"- reference average ms: {reference.get('average_ms')}",
        "",
        "## Artifacts",
        f"- benchmark json: {record['benchmark_json']}",
        f"- targeted report: {record.get('targeted_report')}",
        f"- full report: {record.get('full_report')}",
        f"- targeted summary: {record.get('targeted_import', {}).get('summary_txt')}",
        f"- full summary: {record.get('full_import', {}).get('summary_txt')}",
        f"- targeted details: {record.get('targeted_import', {}).get('details_txt')}",
        f"- full details: {record.get('full_import', {}).get('details_txt')}",
        "",
        "## Claude follow-up",
        "- Read the targeted/full summaries and details before deciding the next optimization.",
        "- Avoid blocked fingerprints and prioritize preferred fingerprints.",
        "- Write the optimization hypothesis into optimization_proposal.md for this iteration.",
        "- Only promote this version as best when correctness passes and the full report exists.",
        "",
    ]
    return "\n".join(lines)



def render_final_summary(manifest: dict[str, Any]) -> str:
    preflight = manifest.get("preflight") or {}
    backend = manifest.get("backend", "unknown")
    strategy_memory = manifest.get("strategy_memory") or {}
    current_run_memory = strategy_memory.get("current_run") or {}
    global_sync = strategy_memory.get("global_sync") or {}

    lines = [
        "# Operator Optimization Loop Summary",
        "",
        "## Run info",
        f"- run dir: {manifest['run_dir']}",
        f"- source file: {manifest['source_file']}",
        f"- backend: {backend}",
        f"- reference file: {manifest['reference_file'] or 'not provided'}",
        f"- max iterations: {manifest['max_iterations']}",
        f"- warmup: {manifest['warmup']}",
        f"- repeat: {manifest['repeat']}",
        f"- gpu: {manifest['gpu']}",
        f"- arch: {manifest['arch'] or 'auto'}",
        f"- preflight ready: {'yes' if preflight.get('ready') else 'no'}" if preflight else "- preflight ready: not run",
        f"- ncu supported: {'yes' if manifest.get('ncu_supported') else 'no'}",
        "",
        "## Reference docs",
    ]

    docs = manifest.get("reference_docs") or []
    if docs:
        lines.extend(f"- {item}" for item in docs)
    else:
        lines.append("- none")

    lines.extend(
        [
            "",
            "## Environment",
            f"- gpu name: {preflight.get('gpu_name') or 'unknown'}" if preflight else "- gpu name: unknown",
            f"- compute capability: {preflight.get('gpu_compute_capability') or 'unknown'}" if preflight else "- compute capability: unknown",
            f"- nvcc: {preflight.get('nvcc_bin') or 'not required'}" if preflight else "- nvcc: unknown",
            f"- ncu: {preflight.get('ncu_bin') or 'not required'}" if preflight else "- ncu: unknown",
            f"- preflight report: {preflight.get('markdown_path')}" if preflight.get("markdown_path") else "- preflight report: not generated",
            "",
            "## Strategy memory",
            f"- scope key: {strategy_memory.get('scope_key') or 'not initialized'}",
            f"- fingerprint algo: {strategy_memory.get('fingerprint_algo') or 'n/a'}",
            f"- seen strategies: {len(current_run_memory.get('seen_order') or [])}",
            f"- positive strategies: {len((current_run_memory.get('positive') or {}).keys())}",
            f"- negative strategies: {len((current_run_memory.get('negative') or {}).keys())}",
            f"- rejected strategies: {len((current_run_memory.get('rejected') or {}).keys())}",
            f"- global memory file: {global_sync.get('global_file') or 'n/a'}",
            f"- global loaded at: {global_sync.get('loaded_at') or 'n/a'}",
            f"- global updated at: {global_sync.get('updated_at') or 'n/a'}",
            "",
            "## Iterations",
            "",
            "| Iter | Backend | Strategy | Outcome | Correctness | Kernel median ms | Kernel avg ms | Full NCU | Snapshot |",
            "| --- | --- | --- | --- | --- | ---: | ---: | --- | --- |",
        ]
    )

    for item in manifest.get("iterations", []):
        bench = item.get("benchmark_result") or {}
        correctness = bench.get("correctness") or {}
        kernel = bench.get("kernel") or {}
        strategy = item.get("strategy") or {}
        if not bench.get("has_reference"):
            correctness_text = "not checked"
        else:
            correctness_text = "pass" if correctness.get("passed") else "fail"
        lines.append(
            "| v{iteration} | {backend} | {strategy_fp} | {outcome} | {correctness} | {median} | {avg} | {full_report} | {snapshot} |".format(
                iteration=item.get("iteration"),
                backend=item.get("backend", backend),
                strategy_fp=(strategy.get("fingerprint") or "none")[:12],
                outcome=strategy.get("outcome") or "pending",
                correctness=correctness_text,
                median=kernel.get("median_ms", "-"),
                avg=kernel.get("average_ms", "-"),
                full_report="yes" if item.get("full_report_exists") else "no",
                snapshot=item.get("snapshot_file", "-"),
            )
        )

    best_iteration = manifest.get("best_iteration")
    lines.extend(["", "## Best version"])
    if best_iteration is None:
        lines.append("- No eligible best version yet. Need a benchmark-successful iteration, correctness pass when reference exists, and full NCU report.")
    else:
        best = next(item for item in manifest["iterations"] if item["iteration"] == best_iteration)
        bench = best.get("benchmark_result") or {}
        kernel = bench.get("kernel") or {}
        strategy = best.get("strategy") or {}
        lines.extend(
            [
                f"- best iteration: v{best_iteration}",
                f"- best file path: {best.get('snapshot_file')}",
                f"- best strategy fingerprint: {strategy.get('fingerprint') or 'none'}",
                f"- best strategy tags: {', '.join(strategy.get('tags') or []) or 'none'}",
                f"- full NCU report: {best.get('full_report') or 'n/a'}",
                f"- targeted NCU report: {best.get('targeted_report') or 'n/a'}",
                f"- kernel median ms: {kernel.get('median_ms')}",
                f"- kernel average ms: {kernel.get('average_ms')}",
                f"- speedup vs reference: {bench.get('speedup_vs_reference')}",
                f"- full NCU import summary: {best.get('full_import', {}).get('summary_txt') or 'n/a'}",
                f"- full NCU import details: {best.get('full_import', {}).get('details_txt') or 'n/a'}",
            ]
        )

    lines.extend(
        [
            "",
            "## Strategy memory details",
            "### Positive fingerprints",
        ]
    )
    positive = current_run_memory.get("positive") or {}
    if positive:
        for fp, item in positive.items():
            lines.append(f"- {fp}: tags={','.join(item.get('tags') or [])} reason={item.get('last_reason')} count={item.get('count')}")
    else:
        lines.append("- none")

    lines.extend(["", "### Negative fingerprints"])
    negative = current_run_memory.get("negative") or {}
    if negative:
        for fp, item in negative.items():
            lines.append(f"- {fp}: tags={','.join(item.get('tags') or [])} reason={item.get('last_reason')} count={item.get('count')}")
    else:
        lines.append("- none")

    lines.extend(["", "### Rejected fingerprints"])
    rejected = current_run_memory.get("rejected") or {}
    if rejected:
        for fp, item in rejected.items():
            lines.append(f"- {fp}: tags={','.join(item.get('tags') or [])} reason={item.get('last_reason')} count={item.get('count')}")
    else:
        lines.append("- none")

    lines.extend(
        [
            "",
            "## Required final answer checklist",
            "- Compare baseline vs best benchmark numbers.",
            "- Cite the best full NCU report path.",
            "- Summarize the bottleneck and the winning optimization idea.",
            "- Mention any failed iterations and why they were rejected.",
            "- Include strategy memory outcome (positive/negative/rejected) and whether blocked fingerprints were avoided.",
            "",
        ]
    )
    return "\n".join(lines)



def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description="Evaluate one iteration in an operator optimization loop")
    parser.add_argument("solution_file", help="Path to the current operator file (.cu or .py)")
    parser.add_argument("--backend", type=str, default="auto", choices=["auto", "cuda", "cutlass", "triton"], help="Backend type")
    parser.add_argument("--ref", type=str, default="", help="Optional Python reference file")
    parser.add_argument("--run-dir", type=str, default="", help="Existing or new run directory")
    parser.add_argument("--iteration", type=int, default=-1, help="Iteration index; defaults to next index")
    parser.add_argument("--max-iterations", type=int, required=True, help="Required maximum iterations for the run")
    parser.add_argument("--warmup", type=int, default=10, help="Warmup iterations for benchmark.py")
    parser.add_argument("--repeat", type=int, default=20, help="Benchmark repeat count for benchmark.py")
    parser.add_argument("--ptr-size", type=int, default=0, help="Pointer buffer element override for benchmark.py")
    parser.add_argument("--arch", type=str, default="", help="GPU arch, e.g. sm_90")
    parser.add_argument("--gpu", type=int, default=0, help="GPU index")
    parser.add_argument("--atol", type=float, default=1e-4, help="Correctness absolute tolerance")
    parser.add_argument("--rtol", type=float, default=1e-3, help="Correctness relative tolerance")
    parser.add_argument("--seed", type=int, default=42, help="Random seed passed to benchmark.py")
    parser.add_argument("--launch-skip", type=int, default=20, help="NCU launch-skip value")
    parser.add_argument("--launch-count", type=int, default=1, help="NCU launch-count value")
    parser.add_argument("--nvcc-bin", type=str, default="nvcc", help="NVCC executable or full path")
    parser.add_argument("--ncu-bin", type=str, default="ncu", help="Nsight Compute executable")
    parser.add_argument("--kernel-name-regex", type=str, default="", help="Optional NCU kernel filter regex")
    parser.add_argument("--preflight-only", action="store_true", help="Only run environment checks and exit")

    args, unknown = parser.parse_known_args()
    args.dim_args = [item for item in unknown if item.startswith("--") and "=" in item]
    return args, unknown



def main() -> int:
    args, unknown = parse_args()
    if any(not (item.startswith("--") and "=" in item) for item in unknown):
        bad = [item for item in unknown if not (item.startswith("--") and "=" in item)]
        print(f"Unsupported extra args: {bad}", file=sys.stderr)
        return 2

    repo_root = Path(__file__).resolve().parents[4]
    benchmark_script = repo_root / "skills" / "optimized-skill" / "kernel-benchmark" / "scripts" / "benchmark.py"
    solution_file = Path(args.solution_file).resolve()
    backend = infer_backend(solution_file, args.backend)
    ref_file = Path(args.ref).resolve() if args.ref else None

    run_dir = ensure_run_dir(solution_file, args.run_dir)
    manifest_path = run_dir / "run_manifest.json"
    summary_path = run_dir / "final_summary.md"

    scope_key = build_scope_key(backend, solution_file, ref_file, list(args.dim_args), args.arch)
    global_memory_file = repo_root / "skills" / "optimized-skill" / "operator-optimize-loop" / "strategy-memory" / "global_strategy_memory.json"

    manifest = load_manifest(manifest_path, args, run_dir, solution_file, backend)
    strategy_memory = ensure_strategy_memory(manifest, scope_key, global_memory_file)

    global_memory = load_global_strategy_memory(global_memory_file)
    global_scopes = global_memory.setdefault("scopes", {})
    global_scope = global_scopes.setdefault(
        scope_key,
        {
            "meta": {
                "backend": backend,
                "source_file": str(solution_file.resolve()),
                "reference_file": str(ref_file) if ref_file else "",
                "dims_args": list(args.dim_args),
                "arch": args.arch,
            },
            "positive": {},
            "negative": {},
            "rejected": {},
        },
    )
    strategy_memory["global_sync"]["loaded_at"] = now_iso()

    preflight = collect_preflight(args, benchmark_script, solution_file, ref_file, backend)
    preflight_json = run_dir / "preflight_check.json"
    preflight_md = run_dir / "preflight_check.md"
    write_json(preflight_json, preflight)
    write_text(preflight_md, render_preflight_markdown(preflight))
    manifest["preflight"] = {
        "checked_at": preflight["checked_at"],
        "ready": preflight["ready"],
        "gpu_name": (preflight.get("gpu") or {}).get("name", ""),
        "gpu_compute_capability": (preflight.get("gpu") or {}).get("compute_capability", ""),
        "nvcc_bin": (preflight.get("nvcc") or {}).get("resolved", ""),
        "ncu_bin": (preflight.get("ncu") or {}).get("resolved", ""),
        "json_path": str(preflight_json),
        "markdown_path": str(preflight_md),
        "errors": list(preflight.get("errors", [])),
        "warnings": list(preflight.get("warnings", [])),
    }
    write_json(manifest_path, manifest)
    write_text(summary_path, render_final_summary(manifest))

    if preflight.get("nvcc", {}).get("resolved"):
        args.nvcc_bin = preflight["nvcc"]["resolved"]
    if preflight.get("ncu", {}).get("resolved"):
        args.ncu_bin = preflight["ncu"]["resolved"]

    if args.preflight_only or not preflight.get("ready"):
        if not preflight.get("ready"):
            print(f"Preflight failed. See {preflight_md}", file=sys.stderr)
            for item in preflight.get("errors", []):
                print(f"- {item}", file=sys.stderr)
            return 2
        return 0

    iteration = pick_iteration_index(manifest, args.iteration)
    iter_dir = run_dir / f"iter_v{iteration}"
    iter_dir.mkdir(parents=True, exist_ok=True)

    constraints = merge_strategy_constraints(strategy_memory, global_scope)

    snapshot_file = iter_dir / f"{solution_file.stem}_v{iteration}{solution_file.suffix}"
    shutil.copy2(solution_file, snapshot_file)
    if ref_file:
        shutil.copy2(ref_file, iter_dir / ref_file.name)

    benchmark_json = iter_dir / "benchmark_result.json"
    benchmark_stdout = iter_dir / "benchmark.stdout.txt"
    benchmark_stderr = iter_dir / "benchmark.stderr.txt"
    bench_cmd = build_benchmark_cmd(args, benchmark_script, snapshot_file, benchmark_json, backend)
    bench_res = run_command(bench_cmd, benchmark_stdout, benchmark_stderr)
    bench_json = read_json(benchmark_json, {})

    record: dict[str, Any] = {
        "iteration": iteration,
        "created_at": now_iso(),
        "backend": backend,
        "snapshot_file": str(snapshot_file),
        "snapshot_cu": str(snapshot_file),
        "benchmark_command": shell_join(bench_cmd),
        "benchmark_stdout": str(benchmark_stdout),
        "benchmark_stderr": str(benchmark_stderr),
        "benchmark_json": str(benchmark_json),
        "benchmark_rc": bench_res.returncode,
        "benchmark_result": bench_json,
        "ncu_expected": backend_supports_ncu(backend),
        "targeted_ncu_rc": None,
        "full_ncu_rc": None,
        "targeted_report": "",
        "full_report": "",
        "targeted_report_exists": False,
        "full_report_exists": False,
        "targeted_import": {},
        "full_import": {},
    }

    correctness = (bench_json or {}).get("correctness") or {}
    correctness_failed = bool(bench_json.get("has_reference")) and correctness.get("passed") is False

    if bench_res.returncode == 0 and not correctness_failed and backend_supports_ncu(backend):
        targeted_prefix = iter_dir / "targeted"
        targeted_stdout = iter_dir / "targeted_ncu.stdout.txt"
        targeted_stderr = iter_dir / "targeted_ncu.stderr.txt"
        targeted_cmd = build_targeted_ncu_cmd(args, bench_cmd, targeted_prefix)
        targeted_res = run_command(targeted_cmd, targeted_stdout, targeted_stderr)
        targeted_rep = targeted_prefix.with_suffix(".ncu-rep")
        record["targeted_ncu_command"] = shell_join(targeted_cmd)
        record["targeted_ncu_stdout"] = str(targeted_stdout)
        record["targeted_ncu_stderr"] = str(targeted_stderr)
        record["targeted_ncu_rc"] = targeted_res.returncode
        record["targeted_report"] = str(targeted_rep)
        record["targeted_report_exists"] = valid_report_exists(targeted_rep)
        if record["targeted_report_exists"]:
            record["targeted_import"] = import_ncu_report(
                args,
                targeted_rep,
                iter_dir / "targeted_summary.txt",
                iter_dir / "targeted_details.txt",
            )

        full_prefix = iter_dir / "full"
        full_stdout = iter_dir / "full_ncu.stdout.txt"
        full_stderr = iter_dir / "full_ncu.stderr.txt"
        full_cmd = build_full_ncu_cmd(args, bench_cmd, full_prefix)
        full_res = run_command(full_cmd, full_stdout, full_stderr)
        full_rep = full_prefix.with_suffix(".ncu-rep")
        record["full_ncu_command"] = shell_join(full_cmd)
        record["full_ncu_stdout"] = str(full_stdout)
        record["full_ncu_stderr"] = str(full_stderr)
        record["full_ncu_rc"] = full_res.returncode
        record["full_report"] = str(full_rep)
        record["full_report_exists"] = valid_report_exists(full_rep)
        if record["full_report_exists"]:
            record["full_import"] = import_ncu_report(
                args,
                full_rep,
                iter_dir / "full_summary.txt",
                iter_dir / "full_details.txt",
            )
    else:
        record["targeted_ncu_command"] = ""
        record["full_ncu_command"] = ""

    proposal_path = iter_dir / "optimization_proposal.md"
    blocked_text = ", ".join(constraints.get("blocked") or []) or "none"
    preferred_text = ", ".join(constraints.get("preferred") or []) or "none"
    if not proposal_path.exists():
        if backend == "cuda":
            proposal_stub = "# Optimization proposal\n\n## Backend\n- cuda\n\n## Primary references\n- skills/optimized-skill/reference/cuda/optim.md\n- skills/optimized-skill/reference/cuda/memory-optim.md\n- skills/optimized-skill/reference/cuda/compute-optim.md\n- skills/optimized-skill/reference/cuda/sync-optim.md\n\n## Strategy constraints from memory\n- blocked fingerprints: {blocked}\n- preferred fingerprints: {preferred}\n\n## Strategy tags\n- fill_me_tag\n\n## This iteration\n- Fill in the bottleneck diagnosis from the targeted/full NCU reports.\n- Avoid blocked fingerprints and prioritize preferred fingerprints.\n- Describe the next kernel change for v{iteration_plus_one}.\n"
        elif backend == "cutlass":
            proposal_stub = "# Optimization proposal\n\n## Backend\n- cutlass\n\n## Primary references\n- skills/optimized-skill/reference/cutlass/cutlass-optim.md\n\n## Strategy constraints from memory\n- blocked fingerprints: {blocked}\n- preferred fingerprints: {preferred}\n\n## Strategy tags\n- fill_me_tag\n\n## This iteration\n- Fill in the bottleneck diagnosis from the targeted/full NCU reports.\n- Map the issue to CUTLASS-specific choices: Tensor Core path, tile shape, stage count, epilogue fusion, stream-k/split-k, swizzle, architecture-specific collective builder.\n- Avoid blocked fingerprints and prioritize preferred fingerprints.\n- Describe the next kernel change for v{iteration_plus_one}.\n"
        else:
            proposal_stub = "# Optimization proposal\n\n## Backend\n- triton\n\n## Primary references\n- skills/optimized-skill/reference/triton/triton-optim.md\n\n## Strategy constraints from memory\n- blocked fingerprints: {blocked}\n- preferred fingerprints: {preferred}\n\n## Strategy tags\n- fill_me_tag\n\n## This iteration\n- Fill in the bottleneck diagnosis from the targeted/full NCU reports and benchmark results.\n- Map the issue to Triton-specific choices: BLOCK sizes, num_warps, num_stages, coalescing, vectorization hints, swizzle, persistent kernel, split-k, fusion.\n- Avoid blocked fingerprints and prioritize preferred fingerprints.\n- Describe the next kernel change for v{iteration_plus_one}.\n"
        write_text(proposal_path, proposal_stub.format(iteration_plus_one=iteration + 1, blocked=blocked_text, preferred=preferred_text))

    strategy_source_path = None
    if iteration > 0:
        strategy_source_path = run_dir / f"iter_v{iteration - 1}" / "optimization_proposal.md"

    if strategy_source_path and strategy_source_path.exists():
        strategy_tags = extract_strategy_tags(strategy_source_path)
        if not strategy_tags:
            strategy_tags = ["unlabeled_strategy"]
    else:
        strategy_tags = ["baseline"]

    strategy_tags = normalize_strategy_tags(strategy_tags)

    strategy_fingerprint = build_strategy_fingerprint(backend, strategy_tags)

    previous_record = None
    if iteration > 0:
        previous_record = next((item for item in manifest.get("iterations", []) if item.get("iteration") == iteration - 1), None)

    outcome, reason = classify_strategy_outcome(record, previous_record)
    strategy_blocked = strategy_fingerprint in set(constraints.get("blocked") or [])
    if strategy_blocked:
        outcome = "rejected"
        reason = "blocked_strategy_reused"

    record["strategy"] = {
        "tags": strategy_tags,
        "fingerprint": strategy_fingerprint,
        "outcome": outcome,
        "reason": reason,
        "blocked_before_iteration": strategy_blocked,
        "constraints": constraints,
    }

    current_run_memory = strategy_memory.get("current_run") or {}
    seen_order = current_run_memory.setdefault("seen_order", [])
    if strategy_fingerprint not in seen_order:
        seen_order.append(strategy_fingerprint)

    if outcome == "positive":
        update_memory_bucket(current_run_memory.setdefault("positive", {}), strategy_fingerprint, strategy_tags, iteration, reason, outcome, record, previous_record)
        update_memory_bucket(global_scope.setdefault("positive", {}), strategy_fingerprint, strategy_tags, iteration, reason, outcome, record, previous_record)
    elif outcome == "negative":
        update_memory_bucket(current_run_memory.setdefault("negative", {}), strategy_fingerprint, strategy_tags, iteration, reason, outcome, record, previous_record)
        update_memory_bucket(global_scope.setdefault("negative", {}), strategy_fingerprint, strategy_tags, iteration, reason, outcome, record, previous_record)
    elif outcome == "rejected":
        update_memory_bucket(current_run_memory.setdefault("rejected", {}), strategy_fingerprint, strategy_tags, iteration, reason, outcome, record, previous_record)
        update_memory_bucket(global_scope.setdefault("rejected", {}), strategy_fingerprint, strategy_tags, iteration, reason, outcome, record, previous_record)

    strategy_memory["global_sync"]["updated_at"] = now_iso()
    save_global_strategy_memory(global_memory_file, global_memory)

    write_text(iter_dir / "iteration_summary.md", render_iteration_markdown(record))

    iterations = [item for item in manifest.get("iterations", []) if item.get("iteration") != iteration]
    iterations.append(record)
    iterations.sort(key=lambda item: item["iteration"])
    manifest["strategy_memory"] = strategy_memory
    manifest["iterations"] = iterations
    best = choose_best_iteration(iterations)
    manifest["best_iteration"] = None if best is None else best["iteration"]
    manifest["best_kernel_path"] = "" if best is None else best["snapshot_file"]
    manifest["updated_at"] = now_iso()

    write_json(manifest_path, manifest)
    write_text(summary_path, render_final_summary(manifest))

    if bench_res.returncode != 0:
        return bench_res.returncode
    if record.get("targeted_ncu_rc") not in (None, 0):
        return record["targeted_ncu_rc"]
    if record.get("full_ncu_rc") not in (None, 0):
        return record["full_ncu_rc"]
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
