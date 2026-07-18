<!--
 Copyright 2026 FlagOS Contributors

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 -->

# AKO4ALL

Optimize the kernel in `solution/` for maximum performance, measured by `bash scripts/bench.sh`. The optimized kernel must produce outputs identical to the golden reference.

Your goal is genuine latency reduction — not maximizing the reported speedup ratio. Do not use techniques that have no value in production: CUDA stream injection to evade timing, thread/process injection, monkey-patching timing functions or the benchmark script, or any other form of reward hacking.

The kernel is a standalone Triton Python file. Do NOT wrap it in KernelBench format (no `class Model`, no `get_inputs`). The benchmark tool handles evaluation directly.

## Generation (if no kernel provided)

If `input/` is empty or contains no kernel files (`.py`):
1. Read the operator specification from `context/operator_spec.md`.
2. Generate a Triton kernel implementation for the specified operator. Save to `solution/kernel.py`.
3. Proceed directly to **Setup step 4** (Generate bench.sh) — skip steps 1–3.

## Setup

Ensure the user has populated:
- `input/` — kernel files **(optional — if empty, generate first; see above)**
- `context/` — reference materials **(optional)**
- `bench/` — benchmark script and its dependencies **(optional — if empty, uses KernelBench eval)**

Then:
1. **Analyze inputs:** Read `input/`, `context/`, `bench/`, and `HINTS.md`. Detect bench mode — if `bench/` contains files besides `kernelbench/`, use the user-provided benchmark; otherwise use default bench mode (see `bench/kernelbench/GUIDE.md`). Confirm that input shapes can be determined; if not, **stop and ask the user**.
2. **Create branch:** Create and switch to a new branch (e.g., `opt/<kernel-name>`).
3. **Initialize solution:** Create `solution/` and `scripts/` directories. Copy kernel files from `input/` to `solution/`.
4. **Generate bench.sh:** Build the bench command with adjusted paths, pipe through `2>&1 | tee _bench_output.txt`. Replace `{{BENCH_COMMAND}}` in `bench-wrapper.sh` to produce `scripts/bench.sh`.
5. **Verify environment:** Run `bash scripts/bench.sh`. Expected: `CORRECT=True`. If it fails, diagnose and fix before proceeding. Then `git add -A && git commit -m "[baseline] Initialize solution and benchmark"`.

## Optimization

- Use `bash scripts/bench.sh` to measure performance.
- Use `ncu` to profile and identify bottlenecks — do not optimize blindly.
- Leverage all available information: `context/`, `HINTS.md`, prior attempts, web search, etc.
- Follow stall rules defined in `HINTS.md`.

### Iteration Protocol

Every modification to `solution/` code followed by a `bash scripts/bench.sh` run counts as one iteration — regardless of whether the result is an improvement, regression, or failure. Number iterations sequentially (1, 2, 3, …).

**Do NOT start the next iteration until ALL steps below are completed:**

1. **Run benchmark** — `bash scripts/bench.sh iter-N` (label is required, must match `iter-N` format).
2. **Compare with best** — If `CORRECT=True` and the mean speedup is higher than the current best, this is the new best. If the result is a regression or failure, revert `solution/kernel.py` to the best version: `git checkout HEAD~1 -- solution/kernel.py` (or the commit that had the best speedup). Always keep `solution/kernel.py` as the best-known correct version.
3. **Update `ITERATIONS.md`** — Record the iteration result including whether it became the new best.
4. **Git commit** — `[iter N] Short description of optimization direction`.
