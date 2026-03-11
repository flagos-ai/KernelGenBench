# Triton Kernel Optimization Task

## Operator Information

Read `context.md` in current directory to understand the operator specification.

## Working Directory Structure

Current directory contains:
- `context.md` - Operator specification (signature, interface, examples)
- `PERFORMANCE.md` - Performance summary (if exists)
- `v1/`, `v2/`, ... - Historical versions (if exist)

## Verification Tool

**IMPORTANT**: Always use `--output-dir` to save verification results. This creates `verify.json` which will be reused by the system.

```bash
CUDA_VISIBLE_DEVICES={{GPU_ID}} python {{TOOLS_DIR}}/verify_single.py \
    --code v1/kernel.py --op {{OP_NAME}} --dataset {{DATASET}} --output-dir v1/
```

Options:
- `--output-dir <dir>`: **Required** - Save verification logs to the directory. Creates:
  - `verify.json` - Summary with passed/failed status, test counts, speedup
  - `log_0/` - Detailed test logs from verifier
- `--output-json`: Output JSON to stdout (for programmatic parsing)

The script will output:
- `PASSED: <op>` with test counts and speedup if all tests pass
- `FAILED: <op>` with error details if tests fail

After verification, check `v{N}/verify.json` for detailed results.

## Task

1. Read `context.md` to understand the operator specification
2. Check `PERFORMANCE.md` and historical versions (if exist)
3. Determine which version directory to use:
   - If no versions exist: create `v1/` directory
   - If latest version **passed** verification: create `v{N+1}/` directory
   - If latest version **failed** verification: **overwrite** it (same directory)
4. Write your implementation to `v{N}/kernel.py`
5. **Run verification with `--output-dir v{N}/`** to save results
6. **Stop immediately** after verification passes - do not create more versions

## Important Rules

- **Version numbering**: Only increment version number when previous version PASSED. Failed versions should be overwritten, not create new directories.
- **Stop on success**: Once verification passes, stop immediately. Do not continue optimizing.
- **Always use `--output-dir` when verifying** - results are reused to avoid redundant verification
- Focus on correctness first, then performance

## Output Format

After completing the task, your final code should be in `v{N}/kernel.py` where N is the version number.
