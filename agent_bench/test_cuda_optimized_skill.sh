#!/bin/bash

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

# One-click script for cuda-optimized-skill kernel optimization + verification
#
# Usage:
#   ./test_cuda_optimized_skill.sh --baseline-dir /path/to/round_0
#   ./test_cuda_optimized_skill.sh --baseline-dir /path/to/round_0 aten__add
#   ./test_cuda_optimized_skill.sh --baseline-dir /path/to/round_0 aten__add,aten__mul
#   ./test_cuda_optimized_skill.sh -b runs/normal_cc_xxx
#   ./test_cuda_optimized_skill.sh --skip-verify
#   ./test_cuda_optimized_skill.sh --skip-run

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Auto-detect default dataset based on device type
_DEVICE_TYPE=$(python -c "
import sys; sys.path.insert(0, '$(dirname "${BASH_SOURCE[0]}")')
from device_manager import detect_device_type
print(detect_device_type())
" 2>/dev/null || echo "cuda")

if [[ "$_DEVICE_TYPE" == "cuda" ]]; then
    DEFAULT_DATASET="KernelGenBench"
else
    DEFAULT_DATASET="KernelGenBench-aten"
fi

# Default values
DATASET="$DEFAULT_DATASET"
BASELINE_RUN=""
BASELINE_DIR=""
TIMEOUT=1800
VERIFY_TIMEOUT=600
DEVICE_COUNT=8
BUDGET=""
CLAUDE_BIN=""
MAX_RETRIES=""
MAX_ITERATIONS=""
SKIP_RUN=false
SKIP_VERIFY=false
SKIP_GEN=false
VERBOSE=""
RESUME_RUN=""

# Parse arguments
OPERATORS=""
while [[ $# -gt 0 ]]; do
    case $1 in
        -d|--dataset)
            DATASET="$2"
            shift 2
            ;;
        -b|--baseline-run)
            BASELINE_RUN="$2"
            shift 2
            ;;
        --baseline-dir)
            BASELINE_DIR="$2"
            shift 2
            ;;
        -t|--timeout)
            TIMEOUT="$2"
            shift 2
            ;;
        --verify-timeout)
            VERIFY_TIMEOUT="$2"
            shift 2
            ;;
        --device-count)
            DEVICE_COUNT="$2"
            shift 2
            ;;
        --budget)
            BUDGET="$2"
            shift 2
            ;;
        --claude-bin)
            CLAUDE_BIN="$2"
            shift 2
            ;;
        --max-retries)
            MAX_RETRIES="$2"
            shift 2
            ;;
        --max-iterations)
            MAX_ITERATIONS="$2"
            shift 2
            ;;
        --skip-run)
            SKIP_RUN=true
            shift
            ;;
        --skip-verify)
            SKIP_VERIFY=true
            shift
            ;;
        --skip-gen)
            SKIP_GEN=true
            shift
            ;;
        --resume)
            RESUME_RUN="$2"
            SKIP_GEN=true
            shift 2
            ;;
        -v|--verbose)
            VERBOSE="--verbose"
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS] [OPERATORS]"
            echo ""
            echo "One-click cuda-optimized-skill kernel optimization + verification"
            echo ""
            echo "Options:"
            echo "  -d, --dataset          Dataset (default: $DEFAULT_DATASET)"
            echo "  -b, --baseline-run     Baseline run directory with kernels/ subdir"
            echo "  --baseline-dir         Baseline directory with kernel .py files"
            echo "  -t, --timeout          Timeout per kernel in seconds (default: 1800)"
            echo "  --verify-timeout       Timeout for verification in seconds (default: 600)"
            echo "  --device-count         Number of GPUs to use (default: 8)"
            echo "  --budget               Budget limit per kernel in USD"
            echo "  --claude-bin           Path to claude binary"
            echo "  --max-retries          Max attempts per kernel (default: 1)"
            echo "  --max-iterations       Max optimization iterations per kernel (default: 5)"
            echo "  --skip-run             Skip running agent (only verify existing run)"
            echo "  --skip-verify          Skip verification step"
            echo "  --skip-gen             Skip prompt generation"
            echo "  --resume <run>         Resume a previous run"
            echo "  -v, --verbose          Verbose output"
            echo "  -h, --help             Show this help"
            echo ""
            echo "Examples:"
            echo "  $0 --baseline-dir /path/to/round_0"
            echo "  $0 --baseline-dir /path/to/round_0 aten__add,aten__mul"
            echo "  $0 -b runs/normal_cc_xxx"
            echo "  $0 --skip-run --resume cuda_optimized_xxx"
            exit 0
            ;;
        -*)
            echo "Unknown option: $1"
            exit 1
            ;;
        *)
            OPERATORS="$1"
            shift
            ;;
    esac
done

# Validate: need baseline-dir or baseline-run (unless resuming or skip-run)
if [[ "$SKIP_RUN" == "false" && -z "$RESUME_RUN" && -z "$BASELINE_DIR" && -z "$BASELINE_RUN" ]]; then
    echo "Error: --baseline-dir or --baseline-run is required"
    echo "Run $0 --help for usage"
    exit 1
fi

echo "=================================================="
echo "cuda-optimized-skill Benchmark"
echo "=================================================="
echo "Dataset:    $DATASET"
echo "Timeout:    ${TIMEOUT}s per kernel"
echo "Devices:    $DEVICE_COUNT"
if [[ -n "$BASELINE_DIR" ]]; then
    echo "Baseline:   $BASELINE_DIR"
elif [[ -n "$BASELINE_RUN" ]]; then
    echo "Baseline:   $BASELINE_RUN"
fi
if [[ -n "$OPERATORS" ]]; then
    echo "Operators:  $OPERATORS"
fi
echo "=================================================="
echo ""

START_TIME=$(date +%s)

# Step 1: Generate prompts (for verify to use)
if [[ "$SKIP_GEN" == "false" ]]; then
    echo "[Step 1/3] Generating prompts..."
    if [[ -n "$OPERATORS" ]]; then
        python generate_prompts.py --dataset "$DATASET" --op "$OPERATORS" --force
    else
        python generate_prompts.py --dataset "$DATASET" --force
    fi
else
    echo "[Step 1/3] Skipping prompt generation (--skip-gen)"
fi

# Step 2: Run cuda-optimized-skill
if [[ "$SKIP_RUN" == "false" ]]; then
    echo ""
    echo "[Step 2/3] Running cuda-optimized-skill optimization..."

    RUN_ARGS=(--dataset "$DATASET" -t "$TIMEOUT" --device-count "$DEVICE_COUNT")

    if [[ -n "$BASELINE_DIR" ]]; then
        RUN_ARGS+=(--baseline-dir "$BASELINE_DIR")
    elif [[ -n "$BASELINE_RUN" ]]; then
        RUN_ARGS+=(-b "$BASELINE_RUN")
    fi

    if [[ -n "$OPERATORS" ]]; then
        RUN_ARGS+=(-k "$OPERATORS")
    fi
    if [[ -n "$BUDGET" ]]; then
        RUN_ARGS+=(--budget "$BUDGET")
    fi
    if [[ -n "$CLAUDE_BIN" ]]; then
        RUN_ARGS+=(--claude-bin "$CLAUDE_BIN")
    fi
    if [[ -n "$MAX_RETRIES" ]]; then
        RUN_ARGS+=(--max-retries "$MAX_RETRIES")
    fi
    if [[ -n "$MAX_ITERATIONS" ]]; then
        RUN_ARGS+=(--max-iterations "$MAX_ITERATIONS")
    fi
    if [[ -n "$RESUME_RUN" ]]; then
        RUN_ARGS+=(--resume "$RESUME_RUN")
    fi
    if [[ -n "$VERBOSE" ]]; then
        RUN_ARGS+=($VERBOSE)
    fi

    python run_cuda_optimized_skill.py "${RUN_ARGS[@]}"
else
    echo ""
    echo "[Step 2/3] Skipping agent run (--skip-run)"
fi

# Find latest run directory
if [[ -n "$RESUME_RUN" ]]; then
    LATEST_RUN="runs/$RESUME_RUN"
else
    LATEST_RUN=$(ls -dt runs/cuda_optimized_* 2>/dev/null | head -1)
fi

if [[ -z "$LATEST_RUN" || ! -d "$LATEST_RUN" ]]; then
    echo "Error: No cuda_optimized run directory found"
    exit 1
fi
RUN_NAME=$(basename "$LATEST_RUN")
echo ""
echo "Run: $RUN_NAME"

# Step 3: Verify
if [[ "$SKIP_VERIFY" == "false" ]]; then
    echo ""
    echo "[Step 3/3] Verifying optimized kernels..."

    VERIFY_ARGS=(--run "$RUN_NAME" --device-count "$DEVICE_COUNT" --timeout "$VERIFY_TIMEOUT")
    if [[ -n "$OPERATORS" ]]; then
        VERIFY_ARGS+=(--op "$OPERATORS")
    fi
    if [[ -n "$VERBOSE" ]]; then
        VERIFY_ARGS+=($VERBOSE)
    fi

    python verify.py "${VERIFY_ARGS[@]}"
else
    echo ""
    echo "[Step 3/3] Skipping verification (--skip-verify)"
fi

# Elapsed time
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
HOURS=$(echo "scale=2; $ELAPSED / 3600" | bc)

echo ""
echo "=================================================="
echo "Done! Results in: runs/$RUN_NAME/"
echo "Elapsed: ${HOURS}h (${ELAPSED}s)"
echo "=================================================="
