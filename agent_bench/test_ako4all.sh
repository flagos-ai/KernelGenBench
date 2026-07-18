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

# One-click script for AKO4ALL kernel generation/optimization + verification
#
# Usage:
#   ./test_ako4all.sh                              # Generate all operators
#   ./test_ako4all.sh aten___add                   # Generate single operator
#   ./test_ako4all.sh aten___add,aten___mul         # Generate multiple operators
#   ./test_ako4all.sh --mode optimize -b <run>     # Optimize from baseline
#   ./test_ako4all.sh --skip-verify                # Skip verification step
#   ./test_ako4all.sh --skip-run                   # Only verify existing run

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
MODE="generate"
BASELINE_RUN=""
ITERATIONS=5
TIMEOUT=1800
VERIFY_TIMEOUT=600
DEVICE_COUNT=8
BUDGET=""
CLAUDE_BIN=""
MAX_RETRIES=""
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
        --mode)
            MODE="$2"
            shift 2
            ;;
        -b|--baseline-run)
            BASELINE_RUN="$2"
            MODE="optimize"
            shift 2
            ;;
        -i|--iterations)
            ITERATIONS="$2"
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
            SKIP_RUN=true
            shift 2
            ;;
        -v|--verbose)
            VERBOSE="-v"
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [operators] [options]"
            echo ""
            echo "Arguments:"
            echo "  operators              Comma-separated kernel names (optional)"
            echo ""
            echo "Options:"
            echo "  -d, --dataset          Dataset (default: $DEFAULT_DATASET)"
            echo "  --mode                 generate or optimize (default: generate)"
            echo "  -b, --baseline-run     Baseline run dir (implies --mode optimize)"
            echo "  -i, --iterations       Optimization iterations per kernel (default: 5)"
            echo "  -t, --timeout          Timeout per kernel in seconds (default: 1800)"
            echo "  --verify-timeout       Verification timeout per op (default: 600)"
            echo "  --device-count         GPUs for verification (default: 8)"
            echo "  --budget               Budget limit per kernel in USD"
            echo "  --claude-bin           Path to claude binary"
            echo "  --max-retries          Max attempts per kernel (default: 1 = no retry)"
            echo "  --skip-run             Skip run, only verify latest run"
            echo "  --skip-verify          Skip verification step"
            echo "  --skip-gen             Skip prompt generation"
            echo "  --resume RUN_NAME      Verify a specific existing run"
            echo "  -v, --verbose          Verbose output"
            echo ""
            echo "Examples:"
            echo "  $0                                         # Generate all ops"
            echo "  $0 aten___add                              # Generate single op"
            echo "  $0 -b normal_cc_KernelGenBench_20260325    # Optimize from baseline"
            echo "  $0 --skip-run --resume ako4all_generate_KernelGenBench_20260417_170104"
            exit 0
            ;;
        -*)
            echo "Unknown option: $1"
            exit 1
            ;;
        *)
            if [[ -z "$OPERATORS" ]]; then
                OPERATORS="$1"
            else
                echo "Error: Multiple positional arguments not supported"
                exit 1
            fi
            shift
            ;;
    esac
done

if [[ -n "$OPERATORS" ]]; then
    DISPLAY_TARGET="$OPERATORS"
else
    DISPLAY_TARGET="all operators"
fi

echo "=================================================="
echo "AKO4ALL Benchmark"
echo "Device type: $_DEVICE_TYPE"
echo "Dataset: $DATASET"
echo "Mode: $MODE"
echo "Iterations: $ITERATIONS"
echo "Target: $DISPLAY_TARGET"
echo "=================================================="

# Step 1: Generate prompts (for generate mode)
if [[ "$MODE" == "generate" && "$SKIP_GEN" == "false" && "$SKIP_RUN" == "false" ]]; then
    echo ""
    echo "[Step 1/3] Generating prompts..."
    if [[ -n "$OPERATORS" ]]; then
        python generate_prompts.py --dataset "$DATASET" --op "$OPERATORS" --force
    else
        python generate_prompts.py --dataset "$DATASET" --force
    fi
else
    echo ""
    echo "[Step 1/3] Skipping prompt generation"
fi

# Step 2: Run AKO4ALL
if [[ "$SKIP_RUN" == "false" ]]; then
    echo ""
    echo "[Step 2/3] Running AKO4ALL ($MODE mode)..."

    RUN_ARGS=(--mode "$MODE" --dataset "$DATASET" --iterations "$ITERATIONS" --timeout "$TIMEOUT")
    if [[ -n "$OPERATORS" ]]; then
        RUN_ARGS+=(--kernels "$OPERATORS")
    fi
    if [[ -n "$BASELINE_RUN" ]]; then
        RUN_ARGS+=(--baseline-run "$BASELINE_RUN")
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

    python run_ako4all.py "${RUN_ARGS[@]}"
fi

# Find run directory
if [[ -n "$RESUME_RUN" ]]; then
    LATEST_RUN="runs/$RESUME_RUN"
    if [[ ! -d "$LATEST_RUN" ]]; then
        echo "Error: Run directory not found: $LATEST_RUN"
        exit 1
    fi
else
    LATEST_RUN=$(ls -td runs/ako4all_*${DATASET}* 2>/dev/null | head -1)
fi

if [[ -z "$LATEST_RUN" ]]; then
    echo "Error: No AKO4ALL run directory found"
    exit 1
fi
RUN_NAME=$(basename "$LATEST_RUN")

echo ""
echo "Run directory: $RUN_NAME"

# Step 3: Verify
if [[ "$SKIP_VERIFY" == "false" ]]; then
    echo ""
    echo "[Step 3/3] Waiting for kernel files to sync..."
    sleep 10
    echo "[Step 3/3] Verifying generated kernels..."

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

# Step 4: Analyze token usage
echo ""
echo "[Step 4] Analyzing token usage..."
python ../scripts/analyze/analyze_tokens.py "runs/$RUN_NAME" || echo "(token analysis skipped)"

echo ""
echo "=================================================="
echo "Done! Results in: runs/$RUN_NAME/"
echo "=================================================="
