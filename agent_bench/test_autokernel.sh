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

# One-click script for AutoKernel kernel generation/optimization + verification
#
# Usage:
#   ./test_autokernel.sh                              # Generate all operators
#   ./test_autokernel.sh aten__add                     # Generate single operator
#   ./test_autokernel.sh aten__add,aten__mul           # Generate multiple operators
#   ./test_autokernel.sh --mode optimize -b <run>      # Optimize from baseline run
#   ./test_autokernel.sh --baseline-dir /path/to/round_0  # Optimize from kernel dir
#   ./test_autokernel.sh --baseline-dir /path/to/round_0 aten__add,aten__mul  # Specific ops
#   ./test_autokernel.sh --skip-verify                 # Skip verification step
#   ./test_autokernel.sh --skip-run                    # Only verify existing run

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
BASELINE_DIR=""
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
        --baseline-dir)
            BASELINE_DIR="$2"
            MODE="optimize"
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
            echo "  operators              Comma-separated operator names (optional)"
            echo ""
            echo "Options:"
            echo "  -d, --dataset          Dataset (default: KernelGenBench on NVIDIA, KernelGenBench-aten on other chips)"
            echo "  --mode                 Mode: generate or optimize (default: generate)"
            echo "  -b, --baseline-run     Baseline run for optimize mode (uses kernels/ subdir)"
            echo "  --baseline-dir         Directory of baseline kernel .py files (for optimize mode)"
            echo "  -t, --timeout          Timeout per kernel in seconds (default: 1800)"
            echo "  --verify-timeout       Timeout for verification in seconds (default: 600)"
            echo "  --device-count         Number of GPUs for verification (default: 8)"
            echo "  --budget               Budget limit per kernel in USD"
            echo "  --claude-bin           Path to claude binary"
            echo "  --max-retries          Max attempts per kernel (default: 1)"
            echo "  --skip-run             Skip running AutoKernel (only verify)"
            echo "  --skip-verify          Skip verification step"
            echo "  --skip-gen             Skip prompt generation"
            echo "  --resume               Resume a previous run by name"
            echo "  -v, --verbose          Enable verbose output"
            echo "  -h, --help             Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0                                         # Generate all KernelGenBench operators"
            echo "  $0 aten__add                               # Generate single operator"
            echo "  $0 aten__add,aten__mul                     # Multiple operators"
            echo "  $0 --mode optimize -b runs/normal_cc_xxx   # Optimize from baseline run"
            echo "  $0 --baseline-dir /path/to/round_0          # Optimize from kernel directory"
            echo "  $0 --baseline-dir /path/to/round_0 aten__add,aten__mul  # Optimize specific ops"
            echo "  $0 --skip-run -d KernelGenBench            # Only verify existing run"
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

echo "=================================================="
echo "AutoKernel Benchmark"
echo "Device type: $_DEVICE_TYPE"
echo "Dataset: $DATASET"
echo "Mode: $MODE"
if [[ -n "$OPERATORS" ]]; then
    echo "Target: $OPERATORS"
else
    echo "Target: all operators"
fi
echo "Timeout: ${TIMEOUT}s"
echo "=================================================="

START_TIME=$(date +%s)

# Step 1: Generate prompts (if needed)
if [[ "$SKIP_GEN" == "false" && "$SKIP_RUN" == "false" ]]; then
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

# Step 2: Run AutoKernel
if [[ "$SKIP_RUN" == "false" ]]; then
    echo ""
    echo "[Step 2/3] Running AutoKernel..."

    RUN_ARGS=(--mode "$MODE" --dataset "$DATASET" -t "$TIMEOUT")

    if [[ -n "$OPERATORS" ]]; then
        RUN_ARGS+=(-k "$OPERATORS")
    fi
    if [[ -n "$BASELINE_RUN" ]]; then
        RUN_ARGS+=(--baseline-run "$BASELINE_RUN")
    fi
    if [[ -n "$BASELINE_DIR" ]]; then
        RUN_ARGS+=(--baseline-dir "$BASELINE_DIR")
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
    if [[ -n "$RESUME_RUN" ]]; then
        RUN_ARGS+=(--resume "$RESUME_RUN")
    fi
    RUN_ARGS+=(--device-count "$DEVICE_COUNT")

    python run_autokernel.py "${RUN_ARGS[@]}"
else
    echo ""
    echo "[Step 2/3] Skipping AutoKernel run (--skip-run)"
fi

# Find latest run directory
if [[ -n "$RESUME_RUN" ]]; then
    LATEST_RUN="runs/$RESUME_RUN"
else
    LATEST_RUN=$(ls -td runs/autokernel_* 2>/dev/null | head -1)
fi

if [[ -z "$LATEST_RUN" ]]; then
    echo "Error: No AutoKernel run directory found"
    exit 1
fi
RUN_NAME=$(basename "$LATEST_RUN")
echo ""
echo "Run: $RUN_NAME"

# Step 3: Verify
if [[ "$SKIP_VERIFY" == "false" ]]; then
    echo ""
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

# Elapsed time
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
HOURS=$(echo "scale=2; $ELAPSED / 3600" | bc)

echo ""
echo "=================================================="
echo "Done! Results in: runs/$RUN_NAME/"
echo "Elapsed: ${HOURS}h (${ELAPSED}s)"
echo "=================================================="
