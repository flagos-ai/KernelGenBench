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

# One-click script to test operators with agent benchmark
#
# Usage:
#   ./test_ops.sh -d KernelGenBench    # Test entire dataset
#   ./test_ops.sh add                     # Test single operator
#   ./test_ops.sh add,softmax             # Test multiple operators
#   ./test_ops.sh add --dataset KernelGenBench  # Specify dataset
#   ./test_ops.sh --skip-gen              # Skip prompt generation
#   ./test_ops.sh --skip-verify           # Skip verification
#   ./test_ops.sh --device-count 4        # Use GPUs 0-3 for generation and verification
#   ./test_ops.sh --gpu-ids 1,3,5,7       # Use exact GPUs for both stages

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

CONFIG="${CONFIG:-config.yaml}"
_CFG_PYTHON=$(python3 -c "
import yaml, sys
try:
    cfg = yaml.safe_load(open('$CONFIG'))
    print(cfg.get('paths', {}).get('python', ''))
except: print('')
" 2>/dev/null)
PYTHON="${PYTHON:-${_CFG_PYTHON:-$(which python3)}}"

# Auto-detect device type and select default dataset
# NVIDIA: KernelGenBench (210 ops = 110 aten + 50 vllm + 50 cublas)
# Other chips: KernelGenBench-aten (110 aten ops)
_DEVICE_TYPE=$($PYTHON -c "
import sys; sys.path.insert(0, '$SCRIPT_DIR')
from device_manager import detect_device_type
print(detect_device_type())
" 2>/dev/null || echo "cuda")

if [[ "$_DEVICE_TYPE" == "cuda" ]]; then
    DEFAULT_DATASET="KernelGenBench"
else
    DEFAULT_DATASET="KernelGenBench-aten"
    export TORCH_DEVICE_BACKEND_AUTOLOAD=0
fi

# Default values
DATASET="$DEFAULT_DATASET"
METHOD="normal_cc"
DEVICE_COUNT=8
GPU_IDS=""
DEVICE_COUNT_SET=false
TIMEOUT=600
SKIP_GEN=false
SKIP_VERIFY=false
VERBOSE=""
MAX_OPTIMIZE_CALLS=""
TARGET_SPEEDUP=""

# Parse arguments
OPERATORS=""
while [[ $# -gt 0 ]]; do
    case $1 in
        -d|--dataset)
            DATASET="$2"
            shift 2
            ;;
        -m|--method)
            METHOD="$2"
            shift 2
            ;;
        --device-count)
            if [[ -n "$GPU_IDS" ]]; then
                echo "Error: --device-count and --gpu-ids are mutually exclusive"
                exit 1
            fi
            DEVICE_COUNT="$2"
            DEVICE_COUNT_SET=true
            shift 2
            ;;
        --gpu-ids)
            if [[ "$DEVICE_COUNT_SET" == "true" ]]; then
                echo "Error: --device-count and --gpu-ids are mutually exclusive"
                exit 1
            fi
            GPU_IDS="$2"
            shift 2
            ;;
        --timeout)
            TIMEOUT="$2"
            shift 2
            ;;
        --max-optimize-calls)
            MAX_OPTIMIZE_CALLS="$2"
            shift 2
            ;;
        --target-speedup)
            TARGET_SPEEDUP="$2"
            shift 2
            ;;
        --skip-gen)
            SKIP_GEN=true
            shift
            ;;
        --skip-verify)
            SKIP_VERIFY=true
            shift
            ;;
        -v|--verbose)
            VERBOSE="-v"
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [operators] [options]"
            echo ""
            echo "Arguments:"
            echo "  operators           Comma-separated operator names (optional)"
            echo "                      If not specified, test entire dataset"
            echo ""
            echo "Options:"
            echo "  -d, --dataset       Dataset to use (default: KernelGenBench)"
            echo "  -m, --method        Agent method to use (default: naive_cc)"
            echo "                      Available: naive_cc, normal_cc, iterative_optimizer"
            echo "  --device-count      Use the first N GPUs for generation and verification (default: 8)"
            echo "  --gpu-ids           Use exact GPU IDs for both stages (for example: 1,3,5,7)"
            echo "  --timeout           Timeout per operator in seconds (default: 600)"
            echo "  --skip-gen          Skip prompt generation step"
            echo "  --skip-verify       Skip verification step (only generate)"
            echo "  -v, --verbose       Enable verbose output"
            echo "  -h, --help          Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0                               # Test entire KernelGenBench dataset"
            echo "  $0 add                           # Test add operator"
            echo "  $0 add,softmax                   # Test multiple operators"
            echo "  $0 --skip-gen                    # Skip regenerating prompts"
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

# Build --op argument if operators specified

if [[ -n "$GPU_IDS" ]]; then
    DEVICE_ARGS=(--gpu-ids "$GPU_IDS")
else
    DEVICE_ARGS=(--device-count "$DEVICE_COUNT")
fi
if [[ -n "$OPERATORS" ]]; then
    OP_ARG="--op $OPERATORS"
    DISPLAY_TARGET="$OPERATORS"
else
    OP_ARG=""
    DISPLAY_TARGET="all operators"
fi

echo "=================================================="
echo "Agent Benchmark"
echo "Device type: $_DEVICE_TYPE"
echo "Dataset: $DATASET"
echo "Method: $METHOD"
echo "Target: $DISPLAY_TARGET"
echo "=================================================="

# Step 1: Generate prompts
if [[ -n "$GPU_IDS" ]]; then
    echo "GPUs: $GPU_IDS"
else
    echo "GPUs: first $DEVICE_COUNT"
fi
if [[ "$SKIP_GEN" == "false" ]]; then
    echo ""
    echo "[Step 1/3] Generating prompts..."
    if [[ -n "$OPERATORS" ]]; then
        $PYTHON generate_prompts.py --dataset "$DATASET" --op "$OPERATORS" --force
    else
        $PYTHON generate_prompts.py --dataset "$DATASET" --force
    fi
else
    echo ""
    echo "[Step 1/3] Skipping prompt generation (--skip-gen)"
fi

# Step 2: Run agent
echo ""
echo "[Step 2/3] Running agent to generate kernels..."

# Build run.py command as an array (safer than eval)
RUN_ARGS=(--dataset "$DATASET" --method "$METHOD" "${DEVICE_ARGS[@]}")
if [[ -n "$OPERATORS" ]]; then
    RUN_ARGS+=(--op "$OPERATORS")
fi
if [[ -n "$VERBOSE" ]]; then
    RUN_ARGS+=($VERBOSE)
fi
if [[ -n "$MAX_OPTIMIZE_CALLS" ]]; then
    RUN_ARGS+=(--max-optimize-calls "$MAX_OPTIMIZE_CALLS")
fi
if [[ -n "$TARGET_SPEEDUP" ]]; then
    RUN_ARGS+=(--target-speedup "$TARGET_SPEEDUP")
fi

$PYTHON run.py "${RUN_ARGS[@]}"

# Get the run directory from .last_run marker
if [[ -f runs/.last_run ]]; then
    RUN_NAME=$(cat runs/.last_run)
else
    echo "Error: No run directory found (runs/.last_run missing)"
    exit 1
fi

echo ""
echo "Run completed: $RUN_NAME"

# Step 3: Verify
if [[ "$SKIP_VERIFY" == "false" ]]; then
    echo ""
    echo "[Step 3/3] Verifying generated kernels..."
    VERIFY_ARGS=(--run "$RUN_NAME" "${DEVICE_ARGS[@]}" --timeout "$TIMEOUT")
    if [[ -n "$OPERATORS" ]]; then
        VERIFY_ARGS+=(--op "$OPERATORS")
    fi
    if [[ -n "$VERBOSE" ]]; then
        VERIFY_ARGS+=("$VERBOSE")
    fi

    $PYTHON verify.py "${VERIFY_ARGS[@]}"
else
    echo ""
    echo "[Step 3/3] Skipping verification (--skip-verify)"
fi

echo ""
echo "=================================================="
echo "Done! Results in: runs/$RUN_NAME/"
echo "=================================================="
