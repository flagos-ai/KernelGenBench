#!/bin/bash
# One-click script to test operators with agent benchmark
#
# Usage:
#   ./test_ops.sh -d v2_1                 # Test entire v2_1 dataset
#   ./test_ops.sh add                     # Test single operator
#   ./test_ops.sh add,softmax             # Test multiple operators
#   ./test_ops.sh add --dataset v2        # Specify dataset
#   ./test_ops.sh --skip-gen              # Skip prompt generation
#   ./test_ops.sh --skip-verify           # Skip verification
#   ./test_ops.sh --device-count 4        # Use 4 GPUs for verification

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Default values
DATASET="v2_1"
DEVICE_COUNT=8
TIMEOUT=300
SKIP_GEN=false
SKIP_VERIFY=false
VERBOSE=""

# Parse arguments
OPERATORS=""
while [[ $# -gt 0 ]]; do
    case $1 in
        -d|--dataset)
            DATASET="$2"
            shift 2
            ;;
        --device-count)
            DEVICE_COUNT="$2"
            shift 2
            ;;
        --timeout)
            TIMEOUT="$2"
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
            echo "  -d, --dataset       Dataset to use (default: v2_1)"
            echo "  --device-count      Number of GPUs for verification (default: 8)"
            echo "  --timeout           Timeout per operator in seconds (default: 300)"
            echo "  --skip-gen          Skip prompt generation step"
            echo "  --skip-verify       Skip verification step (only generate)"
            echo "  -v, --verbose       Enable verbose output"
            echo "  -h, --help          Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0 -d v2_1                       # Test entire v2_1 dataset"
            echo "  $0 -d v2                         # Test entire v2 dataset"
            echo "  $0 add                           # Test add operator"
            echo "  $0 add,softmax -d v2_1           # Test multiple operators"
            echo "  $0 --skip-gen -d v2_1            # Skip regenerating prompts"
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
if [[ -n "$OPERATORS" ]]; then
    OP_ARG="--op $OPERATORS"
    DISPLAY_TARGET="$OPERATORS"
else
    OP_ARG=""
    DISPLAY_TARGET="all operators"
fi

echo "=================================================="
echo "Agent Benchmark"
echo "Dataset: $DATASET"
echo "Target: $DISPLAY_TARGET"
echo "=================================================="

# Step 1: Generate prompts
if [[ "$SKIP_GEN" == "false" ]]; then
    echo ""
    echo "[Step 1/3] Generating prompts..."
    if [[ -n "$OPERATORS" ]]; then
        python generate_prompts.py --dataset "$DATASET" --op "$OPERATORS" --force
    else
        python generate_prompts.py --dataset "$DATASET" --force
    fi
else
    echo ""
    echo "[Step 1/3] Skipping prompt generation (--skip-gen)"
fi

# Step 2: Run agent
echo ""
echo "[Step 2/3] Running agent to generate kernels..."
python run.py --dataset "$DATASET" $OP_ARG $VERBOSE

# Get the latest run directory
LATEST_RUN=$(ls -td runs/*_${DATASET}_* 2>/dev/null | head -1)
if [[ -z "$LATEST_RUN" ]]; then
    echo "Error: No run directory found"
    exit 1
fi
RUN_NAME=$(basename "$LATEST_RUN")

echo ""
echo "Run completed: $RUN_NAME"

# Step 3: Verify
if [[ "$SKIP_VERIFY" == "false" ]]; then
    echo ""
    echo "[Step 3/3] Verifying generated kernels..."
    python verify.py --run "$RUN_NAME" $OP_ARG --device-count "$DEVICE_COUNT" --timeout "$TIMEOUT" $VERBOSE
else
    echo ""
    echo "[Step 3/3] Skipping verification (--skip-verify)"
fi

echo ""
echo "=================================================="
echo "Done! Results in: runs/$RUN_NAME/"
echo "=================================================="
