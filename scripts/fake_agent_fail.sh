#!/bin/bash
# Failing fake agent: writes broken solution to test retry logic
TASK_DIR="$1"
cat > "$TASK_DIR/solution.py" << 'EOF'
import torch

def fused_add_rms_norm(input, residual, weight, epsilon):
    # Intentionally wrong: just zeros
    input.zero_()
    residual.add_(input)
EOF
echo "Wrote broken solution"
