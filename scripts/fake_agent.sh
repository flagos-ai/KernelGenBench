#!/bin/bash
# Fake agent: copies baseline.py as solution.py for testing
TASK_DIR="$1"
if [ -f "$TASK_DIR/baseline.py" ]; then
    cp "$TASK_DIR/baseline.py" "$TASK_DIR/solution.py"
    echo "Copied baseline as solution"
fi
