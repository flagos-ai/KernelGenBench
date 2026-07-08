"""
Tests for the competition anti-cheat package.

Tests cover:
- Layer 1: Cache Isolation
- Layer 3: Import Hook Sandbox
- Layer 4: CUDA Protection
- Layer 5: Random Shape Generator
- Layer 7: Timing Validation
- Integration: Full pipeline with cheat kernels

Run with:
    pytest src/sandbox/competition/tests/ -v

Run integration tests (slower, requires GPU):
    RUN_INTEGRATION=1 pytest src/sandbox/competition/tests/ -v
"""