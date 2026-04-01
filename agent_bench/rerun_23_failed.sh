#!/bin/bash
# Rerun 23 failed vllm13 operators (cutlass_pack_scale_fp8 already done)
# 4 GPU parallel

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNDIR="/share/project/zpy/flagbench/agent_bench/runs/normal_opencode_KernelGenBench_20260323_144238"
KERNELS_DIR="$RUNDIR/kernels"
WORKSPACES_DIR="$RUNDIR/workspaces"

OPS=(
  vllm13__allspark_w8a16_gemm
  vllm13__awq_marlin_moe_repack
  vllm13__batched_moe_align_block_size
  vllm13__convert_vertical_slash_indexes
  vllm13__copy_blocks_mla
  vllm13__cp_gather_cache
  vllm13__cp_gather_indexer_k_quant_cache
  vllm13__fused_qk_norm_rope
  vllm13__gather_and_maybe_dequant_cache
  vllm13__ggml_dequantize
  vllm13__ggml_moe_a8
  vllm13__gptq_marlin_24_gemm
  vllm13__gptq_marlin_gemm
  vllm13__gptq_marlin_moe_repack
  vllm13__gptq_shuffle
  vllm13__hadacore_transform
  vllm13__marlin_int4_fp8_preprocess
  vllm13__merge_attn_states
  vllm13__moe_lora_align_block_size
  vllm13__permute_cols
  vllm13__rms_norm_dynamic_per_token_quant
  vllm13__rms_norm_per_block_quant
  vllm13__scaled_int8_quant
  vllm13__selective_scan_fwd
)

# Minus cutlass_pack_scale_fp8 = 24 ops, but allspark_w8a16_gemm was also listed as failed
# Keep all 24 minus cutlass_pack_scale_fp8 = 23 + allspark = 24 total, let's just run all listed

GPU_IDS=(0 1 2 3)
NUM_GPUS=${#GPU_IDS[@]}
PIDS=()
GPU_ASSIGN=()

echo "============================================"
echo "Rerunning ${#OPS[@]} failed vllm13 operators"
echo "GPUs: ${GPU_IDS[*]}"
echo "============================================"

run_op() {
    local op=$1
    local gpu=$2
    local ws="$WORKSPACES_DIR/$op"
    local prompt="$ws/prompt.md"
    local output="$ws/oc_output_rerun.json"
    local log="$ws/oc_rerun.log"

    if [ ! -f "$prompt" ]; then
        echo "[GPU $gpu] SKIP $op - no prompt.md"
        return 1
    fi

    echo "[GPU $gpu] START $op"
    IS_SANDBOX=1 CUDA_VISIBLE_DEVICES=$gpu \
        opencode run "$(cat "$prompt")" \
        --format json \
        --model zyapi/mco-4 \
        --dir "$ws" \
        > "$output" 2> "$log"

    # Extract kernel code
    python3 -c "
import re, sys
with open('$output') as f:
    content = f.read()
matches = re.findall(r'\x60\x60\x60python\s*(.*?)\s*\x60\x60\x60', content, re.DOTALL)
if matches:
    with open('$ws/kernel.py', 'w') as f:
        f.write(matches[-1].strip())
    with open('$KERNELS_DIR/${op}.py', 'w') as f:
        f.write(matches[-1].strip())
    print('[GPU $gpu] DONE $op - kernel extracted')
else:
    print('[GPU $gpu] FAIL $op - no code block found')
"
}

# Launch ops round-robin across GPUs
idx=0
for op in "${OPS[@]}"; do
    gpu=${GPU_IDS[$((idx % NUM_GPUS))]}
    run_op "$op" "$gpu" &
    PIDS+=($!)
    GPU_ASSIGN+=("$op -> GPU $gpu (PID $!)")
    idx=$((idx + 1))

    # If all GPUs busy, wait for one to finish
    if [ ${#PIDS[@]} -ge $NUM_GPUS ]; then
        wait -n 2>/dev/null || wait "${PIDS[0]}"
        # Clean up finished PIDs
        NEW_PIDS=()
        for pid in "${PIDS[@]}"; do
            if kill -0 "$pid" 2>/dev/null; then
                NEW_PIDS+=("$pid")
            fi
        done
        PIDS=("${NEW_PIDS[@]}")
    fi
done

# Wait for all remaining
echo ""
echo "Waiting for remaining jobs..."
for pid in "${PIDS[@]}"; do
    wait "$pid" 2>/dev/null || true
done

echo ""
echo "============================================"
echo "All done. Checking results..."
echo "============================================"

success=0
fail=0
for op in "${OPS[@]}"; do
    if [ -f "$KERNELS_DIR/${op}.py" ] && [ $(stat -c%s "$KERNELS_DIR/${op}.py" 2>/dev/null || echo 0) -gt 50 ]; then
        echo "OK: $op"
        success=$((success + 1))
    else
        echo "FAIL: $op"
        fail=$((fail + 1))
    fi
done
echo ""
echo "Success: $success / ${#OPS[@]}, Failed: $fail"
