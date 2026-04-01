#!/bin/bash
# Rerun 25 failed vllm13 operators across 4 GPUs
set -e

cd "$(dirname "$0")"

export ANTHROPIC_API_KEY=sk-rCqrW8R3UqczpamNUk1W3HyxIi6Yx7YQMF7qGdF5SQttfSKi
export ANTHROPIC_BASE_URL=https://zyapi.xmsxb.com/
export ANTHROPIC_AUTH_TOKEN=sk-rCqrW8R3UqczpamNUk1W3HyxIi6Yx7YQMF7qGdF5SQttfSKi
export ANTHROPIC_MODEL=mco-4
export CLAUDE_CODE_DISABLE_ADAPTIVE_THINKING=1

ORIG_RUNDIR=runs/normal_opencode_KernelGenBench_20260323_144238
RUNDIR=runs/rerun_failed_24
KERNELS_DIR=$RUNDIR/kernels
WORKSPACES_DIR=$RUNDIR/workspaces
PROMPTS_DIR=prompts/KernelGenBench
OPENCODE_BIN=/share/project/zhaohuxing/anaconda3/envs/claude_tool/bin/opencode
OPENCODE_MODEL=zyapi/mco-4

# 24 failed operators (allspark_w8a16_gemm already done)
FAILED_OPS=(
    vllm13__awq_marlin_moe_repack
    vllm13__batched_moe_align_block_size
    vllm13__convert_vertical_slash_indexes
    vllm13__copy_blocks_mla
    vllm13__cp_gather_cache
    vllm13__cp_gather_indexer_k_quant_cache
    vllm13__cutlass_pack_scale_fp8
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

run_op() {
    local op=$1
    local gpu=$2
    local prompt_file="$PROMPTS_DIR/${op}.md"
    local workspace="$WORKSPACES_DIR/${op}"

    if [[ ! -f "$prompt_file" ]]; then
        echo "[GPU $gpu] SKIP $op - no prompt file"
        return
    fi

    echo "[GPU $gpu] START $op"
    mkdir -p "$workspace"

    # Read prompt and build enhanced prompt (reuse existing prompt.md if available)
    local prompt_md="$workspace/prompt.md"
    if [[ ! -f "$prompt_md" ]]; then
        cp "$prompt_file" "$prompt_md"
    fi
    local prompt=$(cat "$prompt_md")

    local output_file="$workspace/oc_output.json"
    local log_file="$workspace/oc.log"

    # Run opencode
    CUDA_VISIBLE_DEVICES=$gpu IS_SANDBOX=1 \
        $OPENCODE_BIN run "$prompt" \
        --format json \
        --dir "$workspace" \
        --model "$OPENCODE_MODEL" \
        > "$output_file" 2> "$log_file" || true

    # Extract kernel.py and copy to kernels dir
    if [[ -f "$workspace/kernel.py" ]]; then
        cp "$workspace/kernel.py" "$KERNELS_DIR/${op}.py"
        echo "[GPU $gpu] SUCCESS $op"
    else
        # Try extract from output
        echo "[GPU $gpu] WARN $op - no kernel.py, check output manually"
    fi
}

# Distribute 25 ops across 4 GPUs
run_gpu_batch() {
    local gpu=$1
    shift
    local ops=("$@")
    for op in "${ops[@]}"; do
        run_op "$op" "$gpu"
    done
}

# Split: GPU0=6, GPU1=6, GPU2=6, GPU3=6
GPU0_OPS=("${FAILED_OPS[@]:0:6}")
GPU1_OPS=("${FAILED_OPS[@]:6:6}")
GPU2_OPS=("${FAILED_OPS[@]:12:6}")
GPU3_OPS=("${FAILED_OPS[@]:18:6}")

echo "=== Rerunning 25 failed vllm13 operators ==="
echo "API: kspmas / ep-20260313004351-feb36"
echo "GPU0: ${#GPU0_OPS[@]} ops, GPU1: ${#GPU1_OPS[@]} ops, GPU2: ${#GPU2_OPS[@]} ops, GPU3: ${#GPU3_OPS[@]} ops"
echo ""

# Launch 4 GPU batches in parallel
run_gpu_batch 0 "${GPU0_OPS[@]}" &
PID0=$!
run_gpu_batch 1 "${GPU1_OPS[@]}" &
PID1=$!
run_gpu_batch 2 "${GPU2_OPS[@]}" &
PID2=$!
run_gpu_batch 3 "${GPU3_OPS[@]}" &
PID3=$!

# Wait for all
wait $PID0 $PID1 $PID2 $PID3

echo ""
echo "=== Generation done ==="
echo "Kernels in $KERNELS_DIR:"
ls "$KERNELS_DIR"/vllm13__*.py 2>/dev/null | wc -l
echo "of 50 vllm13 total"
