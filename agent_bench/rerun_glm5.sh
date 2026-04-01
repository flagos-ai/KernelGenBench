#!/bin/bash
# Rerun 104 missing operators for GLM5 (ep-20260317031105-oe8s2)
# 8 GPUs, sequential per GPU to avoid rate limit

set +e
source /share/project/zhaohuxing/anaconda3/bin/activate claude_tool

RUN_DIR="/share/project/zpy/flagbench/agent_bench/runs/normal_opencode_KernelGenBench_20260323_164848"
WORKSPACES="$RUN_DIR/workspaces"
KERNELS="$RUN_DIR/kernels"
OPENCODE_BIN="/share/project/zhaohuxing/anaconda3/envs/claude_tool/bin/opencode"
OPENCODE_MODEL="kspmas/ep-20260317031105-oe8s2"
TIMEOUT=1800
LOG_DIR="$RUN_DIR/rerun_logs"
mkdir -p "$LOG_DIR"

# 102 ops that failed due to framework (opencode path broken), NOT model failures
OPS=(
  aten__amin
  aten__bernoulli
  aten__copy_
  aten__cumsum
  aten__gather
  aten__mean
  aten__mish
  aten__mm
  aten__mul
  aten__pairwise_distance
  aten__poisson
  aten__polygamma
  aten__pow
  aten__resolve_neg
  aten__rot90
  aten__rrelu_with_noise
  aten__scatter
  cublas__cublasCgemmStridedBatched
  cublas__cublasCgemm_v2
  cublas__cublasCsyrkEx
  cublas__cublasDgemvBatched
  cublas__cublasDgemvStridedBatched
  cublas__cublasDgemv_v2
  cublas__cublasDsbmv_v2
  cublas__cublasDsyr2_v2
  cublas__cublasDtrsmBatched
  cublas__cublasHgemmBatched
  cublas__cublasHgemmStridedBatched
  cublas__cublasSaxpy_v2
  cublas__cublasSdgmm
  cublas__cublasSdot_v2
  cublas__cublasSgeam
  cublas__cublasSgemmBatched_64
  cublas__cublasSgemmEx
  cublas__cublasSgemmStridedBatched
  cublas__cublasSgemm_v2
  cublas__cublasSgemvBatched
  cublas__cublasSgemvStridedBatched
  cublas__cublasSger_v2
  cublas__cublasSscal_v2
  cublas__cublasSsyrk_v2
  cublas__cublasStbmv_v2
  cublas__cublasStrsm_v2
  cublas__cublasStrsv_v2
  cublas__cublasZdotc_v2
  cublas__cublasZgemmBatched
  cublas__cublasZgemmStridedBatched
  cublas__cublasZgemvBatched
  cublas__cublasZgemvStridedBatched
  cublas__cublasZgerc_v2
  cublas__cublasZswap_v2
  cublas__cublasZtrsmBatched
  vllm13__allspark_w8a16_gemm
  vllm13__apply_repetition_penalties_cuda
  vllm13__awq_gemm
  vllm13__awq_marlin_moe_repack
  vllm13__batched_moe_align_block_size
  vllm13__concat_and_cache_mla
  vllm13__convert_fp8
  vllm13__convert_vertical_slash_indexes
  vllm13__copy_blocks
  vllm13__copy_blocks_mla
  vllm13__cp_gather_cache
  vllm13__cp_gather_indexer_k_quant_cache
  vllm13__cutlass_pack_scale_fp8
  vllm13__cutlass_scaled_mm
  vllm13__cutlass_scaled_mm_azp
  vllm13__fused_add_rms_norm
  vllm13__fused_qk_norm_rope
  vllm13__gather_and_maybe_dequant_cache
  vllm13__ggml_dequantize
  vllm13__ggml_moe_a8
  vllm13__ggml_moe_a8_vec
  vllm13__ggml_mul_mat_a8
  vllm13__ggml_mul_mat_vec_a8
  vllm13__gptq_gemm
  vllm13__gptq_marlin_24_gemm
  vllm13__gptq_marlin_gemm
  vllm13__gptq_marlin_moe_repack
  vllm13__gptq_shuffle
  vllm13__grouped_topk
  vllm13__hadacore_transform
  vllm13__marlin_int4_fp8_preprocess
  vllm13__merge_attn_states
  vllm13__moe_align_block_size
  vllm13__moe_lora_align_block_size
  vllm13__moe_sum
  vllm13__paged_attention_v1
  vllm13__paged_attention_v2
  vllm13__permute_cols
  vllm13__reshape_and_cache
  vllm13__reshape_and_cache_flash
  vllm13__rms_norm
  vllm13__rms_norm_dynamic_per_token_quant
  vllm13__rms_norm_per_block_quant
  vllm13__rotary_embedding
  vllm13__scaled_fp8_quant
  vllm13__scaled_int8_quant
  vllm13__selective_scan_fwd
  vllm13__shuffle_rows
  vllm13__swap_blocks
  vllm13__topk_softmax
)

TOTAL=${#OPS[@]}
GPUS=(4 5 6 7)
NUM_GPUS=${#GPUS[@]}

echo "[$(date '+%H:%M:%S')] Rerunning $TOTAL missing operators on ${NUM_GPUS} GPUs"

# Worker function
worker() {
    local gpu_id="$1"
    shift
    local ops=("$@")
    for safe_name in "${ops[@]}"; do
        ws="$WORKSPACES/$safe_name"
        prompt_file="$ws/prompt.md"

        if [ ! -f "$prompt_file" ]; then
            echo "[$(date '+%H:%M:%S')] [SKIP] $safe_name - no prompt (GPU=$gpu_id)"
            continue
        fi

        if [ -f "$KERNELS/${safe_name}.py" ]; then
            echo "[$(date '+%H:%M:%S')] [SKIP] $safe_name - kernel exists (GPU=$gpu_id)"
            continue
        fi

        > "$ws/oc_output.json"
        > "$ws/oc.log"
        rm -f "$ws/kernel.py"

        prompt=$(cat "$prompt_file")

        echo "[$(date '+%H:%M:%S')] [START] $safe_name (GPU=$gpu_id)"
        start=$SECONDS

        IS_SANDBOX=1 CUDA_VISIBLE_DEVICES=$gpu_id \
            timeout $TIMEOUT "$OPENCODE_BIN" run "$prompt" \
                --format json \
                --dir "$ws" \
                --model "$OPENCODE_MODEL" \
                > "$ws/oc_output.json" 2>"$ws/oc.log" || true

        elapsed=$((SECONDS - start))

        if [ -f "$ws/kernel.py" ]; then
            cp "$ws/kernel.py" "$KERNELS/${safe_name}.py"
            echo "[$(date '+%H:%M:%S')] [SUCCESS] $safe_name (${elapsed}s, GPU=$gpu_id)"
        else
            echo "[$(date '+%H:%M:%S')] [FAILED] $safe_name (${elapsed}s, GPU=$gpu_id)"
        fi

        sleep 3
    done
}

# Distribute ops round-robin across 4 GPUs
declare -a GPU_OPS_0 GPU_OPS_1 GPU_OPS_2 GPU_OPS_3
for i in "${!OPS[@]}"; do
    case $((i % 4)) in
        0) GPU_OPS_0+=("${OPS[$i]}") ;;
        1) GPU_OPS_1+=("${OPS[$i]}") ;;
        2) GPU_OPS_2+=("${OPS[$i]}") ;;
        3) GPU_OPS_3+=("${OPS[$i]}") ;;
    esac
done

for i in $(seq 0 3); do
    eval "count=\${#GPU_OPS_${i}[@]}"
    echo "GPU ${GPUS[$i]}: $count ops"
done

# Launch 4 workers in parallel
worker "${GPUS[0]}" "${GPU_OPS_0[@]}" 2>&1 | tee "$LOG_DIR/gpu${GPUS[0]}.log" &
worker "${GPUS[1]}" "${GPU_OPS_1[@]}" 2>&1 | tee "$LOG_DIR/gpu${GPUS[1]}.log" &
worker "${GPUS[2]}" "${GPU_OPS_2[@]}" 2>&1 | tee "$LOG_DIR/gpu${GPUS[2]}.log" &
worker "${GPUS[3]}" "${GPU_OPS_3[@]}" 2>&1 | tee "$LOG_DIR/gpu${GPUS[3]}.log" &

wait
echo ""
echo "[$(date '+%H:%M:%S')] All done."

success=0
failed=0
for d in "$WORKSPACES"/*/; do
    op=$(basename "$d")
    if [ -f "$KERNELS/${op}.py" ]; then
        ((success++))
    else
        ((failed++))
    fi
done
echo "Total: $success/210 have kernels, $failed missing"
