#!/usr/bin/env bash
set -Eeuo pipefail

# LLaMA-1-7B dense W4, WikiText2 only.
# Attention uses the existing pure AWQ-delta baseline.  For gate/up/down,
# GPTQ delta replaces AWQ delta only when its separately median-normalized
# value is below 0.7 times the normalized AWQ delta.

CONDA_BIN="${CONDA_BIN:-/home/zch/miniconda3/bin/conda}"
SQLLM_REPO="${SQLLM_REPO:-/home/zch/project/Quantification/SqueezeLLM}"
MODEL_PATH="${MODEL_PATH:-/mnt/dfq/pretrained_model/Llama-7b}"
ORIGINAL_CHUNKS="${ORIGINAL_CHUNKS:-/mnt/dfq/squeezellm_awq_delta/original_chunks}"
AWQ_CHUNKS="${AWQ_CHUNKS:-/mnt/dfq/squeezellm_awq_delta/awq_dequant_chunks}"
GPTQ_CHUNKS="${GPTQ_CHUNKS:-/home/zch/project/Quantification/llama1_gptq_delta_ds/w4/gptq_dequant_chunks_g128}"
AWQ_SCALE_CACHE="${AWQ_SCALE_CACHE:-/home/zch/project/awq_code/llm-awq_zch0722/awq_cache/llama-7b-w4-g128.pt}"
AWQ_BASELINE="${AWQ_BASELINE:-/home/zch/project/Quantification/llama1_w4_dense_gptq_dual_delta/awq_baseline_o3_rest4}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/home/zch/project/Quantification/llama1_w4_dense_delta_fusion_search}"
LUT_FOLDER="$OUTPUT_ROOT/element_conf_rho0p7_mlp"
RESULT_FILE="$OUTPUT_ROOT/results/element_conf_rho0p7_mlp_wikitext.json"
NUM_WORKERS="${NUM_WORKERS:-32}"
MODEL_TYPE="${MODEL_TYPE:-llama}"
PRIMARY_MODULES="${PRIMARY_MODULES:-gate,up,down}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TOKENIZERS_PARALLELISM=false
mkdir -p "$LUT_FOLDER" "$OUTPUT_ROOT/results"

(
    cd "$SQLLM_REPO/quantization"
    "$CONDA_BIN" run --no-capture-output -n sqllm python nuq.py \
        --bit 4 \
        --model_type "$MODEL_TYPE" \
        --model "$ORIGINAL_CHUNKS" \
        --awq_model "$AWQ_CHUNKS" \
        --delta_reference_source awq \
        --secondary_delta_model "$GPTQ_CHUNKS" \
        --secondary_delta_reference_source gptq \
        --delta_reference_selection normalized_confidence \
        --delta_selection_margin 0.7 \
        --awq_delta_weighting scale_inverse_delta \
        --awq_scale_cache "$AWQ_SCALE_CACHE" \
        --awq_scale_power 4.0 \
        --awq_scale_delta_lambda 1.0 \
        --awq_delta_gamma 1.0 \
        --awq_delta_min_weight 0.1 \
        --kmeans_modules "$PRIMARY_MODULES" \
        --output_folder "$LUT_FOLDER" \
        --num_workers "$NUM_WORKERS"
)

(
    cd "$SQLLM_REPO"
    PYTHONPATH="$SQLLM_REPO" \
    "$CONDA_BIN" run --no-capture-output -n awq python llama.py \
        "$MODEL_PATH" wikitext2 \
        --wbits 4 \
        --fake_lut_folder "$LUT_FOLDER" \
        --fake_lut_fallback_folder "$AWQ_BASELINE" \
        --fake_lut_primary_modules "$PRIMARY_MODULES" \
        --fake_weight_chunk_rows 512 \
        --awq_fair_eval \
        --awq_fair_datasets wikitext \
        --awq_fair_batch_size 8 \
        --ppl_output "$RESULT_FILE"
)

python3 - "$RESULT_FILE" <<'PY'
import json
import sys

with open(sys.argv[1]) as handle:
    result = json.load(handle)
print(f"WikiText2 PPL: {result['results']['wikitext']['ppl']:.12f}")
print(f"Result: {sys.argv[1]}")
PY
