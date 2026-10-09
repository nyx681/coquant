#!/usr/bin/env bash
set -Eeuo pipefail

# LLaMA-1-7B, Dense W4, WikiText2 only:
#   1) AWQ-scale x inverse GPTQ-delta
#   2) AWQ-scale x inverse elementwise-min(AWQ-delta, GPTQ-delta)
# No Fisher gradients, outlier config, sensitivity, importance mask, or sparse LUT.

CONDA_BIN="${CONDA_BIN:-/home/zch/miniconda3/bin/conda}"
SQLLM_ENV="${SQLLM_ENV:-sqllm}"
EVAL_ENV="${EVAL_ENV:-awq}"
SQLLM_REPO="${SQLLM_REPO:-/home/zch/project/Quantification/SqueezeLLM}"
MODEL_PATH="${MODEL_PATH:-/mnt/dfq/pretrained_model/Llama-7b}"
ORIGINAL_CHUNKS="${ORIGINAL_CHUNKS:-/mnt/dfq/squeezellm_awq_delta/original_chunks}"
AWQ_CHUNKS="${AWQ_CHUNKS:-/mnt/dfq/squeezellm_awq_delta/awq_dequant_chunks}"
GPTQ_CHUNKS="${GPTQ_CHUNKS:-/home/zch/project/Quantification/llama1_gptq_delta_ds/w4/gptq_dequant_chunks_g128}"
AWQ_SCALE_CACHE="${AWQ_SCALE_CACHE:-/home/zch/project/awq_code/llm-awq_zch0722/awq_cache/llama-7b-w4-g128.pt}"
AWQ_POWER3_LUT="${AWQ_POWER3_LUT:-/mnt/dfq/squeezellm_awq_delta/dense_awq_delta_llama1_w4/scale3_inverse_delta_lambda1}"
AWQ_POWER4_LUT="${AWQ_POWER4_LUT:-/mnt/dfq/squeezellm_awq_delta/dense_awq_delta_llama1_w4/scale4_inverse_delta_lambda1}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/home/zch/project/Quantification/llama1_w4_dense_gptq_dual_delta}"
NUM_WORKERS="${NUM_WORKERS:-32}"
PPL_BATCH_SIZE="${PPL_BATCH_SIZE:-8}"
FAKE_WEIGHT_CHUNK_ROWS="${FAKE_WEIGHT_CHUNK_ROWS:-512}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TOKENIZERS_PARALLELISM=false

require_file() {
    [[ -f "$1" ]] || { echo "Missing file: $1" >&2; exit 1; }
}

require_chunks() {
    local path="$1"
    [[ -d "$path" ]] || { echo "Missing chunk directory: $path" >&2; exit 1; }
    local count
    count="$(find "$path" -maxdepth 1 -type f -name 'layer_*.pt' | wc -l)"
    [[ "$count" -eq 32 ]] || {
        echo "Expected 32 chunks in $path, found $count" >&2
        exit 1
    }
}

require_file "$CONDA_BIN"
require_file "$AWQ_SCALE_CACHE"
require_chunks "$ORIGINAL_CHUNKS"
require_chunks "$AWQ_CHUNKS"
require_chunks "$GPTQ_CHUNKS"
[[ -d "$MODEL_PATH" ]] || { echo "Missing model: $MODEL_PATH" >&2; exit 1; }
[[ -d "$AWQ_POWER3_LUT/lut" ]] || { echo "Missing AWQ power-3 LUT: $AWQ_POWER3_LUT" >&2; exit 1; }
[[ -d "$AWQ_POWER4_LUT/lut" ]] || { echo "Missing AWQ power-4 LUT: $AWQ_POWER4_LUT" >&2; exit 1; }
mkdir -p "$OUTPUT_ROOT/results"

build_lut() {
    local mode="$1"
    local power="$2"
    local output="$3"
    local primary_chunks primary_source
    local -a secondary_args=()

    if [[ "$mode" == "gptq" ]]; then
        primary_chunks="$GPTQ_CHUNKS"
        primary_source="gptq"
    elif [[ "$mode" == "dual_min" ]]; then
        primary_chunks="$AWQ_CHUNKS"
        primary_source="awq"
        secondary_args=(
            --secondary_delta_model "$GPTQ_CHUNKS"
            --secondary_delta_reference_source gptq
            --delta_reference_selection minimum
        )
    else
        echo "Unsupported mode: $mode" >&2
        exit 2
    fi

    echo "Building $mode LUT: scale power=$power -> $output"
    (
        cd "$SQLLM_REPO/quantization"
        "$CONDA_BIN" run --no-capture-output -n "$SQLLM_ENV" python nuq.py \
            --bit 4 \
            --model_type llama \
            --model "$ORIGINAL_CHUNKS" \
            --awq_model "$primary_chunks" \
            --delta_reference_source "$primary_source" \
            "${secondary_args[@]}" \
            --awq_delta_weighting scale_inverse_delta \
            --awq_scale_cache "$AWQ_SCALE_CACHE" \
            --awq_scale_power "$power" \
            --awq_scale_delta_lambda 1.0 \
            --awq_delta_gamma 1.0 \
            --awq_delta_min_weight 0.1 \
            --output_folder "$output" \
            --num_workers "$NUM_WORKERS"
    )
}

evaluate_hybrid() {
    local power3_lut="$1"
    local power4_lut="$2"
    local result="$3"

    echo "Evaluating o=power3, q/k/v/gate/up/down=power4 -> $result"
    (
        cd "$SQLLM_REPO"
        PYTHONPATH="$SQLLM_REPO" \
        "$CONDA_BIN" run --no-capture-output -n "$EVAL_ENV" python llama.py \
            "$MODEL_PATH" wikitext2 \
            --wbits 4 \
            --fake_lut_folder "$power3_lut" \
            --fake_lut_fallback_folder "$power4_lut" \
            --fake_lut_primary_modules o \
            --fake_weight_chunk_rows "$FAKE_WEIGHT_CHUNK_ROWS" \
            --awq_fair_eval \
            --awq_fair_datasets wikitext \
            --awq_fair_batch_size "$PPL_BATCH_SIZE" \
            --ppl_output "$result"
    )
}

evaluate_joint_mlp() {
    local joint_lut="$1"
    local awq_baseline="$2"
    local result="$3"

    echo "Evaluating joint min-delta MLP with AWQ-delta attention -> $result"
    (
        cd "$SQLLM_REPO"
        PYTHONPATH="$SQLLM_REPO" \
        "$CONDA_BIN" run --no-capture-output -n "$EVAL_ENV" python llama.py \
            "$MODEL_PATH" wikitext2 \
            --wbits 4 \
            --fake_lut_folder "$joint_lut" \
            --fake_lut_fallback_folder "$awq_baseline" \
            --fake_lut_primary_modules gate,up,down \
            --fake_weight_chunk_rows "$FAKE_WEIGHT_CHUNK_ROWS" \
            --awq_fair_eval \
            --awq_fair_datasets wikitext \
            --awq_fair_batch_size "$PPL_BATCH_SIZE" \
            --ppl_output "$result"
    )
}

GPTQ_P3="$OUTPUT_ROOT/gptq_delta_scale3_lambda1"
GPTQ_P4="$OUTPUT_ROOT/gptq_delta_scale4_lambda1"
DUAL_P3="$OUTPUT_ROOT/awq_gptq_min_delta_scale3_lambda1"
DUAL_P4="$OUTPUT_ROOT/awq_gptq_min_delta_scale4_lambda1"
GPTQ_RESULT="$OUTPUT_ROOT/results/gptq_delta_o3_rest4_wikitext.json"
DUAL_RESULT="$OUTPUT_ROOT/results/awq_gptq_min_delta_o3_rest4_wikitext.json"
AWQ_BASELINE="$OUTPUT_ROOT/awq_baseline_o3_rest4"
JOINT_RESULT="$OUTPUT_ROOT/results/joint_mlp_min4_awq_baseline_wikitext.json"

build_lut gptq 3.0 "$GPTQ_P3"
build_lut gptq 4.0 "$GPTQ_P4"
evaluate_hybrid "$GPTQ_P3" "$GPTQ_P4" "$GPTQ_RESULT"

build_lut dual_min 3.0 "$DUAL_P3"
build_lut dual_min 4.0 "$DUAL_P4"
evaluate_hybrid "$DUAL_P3" "$DUAL_P4" "$DUAL_RESULT"

"$CONDA_BIN" run --no-capture-output -n "$SQLLM_ENV" \
    python "$SQLLM_REPO/quantization/compose_lut_modules.py" \
    --base_folder "$AWQ_POWER4_LUT" \
    --override_folder "$AWQ_POWER3_LUT" \
    --override_modules o \
    --output_folder "$AWQ_BASELINE"
evaluate_joint_mlp "$DUAL_P4" "$AWQ_BASELINE" "$JOINT_RESULT"

python3 - "$GPTQ_RESULT" "$DUAL_RESULT" "$JOINT_RESULT" <<'PY'
import json
import sys

labels = ("GPTQ-delta", "AWQ/GPTQ min-delta (all modules)", "Joint MLP min-delta")
for label, path in zip(labels, sys.argv[1:]):
    with open(path) as handle:
        result = json.load(handle)
    stats = result["fake_quant_stats"]
    print(
        f"{label}: WikiText2 PPL={result['results']['wikitext']['ppl']:.9f}, "
        f"sparse={stats['sparse_count']}/{stats['weight_count']}"
    )
    print(f"  {path}")
PY
