#!/usr/bin/env bash
set -Eeuo pipefail

# LLaMA-2-7B W4 Dense AWQ/GPTQ-delta experiment, WikiText2 only.
# No Fisher and no Dense-and-Sparse component.

CONDA_BIN="${CONDA_BIN:-/home/zch/miniconda3/bin/conda}"
SQLLM_ENV="${SQLLM_ENV:-sqllm}"
EVAL_ENV="${EVAL_ENV:-awq}"
SQLLM_REPO="${SQLLM_REPO:-/home/zch/project/Quantification/SqueezeLLM}"
MODEL_PATH="${MODEL_PATH:-/mnt/dfq/pretrained_model/llama-2-7b}"
ORIGINAL_CHUNKS="${ORIGINAL_CHUNKS:-/mnt/dfq/squeezellm_mixed/model_chunks/llama-2-7b}"
AWQ_CHUNKS="${AWQ_CHUNKS:-/mnt/dfq/squeezellm_awq_delta/llama2/awq_dequant_chunks}"
GPTQ_PARAMS="${GPTQ_PARAMS:-/mnt/dfq/gptq_output/llama-2-7b-w4-g128-gptq_params.pt}"
AWQ_SCALE_CACHE="${AWQ_SCALE_CACHE:-/home/zch/project/awq_code/llm-awq_zch0722/awq_cache/llama-2-7b-w4-g128.pt}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/home/zch/project/Quantification/llama2_w4_dense_joint_delta}"
GPTQ_CHUNKS="${GPTQ_CHUNKS:-$OUTPUT_ROOT/gptq_dequant_chunks_g128}"
AWQ_POWER3_LUT="${AWQ_POWER3_LUT:-$OUTPUT_ROOT/awq_delta_scale3_lambda1}"
AWQ_POWER4_LUT="${AWQ_POWER4_LUT:-$OUTPUT_ROOT/awq_delta_scale4_lambda1}"
NUM_WORKERS="${NUM_WORKERS:-32}"
PPL_BATCH_SIZE="${PPL_BATCH_SIZE:-8}"
FAKE_WEIGHT_CHUNK_ROWS="${FAKE_WEIGHT_CHUNK_ROWS:-512}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TOKENIZERS_PARALLELISM=false

chunk_count() {
    [[ -d "$1" ]] || { echo 0; return; }
    find "$1" -maxdepth 1 -type f -name 'layer_*.pt' | wc -l
}

for path in "$MODEL_PATH" "$ORIGINAL_CHUNKS" "$AWQ_CHUNKS"; do
    [[ -d "$path" ]] || { echo "Missing directory: $path" >&2; exit 1; }
done
for path in "$CONDA_BIN" "$GPTQ_PARAMS" "$AWQ_SCALE_CACHE"; do
    [[ -f "$path" ]] || { echo "Missing file: $path" >&2; exit 1; }
done
[[ "$(chunk_count "$ORIGINAL_CHUNKS")" -eq 32 ]] || { echo "Original chunks are incomplete" >&2; exit 1; }
[[ "$(chunk_count "$AWQ_CHUNKS")" -eq 32 ]] || { echo "AWQ chunks are incomplete" >&2; exit 1; }
mkdir -p "$OUTPUT_ROOT/results"

if [[ "$(chunk_count "$GPTQ_CHUNKS")" -ne 32 ]]; then
    "$CONDA_BIN" run --no-capture-output -n "$SQLLM_ENV" \
        python "$SQLLM_REPO/quantization/chunk_gptq_params.py" \
        --gptq_params "$GPTQ_PARAMS" \
        --original_chunks "$ORIGINAL_CHUNKS" \
        --output_path "$GPTQ_CHUNKS" \
        --bit 4 \
        --group_size 128
fi

build_awq_lut() {
    local power="$1"
    local output="$2"
    (
        cd "$SQLLM_REPO/quantization"
        "$CONDA_BIN" run --no-capture-output -n "$SQLLM_ENV" python nuq.py \
            --bit 4 \
            --model_type llama \
            --model "$ORIGINAL_CHUNKS" \
            --awq_model "$AWQ_CHUNKS" \
            --delta_reference_source awq \
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

build_awq_lut 3.0 "$AWQ_POWER3_LUT"
build_awq_lut 4.0 "$AWQ_POWER4_LUT"

MODEL_PATH="$MODEL_PATH" \
ORIGINAL_CHUNKS="$ORIGINAL_CHUNKS" \
AWQ_CHUNKS="$AWQ_CHUNKS" \
GPTQ_CHUNKS="$GPTQ_CHUNKS" \
AWQ_SCALE_CACHE="$AWQ_SCALE_CACHE" \
AWQ_POWER3_LUT="$AWQ_POWER3_LUT" \
AWQ_POWER4_LUT="$AWQ_POWER4_LUT" \
OUTPUT_ROOT="$OUTPUT_ROOT" \
NUM_WORKERS="$NUM_WORKERS" \
PPL_BATCH_SIZE="$PPL_BATCH_SIZE" \
FAKE_WEIGHT_CHUNK_ROWS="$FAKE_WEIGHT_CHUNK_ROWS" \
CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
bash "$SQLLM_REPO/run_llama1_w4_dense_gptq_and_dual_delta.sh"

AWQ_BASELINE="$OUTPUT_ROOT/awq_baseline_o3_rest4"
AWQ_RESULT="$OUTPUT_ROOT/results/awq_delta_o3_rest4_wikitext.json"
(
    cd "$SQLLM_REPO"
    PYTHONPATH="$SQLLM_REPO" \
    "$CONDA_BIN" run --no-capture-output -n "$EVAL_ENV" python llama.py \
        "$MODEL_PATH" wikitext2 \
        --wbits 4 \
        --fake_lut_folder "$AWQ_BASELINE" \
        --fake_weight_chunk_rows "$FAKE_WEIGHT_CHUNK_ROWS" \
        --awq_fair_eval \
        --awq_fair_datasets wikitext \
        --awq_fair_batch_size "$PPL_BATCH_SIZE" \
        --ppl_output "$AWQ_RESULT"
)

python3 - "$OUTPUT_ROOT/results" <<'PY'
import json
import os
import sys

folder = sys.argv[1]
files = (
    ("AWQ-delta", "awq_delta_o3_rest4_wikitext.json"),
    ("GPTQ-delta", "gptq_delta_o3_rest4_wikitext.json"),
    ("all-module min-delta", "awq_gptq_min_delta_o3_rest4_wikitext.json"),
    ("joint MLP min-delta", "joint_mlp_min4_awq_baseline_wikitext.json"),
)
print("LLaMA-2-7B W4 WikiText2 results")
for label, name in files:
    path = os.path.join(folder, name)
    with open(path) as handle:
        result = json.load(handle)
    stats = result["fake_quant_stats"]
    print(
        f"  {label:24s} PPL={result['results']['wikitext']['ppl']:.9f}, "
        f"sparse={stats['sparse_count']}"
    )
PY

