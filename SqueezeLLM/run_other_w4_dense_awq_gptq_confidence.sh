#!/usr/bin/env bash
set -Eeuo pipefail

# Compare three dense W4 methods on WikiText2:
#   1) pure AWQ-delta
#   2) pure GPTQ-delta
#   3) AWQ attention + normalized-confidence AWQ/GPTQ delta on the MLP

if [[ $# -ne 1 || ( "$1" != "llama2" && "$1" != "opt" ) ]]; then
    echo "Usage: bash $0 {llama2|opt}" >&2
    exit 2
fi

TARGET="$1"
CONDA_BIN="${CONDA_BIN:-/home/zch/miniconda3/bin/conda}"
SQLLM_REPO="${SQLLM_REPO:-/home/zch/project/Quantification/SqueezeLLM}"
NUM_WORKERS="${NUM_WORKERS:-32}"

case "$TARGET" in
    llama2)
        MODEL_TYPE=llama
        PRIMARY_MODULES=gate,up,down
        MODEL_PATH=/mnt/dfq/pretrained_model/llama-2-7b
        ORIGINAL_CHUNKS=/mnt/dfq/squeezellm_mixed/model_chunks/llama-2-7b
        AWQ_CHUNKS=/mnt/dfq/squeezellm_awq_delta/llama2/awq_dequant_chunks
        GPTQ_PARAMS=/mnt/dfq/gptq_output/llama-2-7b-w4-g128-gptq_params.pt
        AWQ_SCALE_CACHE=/home/zch/project/awq_code/llm-awq_zch0722/awq_cache/llama-2-7b-w4-g128.pt
        OUTPUT_ROOT=/home/zch/project/Quantification/llama2_w4_dense_joint_delta
        ;;
    opt)
        MODEL_TYPE=opt
        PRIMARY_MODULES=up,down
        MODEL_PATH=/mnt/dfq/pretrained_model/opt-6.7b
        ORIGINAL_CHUNKS=/home/zch/project/Quantification/opt67b_delta_ds/shared/original_chunks
        AWQ_CHUNKS=/home/zch/project/Quantification/opt67b_delta_ds/w4/awq_dequant_chunks
        GPTQ_PARAMS=/mnt/dfq/gptq_output/opt-6.7b-w4-g128-gptq_params.pt
        AWQ_SCALE_CACHE=/home/zch/project/awq_code/llm-awq_zch0722/awq_cache/opt-6.7b-w4-g128.pt
        OUTPUT_ROOT=/home/zch/project/Quantification/opt67b_w4_dense_delta_confidence
        ;;
esac

GPTQ_CHUNKS="$OUTPUT_ROOT/gptq_dequant_chunks_g128"
AWQ_POWER3_LUT="$OUTPUT_ROOT/awq_delta_scale3_lambda1"
AWQ_POWER4_LUT="$OUTPUT_ROOT/awq_delta_scale4_lambda1"
AWQ_BASELINE="$OUTPUT_ROOT/awq_baseline_o3_rest4"
GPTQ_POWER3_LUT="$OUTPUT_ROOT/gptq_delta_scale3_lambda1"
GPTQ_POWER4_LUT="$OUTPUT_ROOT/gptq_delta_scale4_lambda1"
RESULT_DIR="$OUTPUT_ROOT/results"
AWQ_RESULT="$RESULT_DIR/awq_delta_o3_rest4_wikitext.json"
GPTQ_RESULT="$RESULT_DIR/gptq_delta_o3_rest4_wikitext.json"
COMBINED_RESULT="$RESULT_DIR/element_conf_rho0p7_mlp_wikitext.json"

chunk_count() {
    [[ -d "$1" ]] || { echo 0; return; }
    find "$1" -maxdepth 1 -type f -name 'layer_*.pt' | wc -l
}

result_complete() {
    [[ -f "$1" ]] && python3 - "$1" <<'PY' >/dev/null 2>&1
import json
import sys

with open(sys.argv[1]) as handle:
    value = json.load(handle)
assert value["results"]["wikitext"]["ppl"] > 0
PY
}

for path in "$MODEL_PATH" "$ORIGINAL_CHUNKS" "$AWQ_CHUNKS"; do
    [[ -d "$path" ]] || { echo "Missing directory: $path" >&2; exit 1; }
done
for path in "$CONDA_BIN" "$GPTQ_PARAMS" "$AWQ_SCALE_CACHE"; do
    [[ -f "$path" ]] || { echo "Missing file: $path" >&2; exit 1; }
done
[[ "$(chunk_count "$ORIGINAL_CHUNKS")" -eq 32 ]] || { echo "Incomplete original chunks" >&2; exit 1; }
[[ "$(chunk_count "$AWQ_CHUNKS")" -eq 32 ]] || { echo "Incomplete AWQ chunks" >&2; exit 1; }
mkdir -p "$OUTPUT_ROOT" "$RESULT_DIR"

if [[ "$(chunk_count "$GPTQ_CHUNKS")" -ne 32 ]]; then
    "$CONDA_BIN" run --no-capture-output -n sqllm \
        python "$SQLLM_REPO/quantization/chunk_gptq_params.py" \
        --gptq_params "$GPTQ_PARAMS" \
        --original_chunks "$ORIGINAL_CHUNKS" \
        --output_path "$GPTQ_CHUNKS" \
        --model_type "$MODEL_TYPE" \
        --bit 4 \
        --group_size 128
fi

build_awq_lut() {
    local power="$1"
    local output="$2"
    (
        cd "$SQLLM_REPO/quantization"
        "$CONDA_BIN" run --no-capture-output -n sqllm python nuq.py \
            --bit 4 \
            --model_type "$MODEL_TYPE" \
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

build_gptq_lut() {
    local power="$1"
    local output="$2"
    (
        cd "$SQLLM_REPO/quantization"
        "$CONDA_BIN" run --no-capture-output -n sqllm python nuq.py \
            --bit 4 \
            --model_type "$MODEL_TYPE" \
            --model "$ORIGINAL_CHUNKS" \
            --awq_model "$GPTQ_CHUNKS" \
            --delta_reference_source gptq \
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

evaluate_single_lut() {
    local lut_folder="$1"
    local result_file="$2"
    (
        cd "$SQLLM_REPO"
        PYTHONPATH="$SQLLM_REPO" \
        "$CONDA_BIN" run --no-capture-output -n awq python llama.py \
            "$MODEL_PATH" wikitext2 \
            --wbits 4 \
            --fake_lut_folder "$lut_folder" \
            --fake_weight_chunk_rows 512 \
            --awq_fair_eval \
            --awq_fair_datasets wikitext \
            --awq_fair_batch_size 8 \
            --ppl_output "$result_file"
    )
}

evaluate_o3_rest4() {
    local power3_lut="$1"
    local power4_lut="$2"
    local result_file="$3"
    (
        cd "$SQLLM_REPO"
        PYTHONPATH="$SQLLM_REPO" \
        "$CONDA_BIN" run --no-capture-output -n awq python llama.py \
            "$MODEL_PATH" wikitext2 \
            --wbits 4 \
            --fake_lut_folder "$power3_lut" \
            --fake_lut_fallback_folder "$power4_lut" \
            --fake_lut_primary_modules o \
            --fake_weight_chunk_rows 512 \
            --awq_fair_eval \
            --awq_fair_datasets wikitext \
            --awq_fair_batch_size 8 \
            --ppl_output "$result_file"
    )
}

build_awq_lut 3.0 "$AWQ_POWER3_LUT"
build_awq_lut 4.0 "$AWQ_POWER4_LUT"

"$CONDA_BIN" run --no-capture-output -n sqllm \
    python "$SQLLM_REPO/quantization/compose_lut_modules.py" \
    --base_folder "$AWQ_POWER4_LUT" \
    --override_folder "$AWQ_POWER3_LUT" \
    --override_modules o \
    --output_folder "$AWQ_BASELINE"

echo "[1/3] Evaluating pure AWQ-delta"
if result_complete "$AWQ_RESULT"; then
    echo "Reusing complete AWQ-delta result: $AWQ_RESULT"
else
    evaluate_single_lut "$AWQ_BASELINE" "$AWQ_RESULT"
fi

echo "[2/3] Building and evaluating pure GPTQ-delta"
if result_complete "$GPTQ_RESULT"; then
    echo "Reusing complete GPTQ-delta result: $GPTQ_RESULT"
else
    build_gptq_lut 3.0 "$GPTQ_POWER3_LUT"
    build_gptq_lut 4.0 "$GPTQ_POWER4_LUT"
    evaluate_o3_rest4 "$GPTQ_POWER3_LUT" "$GPTQ_POWER4_LUT" "$GPTQ_RESULT"
fi

echo "[3/3] Building and evaluating AWQ+GPTQ confidence delta"
MODEL_TYPE="$MODEL_TYPE" \
PRIMARY_MODULES="$PRIMARY_MODULES" \
MODEL_PATH="$MODEL_PATH" \
ORIGINAL_CHUNKS="$ORIGINAL_CHUNKS" \
AWQ_CHUNKS="$AWQ_CHUNKS" \
GPTQ_CHUNKS="$GPTQ_CHUNKS" \
AWQ_SCALE_CACHE="$AWQ_SCALE_CACHE" \
AWQ_BASELINE="$AWQ_BASELINE" \
OUTPUT_ROOT="$OUTPUT_ROOT" \
NUM_WORKERS="$NUM_WORKERS" \
bash "$SQLLM_REPO/run_llama1_w4_dense_awq_gptq_confidence.sh"

python3 - "$AWQ_RESULT" "$GPTQ_RESULT" "$COMBINED_RESULT" <<'PY'
import json
import sys

labels = ("AWQ-delta", "GPTQ-delta", "AWQ+GPTQ confidence")
print("\nDense W4 WikiText2 comparison")
for label, path in zip(labels, sys.argv[1:]):
    with open(path) as handle:
        value = json.load(handle)
    stats = value["fake_quant_stats"]
    print(
        f"  {label:24s} PPL={value['results']['wikitext']['ppl']:.9f}, "
        f"sparse={stats['sparse_count']}"
    )
    print(f"    {path}")
PY
