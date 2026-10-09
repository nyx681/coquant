#!/usr/bin/env bash
set -Eeuo pipefail

# End-to-end OPT-6.7B WikiText2 comparison:
#   1. original SqueezeLLM Fisher Dense-and-Sparse
#   2. q/k Fisher + Fisher*(1 + inverse-delta) on v/o/up/down
#
# Usage:
#   bash run_opt67b_delta_vs_fisher_ds.sh 3
#   bash run_opt67b_delta_vs_fisher_ds.sh 4

if [[ $# -ne 1 || ( "$1" != "3" && "$1" != "4" ) ]]; then
    echo "Usage: bash $0 {3|4}" >&2
    exit 2
fi

BITS="$1"
CONDA_BIN="${CONDA_BIN:-/home/zch/miniconda3/bin/conda}"
AWQ_ENV="${AWQ_ENV:-awq}"
SQLLM_ENV="${SQLLM_ENV:-sqllm}"
GRAD_ENV="${GRAD_ENV:-sqllm-grad-fixed}"
AWQ_REPO="${AWQ_REPO:-/home/zch/project/awq_code/llm-awq_zch0722}"
SQLLM_REPO="${SQLLM_REPO:-/home/zch/project/Quantification/SqueezeLLM}"
GRAD_REPO="${GRAD_REPO:-/home/zch/project/SqueezeLLM-gradients}"
MODEL_PATH="${MODEL_PATH:-/mnt/dfq/pretrained_model/opt-6.7b}"
WORK_ROOT="${WORK_ROOT:-/home/zch/project/Quantification/opt67b_delta_ds}"
AWQ_RESULTS="${AWQ_RESULTS:-${AWQ_REPO}/awq_cache/opt-6.7b-w${BITS}-g128.pt}"
NUM_WORKERS="${NUM_WORKERS:-32}"
PPL_BATCH_SIZE="${PPL_BATCH_SIZE:-8}"
FAKE_WEIGHT_CHUNK_ROWS="${FAKE_WEIGHT_CHUNK_ROWS:-512}"
FISHER_EXAMPLES="${FISHER_EXAMPLES:-100}"
OUTLIER_PERCENTAGE="${OUTLIER_PERCENTAGE:-0.45}"
SENSITIVITY="${SENSITIVITY:-0.1}"
EXPECTED_LAYERS=32

SHARED_ROOT="${WORK_ROOT}/shared"
ORIGINAL_CHUNKS="${SHARED_ROOT}/original_chunks"
GRADIENT_HF="${SHARED_ROOT}/gradient_hf_incomplete"
GRADIENT_CHUNKS="${SHARED_ROOT}/gradient_chunks"
OUTLIER_CONFIG_DIR="${SHARED_ROOT}/outlier_config_o${OUTLIER_PERCENTAGE}"
RUN_ROOT="${WORK_ROOT}/w${BITS}"
AWQ_DEQUANT_MODEL="${RUN_ROOT}/awq_dequant_model_incomplete"
AWQ_DEQUANT_CHUNKS="${RUN_ROOT}/awq_dequant_chunks"
FISHER_LUT="${RUN_ROOT}/fisher_ds"
DELTA_LUT="${RUN_ROOT}/fisher_mod_inverse_delta_lambda1_qk_fisher_ds"
RESULT_DIR="${RUN_ROOT}/results"
FISHER_RESULT="${RESULT_DIR}/fisher_ds_w${BITS}_wikitext.json"
DELTA_RESULT="${RESULT_DIR}/delta_mod_lambda1_qk_fisher_ds_w${BITS}_wikitext.json"
COMPARISON_RESULT="${RESULT_DIR}/comparison_w${BITS}_wikitext.json"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TOKENIZERS_PARALLELISM=false

fail() {
    echo "ERROR: $*" >&2
    exit 1
}

require_file() {
    [[ -f "$1" ]] || fail "Required file is missing or unreadable: $1"
}

require_dir() {
    [[ -d "$1" ]] || fail "Required directory is missing or unreadable: $1"
}

chunk_count() {
    local chunk_dir="$1"
    if [[ ! -d "$chunk_dir" ]]; then
        echo 0
        return
    fi
    find "$chunk_dir" -maxdepth 1 -type f -name 'layer_*.pt' 2>/dev/null | wc -l
}

lut_complete() {
    local lut_dir="$1"
    [[ "$(find "$lut_dir/lut" -maxdepth 1 -type f -name 'l*.pkl' 2>/dev/null | wc -l)" -eq "$EXPECTED_LAYERS" ]] &&
        [[ "$(find "$lut_dir/outliers" -maxdepth 1 -type f -name 'l*.pkl' 2>/dev/null | wc -l)" -eq "$EXPECTED_LAYERS" ]]
}

checkpoint_complete() {
    local checkpoint="$1"
    [[ -f "${checkpoint}/config.json" ]] &&
        { [[ -f "${checkpoint}/model.safetensors" ]] ||
          [[ -f "${checkpoint}/model.safetensors.index.json" ]] ||
          [[ -f "${checkpoint}/pytorch_model.bin" ]] ||
          [[ -f "${checkpoint}/pytorch_model.bin.index.json" ]]; }
}

result_complete() {
    local result_file="$1"
    [[ -f "$result_file" ]] && python3 - "$result_file" <<'PY' >/dev/null 2>&1
import json
import sys

with open(sys.argv[1]) as handle:
    result = json.load(handle)
assert result.get("status") == "complete"
assert result["results"]["wikitext"]["ppl"] > 0
PY
}

require_file "$CONDA_BIN"
require_dir "$MODEL_PATH"
require_dir "$AWQ_REPO"
require_dir "$SQLLM_REPO"
require_dir "$GRAD_REPO"
require_file "$AWQ_RESULTS"
mkdir -p "$SHARED_ROOT" "$RUN_ROOT" "$RESULT_DIR"

echo "============================================================"
echo "Model:                 OPT-6.7B"
echo "Dense codebook bits:   $BITS (centers per row: $((2 ** BITS)))"
echo "Sparse policy:         magnitude $OUTLIER_PERCENTAGE% + Fisher $SENSITIVITY%"
echo "Original model:        $MODEL_PATH"
echo "Fisher examples:       $FISHER_EXAMPLES C4 samples"
echo "AWQ search artifact:   $AWQ_RESULTS"
echo "Work root:             $WORK_ROOT"
echo "Datasets:              WikiText2 only"
echo "============================================================"

"$CONDA_BIN" run --no-capture-output -n "$AWQ_ENV" python - "$AWQ_RESULTS" <<'PY'
import sys
import torch

path = sys.argv[1]
artifact = torch.load(path, map_location="cpu")
if not isinstance(artifact, dict) or "scale" not in artifact or "clip" not in artifact:
    raise SystemExit(
        "Expected an AWQ search artifact containing scale/clip, got: " + path
    )
print("Validated AWQ search artifact: {} ({} scale records)".format(
    path, len(artifact["scale"])
))
PY

echo "[1/9] Preparing original OPT model chunks"
if [[ "$(chunk_count "$ORIGINAL_CHUNKS")" -eq "$EXPECTED_LAYERS" ]]; then
    echo "Reusing complete original chunks: $ORIGINAL_CHUNKS"
else
    (
        cd "${SQLLM_REPO}/quantization"
        "$CONDA_BIN" run --no-capture-output -n "$SQLLM_ENV" python chunk_models.py \
            --model "$MODEL_PATH" \
            --model_type opt \
            --output_path "$ORIGINAL_CHUNKS"
    )
fi

echo "[2/9] Preparing OPT Fisher gradient chunks"
if [[ "$(chunk_count "$GRADIENT_CHUNKS")" -eq "$EXPECTED_LAYERS" ]]; then
    echo "Reusing complete Fisher chunks: $GRADIENT_CHUNKS"
else
    if ! checkpoint_complete "$GRADIENT_HF"; then
        (
            cd "$GRAD_REPO"
            "$CONDA_BIN" run --no-capture-output -n "$GRAD_ENV" python run.py \
                --model_name_or_path "$MODEL_PATH" \
                --dataset c4 \
                --num_examples "$FISHER_EXAMPLES" \
                --output_dir "$GRADIENT_HF" \
                --save_safetensors true
        )
    else
        echo "Reusing Fisher HF checkpoint: $GRADIENT_HF"
    fi
    (
        cd "${SQLLM_REPO}/quantization"
        "$CONDA_BIN" run --no-capture-output -n "$SQLLM_ENV" python chunk_models.py \
            --model "$GRADIENT_HF" \
            --model_type opt \
            --output_path "$GRADIENT_CHUNKS"
    )
fi

# The chunked tensors are the durable Fisher input.  Remove only the temporary
# HF-format copy created above so it cannot consume another ~13 GB forever.
if [[ "$(chunk_count "$GRADIENT_CHUNKS")" -eq "$EXPECTED_LAYERS" && -d "$GRADIENT_HF" ]]; then
    rm -rf -- "$GRADIENT_HF"
    echo "Removed temporary Fisher HF checkpoint after successful chunking."
fi

echo "[3/9] Preparing the shared magnitude-outlier configuration"
if compgen -G "${OUTLIER_CONFIG_DIR}/outlier_config_o*.json" >/dev/null; then
    echo "Reusing outlier configuration in: $OUTLIER_CONFIG_DIR"
else
    (
        cd "${SQLLM_REPO}/quantization"
        "$CONDA_BIN" run --no-capture-output -n "$SQLLM_ENV" python generate_outlier_config.py \
            --model "$ORIGINAL_CHUNKS" \
            --model_type opt \
            --target_percentage "$OUTLIER_PERCENTAGE" \
            --output_folder "$OUTLIER_CONFIG_DIR"
    )
fi
mapfile -t OUTLIER_CONFIGS < <(find "$OUTLIER_CONFIG_DIR" -maxdepth 1 -type f -name 'outlier_config_o*.json' | sort)
[[ "${#OUTLIER_CONFIGS[@]}" -eq 1 ]] || fail "Expected exactly one outlier config in $OUTLIER_CONFIG_DIR"
OUTLIER_CONFIG="${OUTLIER_CONFIGS[0]}"
echo "Using outlier configuration: $OUTLIER_CONFIG"

echo "[4/9] Preparing bit-matched AWQ-dequantized W_awq chunks"
if [[ "$(chunk_count "$AWQ_DEQUANT_CHUNKS")" -eq "$EXPECTED_LAYERS" ]]; then
    echo "Reusing complete W_awq chunks: $AWQ_DEQUANT_CHUNKS"
else
    if ! checkpoint_complete "$AWQ_DEQUANT_MODEL"; then
        (
            cd "$AWQ_REPO"
            "$CONDA_BIN" run --no-capture-output -n "$AWQ_ENV" python -m awq.entry \
                --model_path "$MODEL_PATH" \
                --w_bit "$BITS" \
                --q_group_size 128 \
                --q_backend fake \
                --mix_sources awq \
                --load_awq "$AWQ_RESULTS" \
                --dump_fake "$AWQ_DEQUANT_MODEL"
        )
    else
        echo "Reusing AWQ-dequantized checkpoint: $AWQ_DEQUANT_MODEL"
    fi
    (
        cd "${SQLLM_REPO}/quantization"
        "$CONDA_BIN" run --no-capture-output -n "$SQLLM_ENV" python chunk_models.py \
            --model "$AWQ_DEQUANT_MODEL" \
            --model_type opt \
            --output_path "$AWQ_DEQUANT_CHUNKS"
    )
fi

if [[ "$(chunk_count "$AWQ_DEQUANT_CHUNKS")" -eq "$EXPECTED_LAYERS" && -d "$AWQ_DEQUANT_MODEL" ]]; then
    rm -rf -- "$AWQ_DEQUANT_MODEL"
    echo "Removed temporary AWQ HF checkpoint after successful chunking."
fi

echo "[5/9] Building original Fisher Dense-and-Sparse LUT"
if lut_complete "$FISHER_LUT"; then
    echo "Reusing complete Fisher D+S LUT: $FISHER_LUT"
else
    (
        cd "${SQLLM_REPO}/quantization"
        "$CONDA_BIN" run --no-capture-output -n "$SQLLM_ENV" python nuq.py \
            --bit "$BITS" \
            --model_type opt \
            --model "$ORIGINAL_CHUNKS" \
            --gradient "$GRADIENT_CHUNKS" \
            --output_folder "$FISHER_LUT" \
            --outlier_config "$OUTLIER_CONFIG" \
            --sensitivity "$SENSITIVITY" \
            --num_workers "$NUM_WORKERS"
    )
fi

echo "[6/9] Building delta-modulated D+S LUT for v/o/up/down"
if lut_complete "$DELTA_LUT"; then
    echo "Reusing complete delta LUT: $DELTA_LUT"
else
    (
        cd "${SQLLM_REPO}/quantization"
        "$CONDA_BIN" run --no-capture-output -n "$SQLLM_ENV" python nuq.py \
            --bit "$BITS" \
            --model_type opt \
            --model "$ORIGINAL_CHUNKS" \
            --gradient "$GRADIENT_CHUNKS" \
            --awq_model "$AWQ_DEQUANT_CHUNKS" \
            --awq_delta_weighting inverse \
            --awq_delta_gamma 1.0 \
            --awq_delta_min_weight 0.01 \
            --awq_delta_fisher_blend 1.0 \
            --awq_delta_fisher_combine multiplicative \
            --kmeans_modules v,o,up,down \
            --output_folder "$DELTA_LUT" \
            --outlier_config "$OUTLIER_CONFIG" \
            --sensitivity "$SENSITIVITY" \
            --num_workers "$NUM_WORKERS"
    )
fi

echo "[7/9] Evaluating original Fisher D+S on WikiText2"
if result_complete "$FISHER_RESULT"; then
    echo "Reusing complete result: $FISHER_RESULT"
else
    (
        cd "$SQLLM_REPO"
        "$CONDA_BIN" run --no-capture-output -n "$SQLLM_ENV" python llama.py \
            "$MODEL_PATH" wikitext2 \
            --wbits "$BITS" \
            --fake_lut_folder "$FISHER_LUT" \
            --include_sparse \
            --fake_weight_chunk_rows "$FAKE_WEIGHT_CHUNK_ROWS" \
            --awq_fair_eval \
            --awq_fair_datasets wikitext \
            --awq_fair_batch_size "$PPL_BATCH_SIZE" \
            --ppl_output "$FISHER_RESULT"
    )
fi

echo "[8/9] Evaluating q/k Fisher + delta-modulated remaining modules"
if result_complete "$DELTA_RESULT"; then
    echo "Reusing complete result: $DELTA_RESULT"
else
    (
        cd "$SQLLM_REPO"
        "$CONDA_BIN" run --no-capture-output -n "$SQLLM_ENV" python llama.py \
            "$MODEL_PATH" wikitext2 \
            --wbits "$BITS" \
            --fake_lut_folder "$DELTA_LUT" \
            --fake_lut_fallback_folder "$FISHER_LUT" \
            --fake_lut_primary_modules v,o,up,down \
            --include_sparse \
            --fake_weight_chunk_rows "$FAKE_WEIGHT_CHUNK_ROWS" \
            --awq_fair_eval \
            --awq_fair_datasets wikitext \
            --awq_fair_batch_size "$PPL_BATCH_SIZE" \
            --ppl_output "$DELTA_RESULT"
    )
fi

echo "[9/9] Writing direct comparison"
python3 - "$BITS" "$OUTLIER_CONFIG" "$SENSITIVITY" "$FISHER_RESULT" "$DELTA_RESULT" "$COMPARISON_RESULT" <<'PY'
import json
import sys

bits, outlier_path, sensitivity, fisher_path, delta_path, output_path = sys.argv[1:]
with open(outlier_path) as handle:
    outlier_config = json.load(handle)
with open(fisher_path) as handle:
    fisher = json.load(handle)
with open(delta_path) as handle:
    delta = json.load(handle)

fisher_ppl = float(fisher["results"]["wikitext"]["ppl"])
delta_ppl = float(delta["results"]["wikitext"]["ppl"])
comparison = {
    "model": "opt-6.7b",
    "bits": int(bits),
    "dataset": "wikitext2",
    "magnitude_outlier_percentage": outlier_config["outlier_threshold"],
    "fisher_sensitivity_percentage": float(sensitivity),
    "fisher_ds": {
        "ppl": fisher_ppl,
        "sparse_percentage": fisher["fake_quant_stats"]["sparse_percentage"],
        "result_file": fisher_path,
    },
    "qk_fisher_rest_fisher_times_inverse_delta": {
        "ppl": delta_ppl,
        "sparse_percentage": delta["fake_quant_stats"]["sparse_percentage"],
        "lambda": 1.0,
        "delta_ppl_vs_fisher": delta_ppl - fisher_ppl,
        "relative_change_percentage": (delta_ppl - fisher_ppl) / fisher_ppl * 100,
        "result_file": delta_path,
    },
}
with open(output_path, "w") as handle:
    json.dump(comparison, handle, indent=2)

print("\nOPT-6.7B WikiText2 PPL comparison")
print("  Bits          : W{}".format(bits))
print("  Fisher D+S    : {:.6f}".format(fisher_ppl))
print("  Delta method  : {:.6f}".format(delta_ppl))
print("  Delta-Fisher  : {:+.6f}  (negative is better)".format(delta_ppl - fisher_ppl))
print("  Fisher sparse : {:.6f}%".format(comparison["fisher_ds"]["sparse_percentage"]))
print("  Delta sparse  : {:.6f}%".format(comparison["qk_fisher_rest_fisher_times_inverse_delta"]["sparse_percentage"]))
print("  JSON           : {}".format(output_path))
PY
