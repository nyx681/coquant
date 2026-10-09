#!/usr/bin/env bash
set -Eeuo pipefail

# LLaMA1-7B WikiText2 comparison:
#   q/k Fisher + Fisher*(1 + inverse GPTQ-delta) on v/o/gate/up/down
# against the already-computed all-module Fisher D+S baseline.

if [[ $# -ne 1 || ( "$1" != "3" && "$1" != "4" ) ]]; then
    echo "Usage: bash $0 {3|4}" >&2
    exit 2
fi

BITS="$1"
CONDA_BIN="${CONDA_BIN:-/home/zch/miniconda3/bin/conda}"
SQLLM_ENV="${SQLLM_ENV:-sqllm}"
SQLLM_REPO="${SQLLM_REPO:-/home/zch/project/Quantification/SqueezeLLM}"
MODEL_PATH="${MODEL_PATH:-/mnt/dfq/pretrained_model/Llama-7b}"
ORIGINAL_CHUNKS="${ORIGINAL_CHUNKS:-/mnt/dfq/squeezellm_mixed/model_chunks/Llama-7b}"
GRADIENT_CHUNKS="${GRADIENT_CHUNKS:-/mnt/dfq/squeezellm_mixed/gradient_chunks/Llama-7b}"
WORK_ROOT="${WORK_ROOT:-/home/zch/project/Quantification/llama1_gptq_delta_ds}"
NUM_WORKERS="${NUM_WORKERS:-32}"
PPL_BATCH_SIZE="${PPL_BATCH_SIZE:-8}"
FAKE_WEIGHT_CHUNK_ROWS="${FAKE_WEIGHT_CHUNK_ROWS:-512}"
EXPECTED_LAYERS=32

if [[ "$BITS" == "3" ]]; then
    GPTQ_PARAMS="${GPTQ_PARAMS:-/home/zch/project/Quantification/gptq_artifacts/llama-7b-w3-g128-gptq_params-fixed.pt}"
    FISHER_LUT="${FISHER_LUT:-/mnt/dfq/squeezellm_awq_delta/comparisons/llama1-7b/w3/fisher_ds_s0p1_o0p45}"
    FISHER_RESULT="${FISHER_RESULT:-/mnt/dfq/squeezellm_awq_delta/comparisons/llama1-7b/w3/results/fisher_ds_w3_wikitext.json}"
else
    GPTQ_PARAMS="${GPTQ_PARAMS:-/mnt/dfq/gptq_output/llama-7b-w4-g128-gptq_params-fixed.pt}"
    FISHER_LUT="${FISHER_LUT:-/mnt/dfq/squeezellm_fake/lut/Llama-7b-squeezellm-s0p1-o0p45}"
    FISHER_RESULT="${FISHER_RESULT:-/mnt/dfq/squeezellm_awq_delta/comparisons/llama1-7b/w4/results/fisher_ds_w4_wikitext.json}"
fi

RUN_ROOT="${WORK_ROOT}/w${BITS}"
GPTQ_CHUNKS="${RUN_ROOT}/gptq_dequant_chunks_g128"
DELTA_LUT="${RUN_ROOT}/fisher_mod_inverse_gptq_delta_lambda1_qk_fisher_ds_s0p1_o0p45"
RESULT_DIR="${RUN_ROOT}/results"
DELTA_RESULT="${RESULT_DIR}/gptq_delta_mod_lambda1_qk_fisher_ds_w${BITS}_wikitext.json"
COMPARISON_RESULT="${RESULT_DIR}/comparison_gptq_delta_w${BITS}_wikitext.json"
OUTLIER_CONFIG="${FISHER_LUT}/outlier_config.json"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TOKENIZERS_PARALLELISM=false

fail() {
    echo "ERROR: $*" >&2
    exit 1
}

require_file() {
    [[ -f "$1" ]] || fail "Required file is missing: $1"
}

require_dir() {
    [[ -d "$1" ]] || fail "Required directory is missing: $1"
}

chunk_count() {
    local path="$1"
    [[ -d "$path" ]] || { echo 0; return; }
    find "$path" -maxdepth 1 -type f -name 'layer_*.pt' 2>/dev/null | wc -l
}

lut_complete() {
    local path="$1"
    [[ "$(find "$path/lut" -maxdepth 1 -type f -name 'l*.pkl' 2>/dev/null | wc -l)" -eq "$EXPECTED_LAYERS" ]] &&
        [[ "$(find "$path/outliers" -maxdepth 1 -type f -name 'l*.pkl' 2>/dev/null | wc -l)" -eq "$EXPECTED_LAYERS" ]]
}

result_complete() {
    local path="$1"
    [[ -f "$path" ]] && python3 - "$path" <<'PY' >/dev/null 2>&1
import json, sys
with open(sys.argv[1]) as handle:
    value = json.load(handle)
assert value.get("status") == "complete"
assert value["results"]["wikitext"]["ppl"] > 0
PY
}

require_file "$CONDA_BIN"
require_file "$GPTQ_PARAMS"
require_file "$FISHER_RESULT"
require_file "$OUTLIER_CONFIG"
require_dir "$MODEL_PATH"
require_dir "$ORIGINAL_CHUNKS"
require_dir "$GRADIENT_CHUNKS"
require_dir "$FISHER_LUT/lut"
mkdir -p "$RUN_ROOT" "$RESULT_DIR"

echo "============================================================"
echo "Model:                 LLaMA1-7B"
echo "Dense codebook:        W${BITS}"
echo "Delta reference:       GPTQ W${BITS} g128"
echo "Module policy:         q/k Fisher; v/o/gate/up/down joint"
echo "Joint score:           Fisher_norm * (1 + inverse_delta_norm)"
echo "Sparse policy:         same 0.1% Fisher + 0.45% magnitude mask"
echo "Evaluation:            WikiText2 only"
echo "============================================================"

echo "[1/4] Validating and chunking the GPTQ reference"
if [[ "$(chunk_count "$GPTQ_CHUNKS")" -eq "$EXPECTED_LAYERS" ]]; then
    echo "Reusing GPTQ chunks: $GPTQ_CHUNKS"
else
    "$CONDA_BIN" run --no-capture-output -n "$SQLLM_ENV" \
        python "$SQLLM_REPO/quantization/chunk_gptq_params.py" \
        --gptq_params "$GPTQ_PARAMS" \
        --original_chunks "$ORIGINAL_CHUNKS" \
        --output_path "$GPTQ_CHUNKS" \
        --bit "$BITS" \
        --group_size 128
fi

echo "[2/4] Building q/k Fisher + GPTQ-delta joint LUT"
if lut_complete "$DELTA_LUT"; then
    echo "Reusing complete GPTQ-delta LUT: $DELTA_LUT"
else
    (
        cd "$SQLLM_REPO/quantization"
        "$CONDA_BIN" run --no-capture-output -n "$SQLLM_ENV" python nuq.py \
            --bit "$BITS" \
            --model_type llama \
            --model "$ORIGINAL_CHUNKS" \
            --gradient "$GRADIENT_CHUNKS" \
            --awq_model "$GPTQ_CHUNKS" \
            --delta_reference_source gptq \
            --awq_delta_weighting inverse \
            --awq_delta_gamma 1.0 \
            --awq_delta_min_weight 0.01 \
            --awq_delta_fisher_blend 1.0 \
            --awq_delta_fisher_combine multiplicative \
            --kmeans_modules v,o,gate,up,down \
            --output_folder "$DELTA_LUT" \
            --outlier_config "$OUTLIER_CONFIG" \
            --sensitivity 0.1 \
            --num_workers "$NUM_WORKERS"
    )
fi

echo "[3/4] Evaluating GPTQ-delta method on WikiText2"
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
            --fake_lut_primary_modules v,o,gate,up,down \
            --include_sparse \
            --fake_weight_chunk_rows "$FAKE_WEIGHT_CHUNK_ROWS" \
            --awq_fair_eval \
            --awq_fair_datasets wikitext \
            --awq_fair_batch_size "$PPL_BATCH_SIZE" \
            --ppl_output "$DELTA_RESULT"
    )
fi

echo "[4/4] Writing direct comparison"
python3 - "$BITS" "$GPTQ_PARAMS" "$FISHER_RESULT" "$DELTA_RESULT" "$COMPARISON_RESULT" <<'PY'
import json, os, sys
bits, gptq_params, fisher_path, delta_path, output_path = sys.argv[1:]
with open(fisher_path) as handle:
    fisher = json.load(handle)
with open(delta_path) as handle:
    delta = json.load(handle)
fisher_ppl = fisher["results"]["wikitext"]["ppl"]
delta_ppl = delta["results"]["wikitext"]["ppl"]
fisher_sparse = fisher["fake_quant_stats"]["sparse_percentage"]
delta_sparse = delta["fake_quant_stats"]["sparse_percentage"]
if abs(fisher_sparse - delta_sparse) > 1e-12:
    raise RuntimeError(f"Sparse-rate mismatch: Fisher={fisher_sparse}, delta={delta_sparse}")
comparison = {
    "model": "llama1-7b",
    "bits": int(bits),
    "dataset": "wikitext2",
    "delta_reference": "gptq",
    "gptq_group_size": 128,
    "gptq_params": os.path.abspath(gptq_params),
    "module_policy": "q/k Fisher; v/o/gate/up/down Fisher*(1+inverse GPTQ-delta)",
    "lambda": 1.0,
    "fisher_ds": {"ppl": fisher_ppl, "sparse_percentage": fisher_sparse},
    "gptq_delta_method": {
        "ppl": delta_ppl,
        "sparse_percentage": delta_sparse,
        "delta_ppl_vs_fisher": delta_ppl - fisher_ppl,
        "relative_change_percentage": (delta_ppl / fisher_ppl - 1.0) * 100.0,
    },
}
os.makedirs(os.path.dirname(output_path), exist_ok=True)
with open(output_path, "w") as handle:
    json.dump(comparison, handle, indent=2)
print("LLaMA1-7B W{} WikiText2".format(bits))
print("  Fisher D+S    : {:.6f}".format(fisher_ppl))
print("  GPTQ-delta    : {:.6f}".format(delta_ppl))
print("  Delta-Fisher  : {:+.6f}".format(delta_ppl - fisher_ppl))
print("  Sparse        : {:.6f}% (both)".format(delta_sparse))
print("  JSON          : {}".format(output_path))
PY
