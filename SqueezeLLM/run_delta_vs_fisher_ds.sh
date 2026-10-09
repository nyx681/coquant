#!/usr/bin/env bash
set -Eeuo pipefail

# Compare the best AWQ-delta K-means variant with the original SqueezeLLM
# Fisher Dense-and-Sparse baseline on WikiText2.
#
# Usage:
#   bash run_delta_vs_fisher_ds.sh llama2 3
#   bash run_delta_vs_fisher_ds.sh llama1 4
#   bash run_delta_vs_fisher_ds.sh llama1 3
#
# The comparison keeps the sparse selection, bit width, source model, Fisher
# tensors and evaluator identical.  Only the dense K-means sample weights of
# v/o/gate/up/down change.  q/k are loaded from the bit-matched Fisher LUT.

if [[ $# -ne 2 ]]; then
    echo "Usage: bash $0 {llama1|llama2} {3|4}" >&2
    exit 2
fi

MODEL_FAMILY="$1"
BITS="$2"
if [[ "$MODEL_FAMILY" != "llama1" && "$MODEL_FAMILY" != "llama2" ]]; then
    echo "MODEL_FAMILY must be llama1 or llama2; got: $MODEL_FAMILY" >&2
    exit 2
fi
if [[ "$BITS" != "3" && "$BITS" != "4" ]]; then
    echo "BITS must be 3 or 4; got: $BITS" >&2
    exit 2
fi

CONDA_BIN="${CONDA_BIN:-/home/zch/miniconda3/bin/conda}"
AWQ_ENV="${AWQ_ENV:-awq}"
SQLLM_ENV="${SQLLM_ENV:-sqllm}"
AWQ_REPO="${AWQ_REPO:-/home/zch/project/awq_code/llm-awq_zch0722}"
SQLLM_REPO="${SQLLM_REPO:-/home/zch/project/Quantification/SqueezeLLM}"
DATA_ROOT="${DATA_ROOT:-/mnt/dfq}"
WORK_BASE="${WORK_BASE:-${DATA_ROOT}/squeezellm_awq_delta/comparisons}"
NUM_WORKERS="${NUM_WORKERS:-32}"
PPL_BATCH_SIZE="${PPL_BATCH_SIZE:-8}"
FAKE_WEIGHT_CHUNK_ROWS="${FAKE_WEIGHT_CHUNK_ROWS:-512}"
EXPECTED_LAYERS=32

case "$MODEL_FAMILY" in
    llama1)
        MODEL_TAG="llama1-7b"
        MODEL_PATH_DEFAULT="${DATA_ROOT}/pretrained_model/Llama-7b"
        GRADIENT_CHUNKS_DEFAULT="${DATA_ROOT}/squeezellm_mixed/gradient_chunks/Llama-7b"
        LEGACY_MODEL_CHUNKS_1="${DATA_ROOT}/squeezellm_mixed/model_chunks/Llama-7b"
        LEGACY_MODEL_CHUNKS_2="${DATA_ROOT}/squeezellm_awq_delta/original_chunks"
        LEGACY_FISHER_W4_LUT="${DATA_ROOT}/squeezellm_fake/lut/Llama-7b-squeezellm-s0p1-o0p45"
        LEGACY_AWQ_W4_CHUNKS="${DATA_ROOT}/squeezellm_awq_delta/awq_dequant_chunks"
        AWQ_RESULTS_DEFAULT="${AWQ_REPO}/awq_cache/llama-7b-w${BITS}-g128.pt"
        ;;
    llama2)
        MODEL_TAG="llama2-7b"
        MODEL_PATH_DEFAULT="${DATA_ROOT}/pretrained_model/llama-2-7b"
        GRADIENT_CHUNKS_DEFAULT="${DATA_ROOT}/squeezellm_mixed/gradient_chunks/llama-2-7b"
        LEGACY_MODEL_CHUNKS_1="${DATA_ROOT}/squeezellm_mixed/model_chunks/llama-2-7b"
        LEGACY_MODEL_CHUNKS_2="${DATA_ROOT}/squeezellm_awq_delta/llama2/original_chunks"
        LEGACY_FISHER_W4_LUT="${DATA_ROOT}/squeezellm_fake/lut/llama-2-7b-squeezellm-s0p1-o0p45"
        LEGACY_AWQ_W4_CHUNKS="${DATA_ROOT}/squeezellm_awq_delta/llama2/awq_dequant_chunks"
        AWQ_RESULTS_DEFAULT="${AWQ_REPO}/awq_cache/llama-2-7b-w${BITS}-g128.pt"
        ;;
esac

MODEL_PATH="${MODEL_PATH:-$MODEL_PATH_DEFAULT}"
GRADIENT_CHUNKS="${GRADIENT_CHUNKS:-$GRADIENT_CHUNKS_DEFAULT}"
AWQ_RESULTS="${AWQ_RESULTS:-$AWQ_RESULTS_DEFAULT}"
RUN_ROOT="${WORK_BASE}/${MODEL_TAG}/w${BITS}"
SHARED_ROOT="${WORK_BASE}/${MODEL_TAG}/shared"
AWQ_DEQUANT_MODEL="${AWQ_DEQUANT_MODEL:-${RUN_ROOT}/awq-dequant-w${BITS}-g128}"
AWQ_DEQUANT_CHUNKS_DEFAULT="${RUN_ROOT}/awq_dequant_chunks"
FISHER_GENERATED_LUT="${RUN_ROOT}/fisher_ds_s0p1_o0p45"
DELTA_LUT="${RUN_ROOT}/fisher_mod_inverse_delta_lambda1_qk_fisher_ds_s0p1_o0p45"
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
    [[ -f "${AWQ_DEQUANT_MODEL}/config.json" ]] &&
        { [[ -f "${AWQ_DEQUANT_MODEL}/model.safetensors" ]] ||
          [[ -f "${AWQ_DEQUANT_MODEL}/model.safetensors.index.json" ]] ||
          [[ -f "${AWQ_DEQUANT_MODEL}/pytorch_model.bin" ]] ||
          [[ -f "${AWQ_DEQUANT_MODEL}/pytorch_model.bin.index.json" ]]; }
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

# Prefer already-generated shared chunks.  Explicit environment overrides win.
if [[ -n "${ORIGINAL_CHUNKS:-}" ]]; then
    ORIGINAL_CHUNKS="$ORIGINAL_CHUNKS"
elif [[ "$(chunk_count "$LEGACY_MODEL_CHUNKS_1")" -eq "$EXPECTED_LAYERS" ]]; then
    ORIGINAL_CHUNKS="$LEGACY_MODEL_CHUNKS_1"
elif [[ "$(chunk_count "$LEGACY_MODEL_CHUNKS_2")" -eq "$EXPECTED_LAYERS" ]]; then
    ORIGINAL_CHUNKS="$LEGACY_MODEL_CHUNKS_2"
else
    ORIGINAL_CHUNKS="${SHARED_ROOT}/original_chunks"
fi

# The older Llama-1 W4 experiment already has W_awq chunks; reuse them when
# possible.  W3 always needs its own bit-matched AWQ dequantization.
if [[ -n "${AWQ_DEQUANT_CHUNKS:-}" ]]; then
    AWQ_DEQUANT_CHUNKS="$AWQ_DEQUANT_CHUNKS"
elif [[ "$BITS" == "4" && "$(chunk_count "$LEGACY_AWQ_W4_CHUNKS")" -eq "$EXPECTED_LAYERS" ]]; then
    AWQ_DEQUANT_CHUNKS="$LEGACY_AWQ_W4_CHUNKS"
else
    AWQ_DEQUANT_CHUNKS="$AWQ_DEQUANT_CHUNKS_DEFAULT"
fi

# Reuse the previously generated W4 Fisher D+S LUT.  For W3, generate a
# bit-matched Fisher baseline in this run directory.
if [[ -n "${FISHER_BASELINE_LUT:-}" ]]; then
    FISHER_BASELINE_LUT="$FISHER_BASELINE_LUT"
elif [[ "$BITS" == "4" ]] && lut_complete "$LEGACY_FISHER_W4_LUT"; then
    FISHER_BASELINE_LUT="$LEGACY_FISHER_W4_LUT"
else
    FISHER_BASELINE_LUT="$FISHER_GENERATED_LUT"
fi
OUTLIER_CONFIG="${OUTLIER_CONFIG:-${LEGACY_FISHER_W4_LUT}/outlier_config.json}"

# This also catches the current /mnt/dfq I/O-error state before any output is
# created.  Repair/remount that filesystem first if this check fails.
require_file "$CONDA_BIN"
require_dir "$MODEL_PATH"
require_dir "$GRADIENT_CHUNKS"
require_dir "$AWQ_REPO"
require_dir "$SQLLM_REPO"
require_file "$AWQ_RESULTS"
require_file "$OUTLIER_CONFIG"
mkdir -p "$RUN_ROOT" "$SHARED_ROOT" "$RESULT_DIR"

echo "============================================================"
echo "Model:                 $MODEL_TAG"
echo "Dense codebook bits:   $BITS (centers per row: $((2 ** BITS)))"
echo "Original model:        $MODEL_PATH"
echo "Original chunks:       $ORIGINAL_CHUNKS"
echo "Fisher chunks:         $GRADIENT_CHUNKS"
echo "AWQ search artifact:   $AWQ_RESULTS"
echo "AWQ W_awq chunks:      $AWQ_DEQUANT_CHUNKS"
echo "Fisher D+S LUT:        $FISHER_BASELINE_LUT"
echo "Delta D+S LUT:         $DELTA_LUT"
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

echo "[1/7] Preparing original FP model chunks"
if [[ "$(chunk_count "$ORIGINAL_CHUNKS")" -eq "$EXPECTED_LAYERS" ]]; then
    echo "Reusing complete original chunks: $ORIGINAL_CHUNKS"
else
    (
        cd "${SQLLM_REPO}/quantization"
        "$CONDA_BIN" run --no-capture-output -n "$SQLLM_ENV" python chunk_models.py \
            --model "$MODEL_PATH" \
            --model_type llama \
            --output_path "$ORIGINAL_CHUNKS"
    )
fi

echo "[2/7] Preparing bit-matched AWQ-dequantized W_awq chunks"
if [[ "$(chunk_count "$AWQ_DEQUANT_CHUNKS")" -eq "$EXPECTED_LAYERS" ]]; then
    echo "Reusing complete W_awq chunks: $AWQ_DEQUANT_CHUNKS"
else
    if ! checkpoint_complete; then
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
            --model_type llama \
            --output_path "$AWQ_DEQUANT_CHUNKS"
    )
fi

echo "[3/7] Preparing original Fisher Dense-and-Sparse LUT"
if lut_complete "$FISHER_BASELINE_LUT"; then
    echo "Reusing complete Fisher D+S LUT: $FISHER_BASELINE_LUT"
else
    (
        cd "${SQLLM_REPO}/quantization"
        "$CONDA_BIN" run --no-capture-output -n "$SQLLM_ENV" python nuq.py \
            --bit "$BITS" \
            --model_type llama \
            --model "$ORIGINAL_CHUNKS" \
            --gradient "$GRADIENT_CHUNKS" \
            --output_folder "$FISHER_BASELINE_LUT" \
            --outlier_config "$OUTLIER_CONFIG" \
            --sensitivity 0.1 \
            --num_workers "$NUM_WORKERS"
    )
fi

echo "[4/7] Building delta-modulated D+S LUT for v/o/gate/up/down"
if lut_complete "$DELTA_LUT"; then
    echo "Reusing complete delta LUT: $DELTA_LUT"
else
    (
        cd "${SQLLM_REPO}/quantization"
        "$CONDA_BIN" run --no-capture-output -n "$SQLLM_ENV" python nuq.py \
            --bit "$BITS" \
            --model_type llama \
            --model "$ORIGINAL_CHUNKS" \
            --gradient "$GRADIENT_CHUNKS" \
            --awq_model "$AWQ_DEQUANT_CHUNKS" \
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

echo "[5/7] Evaluating original Fisher D+S on WikiText2"
if result_complete "$FISHER_RESULT"; then
    echo "Reusing complete result: $FISHER_RESULT"
else
    (
        cd "$SQLLM_REPO"
        "$CONDA_BIN" run --no-capture-output -n "$SQLLM_ENV" python llama.py \
            "$MODEL_PATH" wikitext2 \
            --wbits "$BITS" \
            --fake_lut_folder "$FISHER_BASELINE_LUT" \
            --include_sparse \
            --fake_weight_chunk_rows "$FAKE_WEIGHT_CHUNK_ROWS" \
            --awq_fair_eval \
            --awq_fair_datasets wikitext \
            --awq_fair_batch_size "$PPL_BATCH_SIZE" \
            --ppl_output "$FISHER_RESULT"
    )
fi

echo "[6/7] Evaluating q/k Fisher + delta-modulated remaining modules"
if result_complete "$DELTA_RESULT"; then
    echo "Reusing complete result: $DELTA_RESULT"
else
    (
        cd "$SQLLM_REPO"
        "$CONDA_BIN" run --no-capture-output -n "$SQLLM_ENV" python llama.py \
            "$MODEL_PATH" wikitext2 \
            --wbits "$BITS" \
            --fake_lut_folder "$DELTA_LUT" \
            --fake_lut_fallback_folder "$FISHER_BASELINE_LUT" \
            --fake_lut_primary_modules v,o,gate,up,down \
            --include_sparse \
            --fake_weight_chunk_rows "$FAKE_WEIGHT_CHUNK_ROWS" \
            --awq_fair_eval \
            --awq_fair_datasets wikitext \
            --awq_fair_batch_size "$PPL_BATCH_SIZE" \
            --ppl_output "$DELTA_RESULT"
    )
fi

echo "[7/7] Writing a direct comparison"
python3 - "$MODEL_TAG" "$BITS" "$FISHER_RESULT" "$DELTA_RESULT" "$COMPARISON_RESULT" <<'PY'
import json
import sys

model, bits, fisher_path, delta_path, output_path = sys.argv[1:]
with open(fisher_path) as handle:
    fisher = json.load(handle)
with open(delta_path) as handle:
    delta = json.load(handle)

fisher_ppl = float(fisher["results"]["wikitext"]["ppl"])
delta_ppl = float(delta["results"]["wikitext"]["ppl"])
fisher_sparse = fisher.get("fake_quant_stats", {}).get("sparse_percentage")
delta_sparse = delta.get("fake_quant_stats", {}).get("sparse_percentage")
comparison = {
    "model": model,
    "bits": int(bits),
    "dataset": "wikitext2",
    "sparse_policy": "magnitude-o0p45 + fisher-sensitivity-0p1",
    "fisher_ds": {
        "ppl": fisher_ppl,
        "sparse_percentage": fisher_sparse,
        "result_file": fisher_path,
    },
    "qk_fisher_rest_fisher_times_inverse_delta": {
        "ppl": delta_ppl,
        "sparse_percentage": delta_sparse,
        "lambda": 1.0,
        "delta": "abs(W-W_awq), inverse transformed",
        "delta_ppl_vs_fisher": delta_ppl - fisher_ppl,
        "result_file": delta_path,
    },
}
with open(output_path, "w") as handle:
    json.dump(comparison, handle, indent=2)

print("\nWikiText2 PPL comparison")
print("  Model / bits : {} / W{}".format(model, bits))
print("  Fisher D+S   : {:.6f}".format(fisher_ppl))
print("  Delta method : {:.6f}".format(delta_ppl))
print("  Delta-Fisher : {:+.6f}  (negative is better)".format(delta_ppl - fisher_ppl))
print("  Fisher sparse: {}%".format(fisher_sparse))
print("  Delta sparse : {}%".format(delta_sparse))
print("  JSON          : {}".format(output_path))
PY

