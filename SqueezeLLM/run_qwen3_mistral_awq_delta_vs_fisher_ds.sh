#!/usr/bin/env bash
set -Eeuo pipefail

# WikiText-2 comparison for LLaMA-layout models:
#   - original Fisher Dense-and-Sparse
#   - q/k Fisher + Fisher_norm*(1+inverse_AWQ_delta_norm) elsewhere
#
# Usage:
#   bash run_qwen3_mistral_awq_delta_vs_fisher_ds.sh qwen3 3
#   bash run_qwen3_mistral_awq_delta_vs_fisher_ds.sh mistral 4

if [[ $# -ne 2 ]]; then
    echo "Usage: bash $0 {qwen3|mistral} {3|4}" >&2
    exit 2
fi

MODEL_FAMILY="$1"
BITS="$2"
if [[ "$BITS" != "3" && "$BITS" != "4" ]]; then
    echo "BITS must be 3 or 4; got: $BITS" >&2
    exit 2
fi

CONDA_BIN="${CONDA_BIN:-/home/zch/miniconda3/bin/conda}"
AWQ_ENV="${AWQ_ENV:-awq}"
SQLLM_ENV="${SQLLM_ENV:-sqllm}"
AWQ_REPO="${AWQ_REPO:-/home/zch/project/awq_code/llm-awq_zch0722}"
SQLLM_REPO="${SQLLM_REPO:-/home/zch/project/Quantification/SqueezeLLM}"
GRAD_REPO="${GRAD_REPO:-/home/zch/project/SqueezeLLM-gradients}"
HOME_WORK="${HOME_WORK:-/home/zch/project/Quantification/qwen3_mistral_awq_delta}"
MNT_WORK="${MNT_WORK:-/mnt/dfq/qwen3_mistral_awq_delta}"
NUM_WORKERS="${NUM_WORKERS:-24}"
FISHER_EXAMPLES="${FISHER_EXAMPLES:-100}"
PPL_BATCH_SIZE="${PPL_BATCH_SIZE:-1}"
OUTLIER_PERCENTAGE="${OUTLIER_PERCENTAGE:-0.45}"
SENSITIVITY="${SENSITIVITY:-0.1}"

case "$MODEL_FAMILY" in
    qwen3)
        MODEL_TAG="qwen3-8b-base"
        MODEL_PATH="${MODEL_PATH:-/mnt/dfq/pretrained_model/Qwen3-8B-Base}"
        AWQ_RESULTS="${AWQ_RESULTS:-${AWQ_REPO}/awq_cache/qwen3-8b-w${BITS}-g128.pt}"
        EXPECTED_LAYERS=36
        ;;
    mistral)
        MODEL_TAG="mistral-7b-v0.1"
        MODEL_PATH="${MODEL_PATH:-/mnt/dfq/pretrained_model/Mistral-7B-v0.1}"
        AWQ_RESULTS="${AWQ_RESULTS:-${AWQ_REPO}/awq_cache/mistral-7b-w${BITS}-g128.pt}"
        EXPECTED_LAYERS=32
        ;;
    *)
        echo "MODEL_FAMILY must be qwen3 or mistral; got: $MODEL_FAMILY" >&2
        exit 2
        ;;
esac

SHARED_MNT="${MNT_WORK}/${MODEL_TAG}/shared"
SHARED_HOME="${HOME_WORK}/${MODEL_TAG}/shared"
ORIGINAL_CHUNKS="${SHARED_MNT}/original_chunks"
GRADIENT_CHUNKS="${SHARED_HOME}/gradient_chunks"
OUTLIER_DIR="${SHARED_MNT}/outlier_config_o${OUTLIER_PERCENTAGE}"
RUN_MNT="${MNT_WORK}/${MODEL_TAG}/w${BITS}"
RUN_HOME="${HOME_WORK}/${MODEL_TAG}/w${BITS}"
AWQ_FAKE_MODEL="${RUN_MNT}/awq_fake_model"
AWQ_CHUNKS="${RUN_HOME}/awq_dequant_chunks"
FISHER_LUT="${RUN_MNT}/fisher_ds"
DELTA_LUT="${RUN_MNT}/fisher_mod_inverse_delta_lambda1_qk_fisher_ds"
RESULT_DIR="${RUN_HOME}/results"
FISHER_RESULT="${RESULT_DIR}/fisher_ds_w${BITS}_wikitext.json"
DELTA_RESULT="${RESULT_DIR}/delta_mod_lambda1_qk_fisher_ds_w${BITS}_wikitext.json"
COMPARISON_RESULT="${RESULT_DIR}/comparison_w${BITS}_wikitext.json"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

fail() { echo "ERROR: $*" >&2; exit 1; }
chunk_count() {
    [[ -d "$1" ]] || { echo 0; return; }
    find "$1" -maxdepth 1 -type f -name 'layer_*.pt' | wc -l
}
lut_complete() {
    [[ "$(find "$1/lut" -maxdepth 1 -type f -name 'l*.pkl' 2>/dev/null | wc -l)" -eq "$EXPECTED_LAYERS" ]] &&
    [[ "$(find "$1/outliers" -maxdepth 1 -type f -name 'l*.pkl' 2>/dev/null | wc -l)" -eq "$EXPECTED_LAYERS" ]]
}
result_complete() {
    [[ -f "$1" ]] && python3 - "$1" <<'PY' >/dev/null 2>&1
import json, sys
with open(sys.argv[1]) as f:
    value = json.load(f)
assert value.get("status") == "complete"
assert value["results"]["wikitext"]["ppl"] > 0
PY
}

[[ -x "$CONDA_BIN" ]] || fail "Missing conda: $CONDA_BIN"
[[ -d "$MODEL_PATH" ]] || fail "Missing model: $MODEL_PATH"
[[ -f "$AWQ_RESULTS" ]] || fail "Missing AWQ search result: $AWQ_RESULTS"
mkdir -p "$SHARED_MNT" "$SHARED_HOME" "$RUN_MNT" "$RUN_HOME" "$RESULT_DIR"

echo "============================================================"
echo "Model:               $MODEL_TAG"
echo "Bits:                W$BITS"
echo "Fisher calibration:  C4, $FISHER_EXAMPLES samples"
echo "Sparse policy:       magnitude ${OUTLIER_PERCENTAGE}% + Fisher ${SENSITIVITY}%"
echo "Delta policy:        q/k Fisher; v/o/gate/up/down Fisher*(1+inverse-delta)"
echo "Evaluation:          WikiText-2 only"
echo "============================================================"

echo "[1/8] Original model chunks"
if [[ "$(chunk_count "$ORIGINAL_CHUNKS")" -ne "$EXPECTED_LAYERS" ]]; then
    PYTHONPATH="$SQLLM_REPO" "$CONDA_BIN" run --no-capture-output -n "$AWQ_ENV" \
        python "${SQLLM_REPO}/quantization/chunk_models.py" \
        --model "$MODEL_PATH" --model_type llama --output_path "$ORIGINAL_CHUNKS"
fi

echo "[2/8] Fisher gradient chunks"
if [[ "$(chunk_count "$GRADIENT_CHUNKS")" -ne "$EXPECTED_LAYERS" ]]; then
    (
        cd /tmp
        "$CONDA_BIN" run --no-capture-output -n "$AWQ_ENV" \
            python "${GRAD_REPO}/run.py" \
            --model_name_or_path "$MODEL_PATH" \
            --dataset c4 \
            --num_examples "$FISHER_EXAMPLES" \
            --output_dir "${RUN_HOME}/unused_gradient_hf" \
            --gradient_chunk_output "$GRADIENT_CHUNKS" \
            --save_safetensors true
    )
fi

echo "[3/8] Shared magnitude-outlier configuration"
if ! compgen -G "${OUTLIER_DIR}/outlier_config_o*.json" >/dev/null; then
    (
        cd "${SQLLM_REPO}/quantization"
        "$CONDA_BIN" run --no-capture-output -n "$SQLLM_ENV" python generate_outlier_config.py \
            --model "$ORIGINAL_CHUNKS" \
            --model_type llama \
            --target_percentage "$OUTLIER_PERCENTAGE" \
            --output_folder "$OUTLIER_DIR"
    )
fi
mapfile -t OUTLIER_CONFIGS < <(find "$OUTLIER_DIR" -maxdepth 1 -type f -name 'outlier_config_o*.json' | sort)
[[ "${#OUTLIER_CONFIGS[@]}" -eq 1 ]] || fail "Expected one outlier config in $OUTLIER_DIR"
OUTLIER_CONFIG="${OUTLIER_CONFIGS[0]}"

echo "[4/8] Fisher Dense-and-Sparse LUT"
if ! lut_complete "$FISHER_LUT"; then
    (
        cd "${SQLLM_REPO}/quantization"
        "$CONDA_BIN" run --no-capture-output -n "$SQLLM_ENV" python nuq.py \
            --bit "$BITS" --model_type llama \
            --model "$ORIGINAL_CHUNKS" \
            --gradient "$GRADIENT_CHUNKS" \
            --output_folder "$FISHER_LUT" \
            --outlier_config "$OUTLIER_CONFIG" \
            --sensitivity "$SENSITIVITY" \
            --num_workers "$NUM_WORKERS"
    )
fi

echo "[5/8] Bit-matched AWQ-dequantized chunks"
if [[ "$(chunk_count "$AWQ_CHUNKS")" -ne "$EXPECTED_LAYERS" ]]; then
    if [[ ! -f "${AWQ_FAKE_MODEL}/model.safetensors.index.json" && ! -f "${AWQ_FAKE_MODEL}/model.safetensors" ]]; then
        (
            cd "$AWQ_REPO"
            "$CONDA_BIN" run --no-capture-output -n "$AWQ_ENV" python -m awq.entry \
                --model_path "$MODEL_PATH" \
                --w_bit "$BITS" --q_group_size 128 --q_backend fake \
                --mix_sources awq --load_awq "$AWQ_RESULTS" \
                --dump_fake "$AWQ_FAKE_MODEL"
        )
    fi
    PYTHONPATH="$SQLLM_REPO" "$CONDA_BIN" run --no-capture-output -n "$AWQ_ENV" \
        python "${SQLLM_REPO}/quantization/chunk_models.py" \
        --model "$AWQ_FAKE_MODEL" --model_type llama --output_path "$AWQ_CHUNKS"
fi
if [[ "$(chunk_count "$AWQ_CHUNKS")" -eq "$EXPECTED_LAYERS" && -d "$AWQ_FAKE_MODEL" ]]; then
    rm -rf -- "$AWQ_FAKE_MODEL"
fi

echo "[6/8] Delta-modulated Dense-and-Sparse LUT"
if ! lut_complete "$DELTA_LUT"; then
    (
        cd "${SQLLM_REPO}/quantization"
        "$CONDA_BIN" run --no-capture-output -n "$SQLLM_ENV" python nuq.py \
            --bit "$BITS" --model_type llama \
            --model "$ORIGINAL_CHUNKS" \
            --gradient "$GRADIENT_CHUNKS" \
            --awq_model "$AWQ_CHUNKS" \
            --awq_delta_weighting inverse \
            --awq_delta_gamma 1.0 \
            --awq_delta_min_weight 0.01 \
            --awq_delta_fisher_blend 1.0 \
            --awq_delta_fisher_combine multiplicative \
            --kmeans_modules v,o,gate,up,down \
            --output_folder "$DELTA_LUT" \
            --outlier_config "$OUTLIER_CONFIG" \
            --sensitivity "$SENSITIVITY" \
            --num_workers "$NUM_WORKERS"
    )
fi

echo "[7/8] WikiText-2 PPL"
if ! result_complete "$FISHER_RESULT"; then
    PYTHONPATH="$SQLLM_REPO" "$CONDA_BIN" run --no-capture-output -n "$AWQ_ENV" \
        python "${SQLLM_REPO}/llama.py" "$MODEL_PATH" wikitext2 \
        --wbits "$BITS" --fake_lut_folder "$FISHER_LUT" --include_sparse \
        --fake_weight_chunk_rows 512 --awq_fair_eval \
        --awq_fair_datasets wikitext --awq_fair_batch_size "$PPL_BATCH_SIZE" \
        --ppl_output "$FISHER_RESULT"
fi
if ! result_complete "$DELTA_RESULT"; then
    PYTHONPATH="$SQLLM_REPO" "$CONDA_BIN" run --no-capture-output -n "$AWQ_ENV" \
        python "${SQLLM_REPO}/llama.py" "$MODEL_PATH" wikitext2 \
        --wbits "$BITS" --fake_lut_folder "$DELTA_LUT" \
        --fake_lut_fallback_folder "$FISHER_LUT" \
        --fake_lut_primary_modules v,o,gate,up,down --include_sparse \
        --fake_weight_chunk_rows 512 --awq_fair_eval \
        --awq_fair_datasets wikitext --awq_fair_batch_size "$PPL_BATCH_SIZE" \
        --ppl_output "$DELTA_RESULT"
fi

echo "[8/8] Comparison JSON"
python3 - "$MODEL_TAG" "$BITS" "$FISHER_RESULT" "$DELTA_RESULT" "$COMPARISON_RESULT" <<'PY'
import json, sys
model, bits, fisher_path, delta_path, output = sys.argv[1:]
with open(fisher_path) as f:
    fisher = json.load(f)
with open(delta_path) as f:
    delta = json.load(f)
fisher_ppl = float(fisher["results"]["wikitext"]["ppl"])
delta_ppl = float(delta["results"]["wikitext"]["ppl"])
value = {
    "model": model,
    "bits": int(bits),
    "dataset": "wikitext2",
    "sparse_policy": "magnitude-o0p45 + fisher-sensitivity-0p1",
    "fisher_ds": {
        "ppl": fisher_ppl,
        "sparse_percentage": fisher["fake_quant_stats"]["sparse_percentage"],
        "result_file": fisher_path,
    },
    "qk_fisher_rest_fisher_times_inverse_awq_delta": {
        "ppl": delta_ppl,
        "sparse_percentage": delta["fake_quant_stats"]["sparse_percentage"],
        "lambda": 1.0,
        "delta_ppl_vs_fisher": delta_ppl - fisher_ppl,
        "relative_change_percentage": (delta_ppl - fisher_ppl) / fisher_ppl * 100,
        "result_file": delta_path,
    },
}
with open(output, "w") as f:
    json.dump(value, f, indent=2)
print(json.dumps(value, indent=2))
PY

