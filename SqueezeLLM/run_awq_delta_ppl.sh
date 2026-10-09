#!/usr/bin/env bash
set -Eeuo pipefail

# End-to-end Fisher + inverse-AWQ-delta SqueezeLLM WikiText2 experiment.
# Environment variables may override the defaults, for example:
#   CUDA_VISIBLE_DEVICES=1 NUM_WORKERS=48 PPL_DATASETS=wikitext bash run_awq_delta_ppl.sh

CONDA_BIN="${CONDA_BIN:-/home/zch/miniconda3/bin/conda}"
AWQ_ENV="${AWQ_ENV:-awq}"
SQLLM_ENV="${SQLLM_ENV:-sqllm}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
NUM_WORKERS="${NUM_WORKERS:-32}"
PPL_DATASETS="${PPL_DATASETS:-wikitext}"
PPL_BATCH_SIZE="${PPL_BATCH_SIZE:-8}"
FAKE_WEIGHT_CHUNK_ROWS="${FAKE_WEIGHT_CHUNK_ROWS:-512}"

ORIGINAL_MODEL="${ORIGINAL_MODEL:-/mnt/dfq/pretrained_model/Llama-7b}"
AWQ_REPO="${AWQ_REPO:-/home/zch/project/awq_code/llm-awq_zch0810mixed-precision}"
AWQ_RESULTS="${AWQ_RESULTS:-/home/zch/project/llm-awq514/awq_cache/llama-7b-w4-g128-awq.pt}"
SQLLM_REPO="${SQLLM_REPO:-/home/zch/project/Quantification/SqueezeLLM}"
WORK_ROOT="${WORK_ROOT:-/mnt/dfq/squeezellm_awq_delta}"
GRADIENT_CHUNKS="${GRADIENT_CHUNKS:-/mnt/dfq/squeezellm_mixed/gradient_chunks/Llama-7b}"
OUTLIER_CONFIG="${OUTLIER_CONFIG:-/mnt/dfq/squeezellm_fake/lut/Llama-7b-squeezellm-s0p1-o0p45/outlier_config.json}"
FISHER_BASELINE_LUT="${FISHER_BASELINE_LUT:-/mnt/dfq/squeezellm_fake/lut/Llama-7b-squeezellm-s0p1-o0p45}"

AWQ_DEQUANT_MODEL="${WORK_ROOT}/Llama-7b-awq-dequant-w4-g128"
ORIGINAL_CHUNKS="${WORK_ROOT}/original_chunks"
AWQ_DEQUANT_CHUNKS="${WORK_ROOT}/awq_dequant_chunks"
LUT_FOLDER="${WORK_ROOT}/lut_w4_fisher09_inverse_delta01_s0p1_o0p45"
PPL_OUTPUT="${WORK_ROOT}/ppl_w4_fisher09_inverse_delta01_no_qk_s0p1_o0p45_wikitext.json"
EXPECTED_LAYERS=32

export CUDA_VISIBLE_DEVICES
export TOKENIZERS_PARALLELISM=false

require_file() {
    if [[ ! -f "$1" ]]; then
        echo "Required file not found: $1" >&2
        exit 1
    fi
}

require_dir() {
    if [[ ! -d "$1" ]]; then
        echo "Required directory not found: $1" >&2
        exit 1
    fi
}

chunk_count() {
    local chunk_dir="$1"
    if [[ ! -d "$chunk_dir" ]]; then
        echo 0
        return
    fi
    find "$chunk_dir" -maxdepth 1 -type f -name 'layer_*.pt' | wc -l
}

awq_checkpoint_complete() {
    [[ -f "${AWQ_DEQUANT_MODEL}/config.json" ]] && \
    { [[ -f "${AWQ_DEQUANT_MODEL}/model.safetensors" ]] || \
      [[ -f "${AWQ_DEQUANT_MODEL}/model.safetensors.index.json" ]] || \
      [[ -f "${AWQ_DEQUANT_MODEL}/pytorch_model.bin" ]] || \
      [[ -f "${AWQ_DEQUANT_MODEL}/pytorch_model.bin.index.json" ]]; }
}

require_file "$CONDA_BIN"
require_dir "$ORIGINAL_MODEL"
require_dir "$AWQ_REPO"
require_dir "$SQLLM_REPO"
require_file "$AWQ_RESULTS"
require_dir "$GRADIENT_CHUNKS"
require_file "$OUTLIER_CONFIG"
require_dir "$FISHER_BASELINE_LUT"
mkdir -p "$WORK_ROOT"

# Fail early if a packed/real-quantized state dict was accidentally supplied
# instead of the small AWQ search artifact containing scale and clip records.
"$CONDA_BIN" run --no-capture-output -n "$AWQ_ENV" \
    python -c '
import sys
import torch

path = sys.argv[1]
artifact = torch.load(path, map_location="cpu")
if not isinstance(artifact, dict) or "scale" not in artifact or "clip" not in artifact:
    raise SystemExit(
        "Invalid AWQ_RESULTS: expected an AWQ search artifact containing "
        f"scale/clip, got {path}"
    )
print("Validated AWQ search artifact: {}; scale records={}".format(path, len(artifact["scale"])))
' "$AWQ_RESULTS"

echo "[1/5] Exporting the AWQ fake-quantized/dequantized W1 checkpoint"
if awq_checkpoint_complete; then
    echo "AWQ checkpoint already exists; skipping: ${AWQ_DEQUANT_MODEL}"
else
    (
        cd "$AWQ_REPO"
        "$CONDA_BIN" run --no-capture-output -n "$AWQ_ENV" \
            python -m awq.entry \
            --model_path "$ORIGINAL_MODEL" \
            --w_bit 4 \
            --q_group_size 128 \
            --q_backend fake \
            --mix_sources awq \
            --load_awq "$AWQ_RESULTS" \
            --dump_fake "$AWQ_DEQUANT_MODEL"
    )
fi

echo "[2/5] Chunking original FP weights"
if [[ "$(chunk_count "$ORIGINAL_CHUNKS")" -eq "$EXPECTED_LAYERS" ]]; then
    echo "Original chunks already complete; skipping: ${ORIGINAL_CHUNKS}"
else
    (
        cd "${SQLLM_REPO}/quantization"
        "$CONDA_BIN" run --no-capture-output -n "$SQLLM_ENV" \
            python chunk_models.py \
            --model "$ORIGINAL_MODEL" \
            --model_type llama \
            --output_path "$ORIGINAL_CHUNKS"
    )
fi

echo "[3/5] Chunking AWQ-dequantized W1 weights"
if [[ "$(chunk_count "$AWQ_DEQUANT_CHUNKS")" -eq "$EXPECTED_LAYERS" ]]; then
    echo "AWQ chunks already complete; skipping: ${AWQ_DEQUANT_CHUNKS}"
else
    (
        cd "${SQLLM_REPO}/quantization"
        "$CONDA_BIN" run --no-capture-output -n "$SQLLM_ENV" \
            python chunk_models.py \
            --model "$AWQ_DEQUANT_MODEL" \
            --model_type llama \
            --output_path "$AWQ_DEQUANT_CHUNKS"
    )
fi

echo "[4/5] Building the 4-bit 90% Fisher + 10% inverse-delta LUT"
(
    cd "${SQLLM_REPO}/quantization"
    "$CONDA_BIN" run --no-capture-output -n "$SQLLM_ENV" \
        python nuq.py \
        --bit 4 \
        --model_type llama \
        --model "$ORIGINAL_CHUNKS" \
        --gradient "$GRADIENT_CHUNKS" \
        --awq_model "$AWQ_DEQUANT_CHUNKS" \
        --awq_delta_weighting inverse \
        --awq_delta_gamma 1.0 \
        --awq_delta_min_weight 0.01 \
        --awq_delta_fisher_blend 0.1 \
        --output_folder "$LUT_FOLDER" \
        --outlier_config "$OUTLIER_CONFIG" \
        --sensitivity 0.1 \
        --num_workers "$NUM_WORKERS"
)

echo "[5/5] Expanding the LUT as fake weights and evaluating PPL"
(
    cd "$SQLLM_REPO"
    "$CONDA_BIN" run --no-capture-output -n "$SQLLM_ENV" \
        python llama.py \
        "$ORIGINAL_MODEL" wikitext2 \
        --wbits 4 \
        --fake_lut_folder "$LUT_FOLDER" \
        --fake_lut_fallback_folder "$FISHER_BASELINE_LUT" \
        --fake_lut_primary_modules v,o,gate,up,down \
        --include_sparse \
        --fake_weight_chunk_rows "$FAKE_WEIGHT_CHUNK_ROWS" \
        --awq_fair_eval \
        --awq_fair_datasets "$PPL_DATASETS" \
        --awq_fair_batch_size "$PPL_BATCH_SIZE" \
        --ppl_output "$PPL_OUTPUT"
)

echo "Completed. PPL results: ${PPL_OUTPUT}"
