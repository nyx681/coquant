#!/usr/bin/env bash
set -Eeuo pipefail

# Model-independent dense W4 AWQ/GPTQ delta fusion search.
# Selection uses two deterministic WikiText2-train calibration splits.
# WikiText2 test is evaluated only after the search has finished.

if [[ $# -ne 1 ]]; then
    echo "Usage: bash $0 {llama1|llama2|opt}" >&2
    exit 2
fi

TARGET="$1"
CONDA_BIN="${CONDA_BIN:-/home/zch/miniconda3/bin/conda}"
SQLLM_REPO="${SQLLM_REPO:-/home/zch/project/Quantification/SqueezeLLM}"
NUM_WORKERS="${NUM_WORKERS:-32}"
SELECT_SAMPLES="${SELECT_SAMPLES:-32}"
GUARD_SAMPLES="${GUARD_SAMPLES:-16}"
CALIB_BATCH_SIZE="${CALIB_BATCH_SIZE:-8}"
PPL_BATCH_SIZE="${PPL_BATCH_SIZE:-8}"
IMPROVEMENT_TOLERANCE="${IMPROVEMENT_TOLERANCE:-0.00001}"

case "$TARGET" in
    llama1)
        MODEL_TAG=llama1-7b
        MODEL_TYPE=llama
        MODEL_PATH=/mnt/dfq/pretrained_model/Llama-7b
        ORIGINAL_CHUNKS=/mnt/dfq/squeezellm_awq_delta/original_chunks
        AWQ_CHUNKS=/mnt/dfq/squeezellm_awq_delta/awq_dequant_chunks
        GPTQ_CHUNKS=/home/zch/project/Quantification/llama1_gptq_delta_ds/w4/gptq_dequant_chunks_g128
        GPTQ_PARAMS=""
        MODULES=(q k v o gate up down)
        ;;
    llama2)
        MODEL_TAG=llama2-7b
        MODEL_TYPE=llama
        MODEL_PATH=/mnt/dfq/pretrained_model/llama-2-7b
        ORIGINAL_CHUNKS=/mnt/dfq/squeezellm_mixed/model_chunks/llama-2-7b
        AWQ_CHUNKS=/mnt/dfq/squeezellm_awq_delta/llama2/awq_dequant_chunks
        GPTQ_CHUNKS=/home/zch/project/Quantification/llama2_w4_dense_joint_delta/gptq_dequant_chunks_g128
        GPTQ_PARAMS=/mnt/dfq/gptq_output/llama-2-7b-w4-g128-gptq_params.pt
        MODULES=(q k v o gate up down)
        ;;
    opt)
        MODEL_TAG=opt-6.7b
        MODEL_TYPE=opt
        MODEL_PATH=/mnt/dfq/pretrained_model/opt-6.7b
        ORIGINAL_CHUNKS=/home/zch/project/Quantification/opt67b_delta_ds/shared/original_chunks
        AWQ_CHUNKS=/home/zch/project/Quantification/opt67b_delta_ds/w4/awq_dequant_chunks
        GPTQ_CHUNKS=/home/zch/project/Quantification/opt67b_w4_dense_delta_confidence/gptq_dequant_chunks_g128
        GPTQ_PARAMS=/mnt/dfq/gptq_output/opt-6.7b-w4-g128-gptq_params.pt
        MODULES=(q k v o up down)
        ;;
    *)
        echo "Unknown target: $TARGET; expected llama1, llama2, or opt" >&2
        exit 2
        ;;
esac

OUTPUT_ROOT="${OUTPUT_ROOT:-/home/zch/project/Quantification/auto_dual_delta/${MODEL_TAG}-w4}"
CANDIDATE_ROOT="$OUTPUT_ROOT/candidates"
CALIB_ROOT="$OUTPUT_ROOT/calibration"
FINAL_LUT="$OUTPUT_ROOT/selected_lut"
TRACE_FILE="$OUTPUT_ROOT/search_trace.tsv"
SUMMARY_FILE="$OUTPUT_ROOT/selection.json"
TEST_RESULT="$OUTPUT_ROOT/wikitext_test.json"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TOKENIZERS_PARALLELISM=false

chunk_count() {
    [[ -d "$1" ]] || { echo 0; return; }
    find "$1" -maxdepth 1 -type f -name 'layer_*.pt' | wc -l
}

score_pair() {
    python3 - "$1" "$2" <<'PY'
import math
import sys
print((math.log(float(sys.argv[1])) + math.log(float(sys.argv[2]))) / 2.0)
PY
}

is_better() {
    python3 - "$1" "$2" "$IMPROVEMENT_TOLERANCE" <<'PY'
import sys
candidate, current, tolerance = map(float, sys.argv[1:])
raise SystemExit(0 if candidate < current - tolerance else 1)
PY
}

is_pair_better() {
    python3 - "$1" "$2" "$3" "$4" "$IMPROVEMENT_TOLERANCE" <<'PY'
import math
import sys
candidate_select, candidate_guard, current_select, current_guard, tolerance = map(
    float, sys.argv[1:]
)
select_gain = math.log(current_select) - math.log(candidate_select)
guard_gain = math.log(current_guard) - math.log(candidate_guard)
raise SystemExit(0 if select_gain > tolerance and guard_gain > tolerance else 1)
PY
}

read_ppl() {
    python3 - "$1" <<'PY'
import json
import sys
with open(sys.argv[1]) as handle:
    print(json.load(handle)["ppl"])
PY
}

for path in "$MODEL_PATH" "$ORIGINAL_CHUNKS" "$AWQ_CHUNKS"; do
    [[ -d "$path" ]] || { echo "Missing directory: $path" >&2; exit 1; }
done
[[ "$(chunk_count "$ORIGINAL_CHUNKS")" -eq 32 ]] || { echo "Incomplete original chunks" >&2; exit 1; }
[[ "$(chunk_count "$AWQ_CHUNKS")" -eq 32 ]] || { echo "Incomplete AWQ chunks" >&2; exit 1; }
mkdir -p "$CANDIDATE_ROOT" "$CALIB_ROOT"

if [[ "$(chunk_count "$GPTQ_CHUNKS")" -ne 32 ]]; then
    [[ -f "$GPTQ_PARAMS" ]] || { echo "Missing GPTQ parameters: $GPTQ_PARAMS" >&2; exit 1; }
    "$CONDA_BIN" run --no-capture-output -n sqllm \
        python "$SQLLM_REPO/quantization/chunk_gptq_params.py" \
        --gptq_params "$GPTQ_PARAMS" \
        --original_chunks "$ORIGINAL_CHUNKS" \
        --output_path "$GPTQ_CHUNKS" \
        --model_type "$MODEL_TYPE" \
        --bit 4 \
        --group_size 128
fi

if [[ -f "$SUMMARY_FILE" && -f "$TEST_RESULT" ]]; then
    echo "Search is already complete: $SUMMARY_FILE"
    python3 - "$SUMMARY_FILE" "$TEST_RESULT" <<'PY'
import json, sys
with open(sys.argv[1]) as f: summary=json.load(f)
with open(sys.argv[2]) as f: result=json.load(f)
print(json.dumps(summary, indent=2))
print("WikiText2 test PPL:", result["results"]["wikitext"]["ppl"])
PY
    exit 0
fi
if [[ -d "$FINAL_LUT" ]]; then
    echo "Partial selected LUT exists without a completed summary: $FINAL_LUT" >&2
    echo "Move that directory aside before restarting the search." >&2
    exit 1
fi

ALPHA_TAGS=(0 0p25 0p5 0p75 1)
ALPHA_VALUES=(0 0.25 0.5 0.75 1)
for index in "${!ALPHA_VALUES[@]}"; do
    alpha="${ALPHA_VALUES[$index]}"
    tag="${ALPHA_TAGS[$index]}"
    folder="$CANDIDATE_ROOT/alpha${tag}"
    echo "Building true elementwise dual-delta LUT: alpha=$alpha"
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
            --awq_delta_weighting dual_elementwise_mix \
            --delta_mix_operator geometric \
            --delta_mix_alpha "$alpha" \
            --awq_delta_gamma 1.0 \
            --awq_delta_min_weight 0.1 \
            --output_folder "$folder" \
            --num_workers "$NUM_WORKERS"
    )
done

calibrate() {
    local primary="$1"
    local fallback="$2"
    local modules="$3"
    local seed="$4"
    local samples="$5"
    local output="$6"
    local -a composition=()
    if [[ -n "$fallback" ]]; then
        composition=(--fallback_folder "$fallback" --primary_modules "$modules")
    fi
    PYTHONPATH="$SQLLM_REPO" \
    "$CONDA_BIN" run --no-capture-output -n awq \
        python "$SQLLM_REPO/quantization/eval_lut_calibration.py" \
        --model "$MODEL_PATH" \
        --lut_folder "$primary" \
        "${composition[@]}" \
        --samples "$samples" \
        --seed "$seed" \
        --batch_size "$CALIB_BATCH_SIZE" \
        --output "$output"
}

echo -e "stage\tmodule\talpha\tselect_ppl\tguard_ppl\tscore\taccepted" > "$TRACE_FILE"

best_endpoint_tag=""
best_score=""
best_select=""
best_guard=""
for tag in 0 1; do
    folder="$CANDIDATE_ROOT/alpha${tag}"
    select_json="$CALIB_ROOT/endpoint_alpha${tag}_select.json"
    guard_json="$CALIB_ROOT/endpoint_alpha${tag}_guard.json"
    calibrate "$folder" "" "" 0 "$SELECT_SAMPLES" "$select_json"
    calibrate "$folder" "" "" 1 "$GUARD_SAMPLES" "$guard_json"
    select_ppl="$(read_ppl "$select_json")"
    guard_ppl="$(read_ppl "$guard_json")"
    score="$(score_pair "$select_ppl" "$guard_ppl")"
    accepted=false
    if [[ -z "$best_score" ]] || is_better "$score" "$best_score"; then
        best_endpoint_tag="$tag"
        best_score="$score"
        best_select="$select_ppl"
        best_guard="$guard_ppl"
        accepted=true
    fi
    echo -e "endpoint\tall\t${tag}\t${select_ppl}\t${guard_ppl}\t${score}\t${accepted}" >> "$TRACE_FILE"
done

echo "Starting greedy module search from alpha=$best_endpoint_tag"
cp -al "$CANDIDATE_ROOT/alpha${best_endpoint_tag}" "$FINAL_LUT"

for module in "${MODULES[@]}"; do
    module_best_tag=""
    module_best_alpha=""
    module_best_score="$best_score"
    module_best_select="$best_select"
    module_best_guard="$best_guard"
    for index in "${!ALPHA_VALUES[@]}"; do
        alpha="${ALPHA_VALUES[$index]}"
        tag="${ALPHA_TAGS[$index]}"
        folder="$CANDIDATE_ROOT/alpha${tag}"
        select_json="$CALIB_ROOT/module_${module}_alpha${tag}_select.json"
        guard_json="$CALIB_ROOT/module_${module}_alpha${tag}_guard.json"
        calibrate "$folder" "$FINAL_LUT" "$module" 0 "$SELECT_SAMPLES" "$select_json"
        calibrate "$folder" "$FINAL_LUT" "$module" 1 "$GUARD_SAMPLES" "$guard_json"
        select_ppl="$(read_ppl "$select_json")"
        guard_ppl="$(read_ppl "$guard_json")"
        score="$(score_pair "$select_ppl" "$guard_ppl")"
        accepted=false
        if is_pair_better \
            "$select_ppl" "$guard_ppl" \
            "$module_best_select" "$module_best_guard"; then
            module_best_tag="$tag"
            module_best_alpha="$alpha"
            module_best_score="$score"
            module_best_select="$select_ppl"
            module_best_guard="$guard_ppl"
        fi
        echo -e "candidate\t${module}\t${alpha}\t${select_ppl}\t${guard_ppl}\t${score}\t${accepted}" >> "$TRACE_FILE"
    done
    if [[ -n "$module_best_tag" ]]; then
        echo "Accepting module=$module from alpha=$module_best_tag"
        "$CONDA_BIN" run --no-capture-output -n sqllm \
            python "$SQLLM_REPO/quantization/override_lut_module.py" \
            --target_folder "$FINAL_LUT" \
            --source_folder "$CANDIDATE_ROOT/alpha${module_best_tag}" \
            --module "$module"
        best_score="$module_best_score"
        best_select="$module_best_select"
        best_guard="$module_best_guard"
        echo -e "accepted\t${module}\t${module_best_alpha}\t${best_select}\t${best_guard}\t${best_score}\ttrue" >> "$TRACE_FILE"
    else
        echo "Keeping current module=$module"
    fi
done

echo "Evaluating the selected LUT once on untouched WikiText2 test"
(
    cd "$SQLLM_REPO"
    PYTHONPATH="$SQLLM_REPO" \
    "$CONDA_BIN" run --no-capture-output -n awq python llama.py \
        "$MODEL_PATH" wikitext2 \
        --wbits 4 \
        --fake_lut_folder "$FINAL_LUT" \
        --fake_weight_chunk_rows 512 \
        --awq_fair_eval \
        --awq_fair_datasets wikitext \
        --awq_fair_batch_size "$PPL_BATCH_SIZE" \
        --ppl_output "$TEST_RESULT"
)

python3 - "$TARGET" "$best_endpoint_tag" "$best_select" "$best_guard" "$best_score" "$TRACE_FILE" "$TEST_RESULT" "$SUMMARY_FILE" <<'PY'
import csv
import json
import sys

target, endpoint, select_ppl, guard_ppl, score, trace_path, test_path, output = sys.argv[1:]
accepted = []
with open(trace_path) as handle:
    for row in csv.DictReader(handle, delimiter="\t"):
        if row["stage"] == "accepted":
            accepted.append({"module": row["module"], "alpha": row["alpha"]})
with open(test_path) as handle:
    test = json.load(handle)
summary = {
    "format": "squeezellm_auto_dual_delta_v1",
    "target": target,
    "selection_dataset": "wikitext2_train",
    "endpoint_alpha": float(endpoint),
    "accepted_module_overrides": accepted,
    "final_select_ppl": float(select_ppl),
    "final_guard_ppl": float(guard_ppl),
    "final_mean_nll_score": float(score),
    "wikitext2_test_ppl": test["results"]["wikitext"]["ppl"],
    "trace_file": trace_path,
}
with open(output, "w") as handle:
    json.dump(summary, handle, indent=2)
print(json.dumps(summary, indent=2))
PY
