#!/usr/bin/env bash
set -Eeuo pipefail
SQLLM_REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_ROOT="${OUTPUT_ROOT:-/home/zch/project/Quantification/llama1_w4_g128_center_fusion/full_$(date +%Y%m%d_%H%M%S)}"
export PYTHONPATH="$SQLLM_REPO${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
exec /home/zch/miniconda3/envs/sqllm/bin/python -u "$SQLLM_REPO/quantization/fuse_group_centers.py" \
  --original_chunks /mnt/dfq/squeezellm_awq_delta/original_chunks \
  --awq_chunks /mnt/dfq/squeezellm_awq_delta/awq_dequant_chunks \
  --gptq_params /mnt/dfq/gptq_output/llama-7b-w4-g128-gptq_params-fixed.pt \
  --awq_scale_cache /home/zch/project/awq_code/llm-awq_zch0722/awq_cache/llama-7b-w4-g128.pt \
  --output "$OUTPUT_ROOT" --workers "${NUM_WORKERS:-32}" "$@"
