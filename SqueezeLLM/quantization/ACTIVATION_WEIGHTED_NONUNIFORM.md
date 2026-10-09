# LLaMA-1 activation-weighted nonuniform center fusion

Run launched 2026-09-19:

- Output: `/home/zch/project/Quantification/llama1_activation_nonuniform_g128_20260919_102506`
- Log: `/mnt/dfq/llama1_activation_nonuniform_g128_20260919_102506.log`
- Launcher record: `/home/zch/project/Quantification/llama1_activation_launch_20260919_102506.json`
- Script: `quantization/run_llama1_activation_fusion.py`

This is a full-model W4-G128 run, not a first-64-row pilot. It evaluates two new
models: exact activation-weighted 32-to-16 selection and selection followed by
up to 50 strictly improving activation-weighted Lloyd updates. WikiText-2 test
uses the existing 2048-token, batch-8 AWQ-compatible evaluator.

## Definition

For each original FP model Linear, collect `alpha[j] = mean_t(abs(input[t,j]))`.
This matches AWQ's `get_act_scale` statistic, but accumulates sums in FP32/FP64
to avoid long FP16 reductions. It is not the saved searched AWQ channel scale.
All variants use the same original-coordinate FP activations. Calibration uses
the local Pile validation data, shuffle seed 42, 512 accepted stripped texts
with 1..512 encoded tokens, concatenated and split into complete 512-token
blocks. This reproduces the local AWQ calibration loader's selection rule;
the number of complete blocks is not necessarily 512. The accepted article
hashes and actual token count are saved in calibration.json.

Reuse existing AWQ-delta nonuniform centers. Previous GPTQ-delta full-model runs
saved only results and RAM weights, so those missing centers are regenerated
with the identical original KMeans function and settings: inverse channel RMS
delta, min_weight=.1, gamma=lambda=1, AWQ-scale power 3 for o / 4 elsewhere,
random_state=0, n_init=auto, max_iter=50. Candidate generation is unchanged.

For each group, optimize `sum_j alpha[j] * min_c (W[j]-c)^2` over 16 candidates
from the AWQ/GPTQ nonuniform union. No AWQ scale factor, delta factor, activation
power, or per-group normalization is applied to this fusion objective.
Refinement uses the same alpha and original W, recomputes nearest assignments,
and updates a center by its assigned samples' alpha-weighted mean. Zero-mass
clusters retain their centers; all-zero-activation groups have objective zero.

The common-objective checks ensure selection does not exceed either source's
nearest-center SSE and refinement does not exceed selection SSE. No PPL
improvement is assumed. Old single-side and old geometric-mean fusion PPLs are
copied as explicitly labeled reused baselines, never presented as new runs.
Activation-refined single-side control PPLs are not part of this two-output run.

## Persistence and resume

Save activation_mean_abs.pt once and one centers/L.M.npz per completed module.
Each checkpoint stores three FP32 books: GPTQ-delta, activation-selected, and
activation-refined, shaped [3, output_rows, groups, 16]. No dense checkpoint or
label array is needed: the labels are deterministically nearest to the original
weights. For groups with zero alpha, this remains the explicit label rule.
Per-module metadata contains the original/importance/source-book fingerprint
and all objective totals. Writes use temporary files followed by rename.
Expected center storage is about 9 GiB on the project volume; space is checked
before center generation. Original artifacts are never removed or overwritten.

From the repository root, after the current process has stopped, resume using:

```bash
PYTHONPATH=.:quantization /home/zch/miniconda3/envs/awq/bin/python -u \
  quantization/run_llama1_activation_fusion.py \
  --output /home/zch/project/Quantification/llama1_activation_nonuniform_g128_20260919_102506 \
  --workers 24 --resume
```

Resume validates configuration, implementation hashes, source metadata, and
module fingerprints. It skips saved activation collection and center generation
for completed modules, reconstructs model weights, and reuses completed PPLs.
An interrupted module is recalculated; completed module books are preserved.
`comparison_ppl.json` appears after both new PPL evaluations succeed.

Validation before launch: existing five group-selection/pipeline tests passed;
synthetic checks passed for activation accumulation, candidate membership,
selection/refinement objective bounds, zero-activation groups, and exact
nearest-center reconstruction into FP16 model weights.
