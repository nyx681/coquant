# G128 AWQ-delta / GPTQ center fusion

This experiment clusters ORIGINAL weights in contiguous groups of 128. AWQ
supplies a nonuniform codebook from AWQ-delta weighted K-means. GPTQ supplies
its native 16 uniform levels from the existing W4-G128 scale/zero parameters;
there is no GPTQ-delta K-means stage.

For a module, compute the existing channel weights, before splitting groups:

    r[j] = RMS_i(abs(W-W_awq)) / max(RMS_i(W), float32_eps)
    tau = median(positive r)
    d[j] = 0.1 + 0.9 / (1 + r[j]/(tau + float32_eps))
    normalize(x) = x / (median(positive x) + float32_eps)
    S[j] = normalize(AWQ_scale[j])**p * (1 + normalize(d[j]))

All-zero relative errors produce d=1. All sample weights must be finite and
positive. p=3 for o, p=4 elsewhere; gamma=lambda=1. This is an EXPLORATORY
configuration inherited from earlier test-informed experiments. It includes
AWQ scale and channel-aggregated AWQ delta, not true elementwise delta weights.
GPTQ participates through its candidate centers, not an extra delta term.
There is no module routing or new validation/test-driven parameter selection.

For each group, form the union of the AWQ-delta K-means centers and the native
GPTQ levels scale*(arange(16)-zero). Sort/deduplicate, then exactly minimize
sum_j S[j]*min_c (W[j]-c)^2 over 16 chosen candidates. The DP partitions the
real line between consecutive selected candidates and includes both tails.
All costs are float64; selected centers remain original FP32 candidate values.
Ties are deterministic. Fewer than 16 distinct candidates are padded by repeats.

Four LUT folders are written:

- awq: group K-means, sklearn random_state=0, n_init=auto, max_iter=50.
- gptq_nearest: native GPTQ centers, NEW nearest-center assignments of original W.
- selected: exact restricted selection; no center movement.
- refined: selected initialization followed by up to 50 weighted Lloyd updates;
  only strictly improving FP32-center updates are accepted, empty centers retained.

metrics.json also records gptq_native error against exported GPTQ weights. This
is distinct from gptq_nearest: GPTQ's error-compensated assignments need not be
nearest to the original weights. Both SSE and ordinary MSE are recorded. The
non-increase assertion is on the common weighted objective per group, not on
PPL, and not on each method's original optimization objective. This objective
uses the AWQ prior and is not a symmetric comparison of method-specific losses.

## Running

From the repository root:

    bash run_llama1_w4_g128_center_fusion.sh

Default output uses a fresh timestamp. Set OUTPUT_ROOT and NUM_WORKERS to
change the destination and CPU process count. Existing destinations are refused;
there is no automatic deletion or resume/overwrite. Per-layer files are written
atomically. An interrupted run is incomplete and must not be evaluated as a full
model. Four uncompressed variants occupy about 6 bytes per quantized weight
(total roughly 37 GiB for LLaMA-7B). The script checks free space with a reserve.

A diagnostic example:

    OUTPUT_ROOT=/absolute/new/path NUM_WORKERS=4 bash run_llama1_w4_g128_center_fusion.sh --layers 0 --row_limit 4

row_limit results contain partial rows and are NOT full-model evaluation inputs.
The builder supports the supplied LLaMA layout. Dynamic noncontiguous GPTQ group
mappings are rejected rather than silently interpreted as contiguous G128.
Static mapping and qparam reconstruction are checked against native weights,
allowing at most one dtype ULP plus the dtype tiny value for export rounding.

## Format / evaluation

Each module in lut/lN.pkl is a dictionary:

    format: squeezellm_group_lut_v1
    group_size: 128
    centers: float32 [out_features, in_features/128, 16]
    labels: uint8 [out_features, in_features/128, 128]

squeezellm/fake_quant.py expands this format and still accepts legacy row LUTs.
Only float simulation is supported; do NOT feed this format to pack.py or the
existing packed CUDA kernels. Index storage in these intermediate files is
uint8, not packed four-bit. At four-bit packed indices and FP32 codebooks, the
codebook overhead is 4 bits/weight, i.e. 8 total bits/weight before other metadata.
All group-LUT baselines must use the same group size and center precision;
native GPTQ has different codebook overhead and should be reported separately.

No test PPL is automatically run. Freeze the comparison and use independently
partitioned validation data before final test evaluation. Existing calibration
scripts sample random train windows and do not establish independent holdouts.

## Verification

    PYTHONPATH=. /home/zch/miniconda3/envs/sqllm/bin/python -m unittest discover -s tests -v

Tests compare DP to exhaustive search, cover ties/duplicate centers/zero weights,
reject invalid mappings and indices, and verify grouped/legacy expansion and
end-to-end objectives through stored FP32 centers and uint8 labels.

## Three-PPL comparison requested after generation started

`eval_group_center_three.py --root FULL_RUN --generator_pid PID` evaluates
`gptq_native` first, then waits for complete, nonpartial LUTs and evaluates `awq`
and `selected`. Each runs in a fresh process using the same existing AWQ-fair
WikiText-2 test tokenizer, 2048-token blocks, batch size 8, FP16 model, and GPU.
GPTQ native weights are copied directly from exported records, preserving their
original assignments; gptq_nearest and refined are not part of the requested
three-way PPL comparison. Results are under FULL_RUN/ppl, with three_ppl.json
written only once all three succeed. Existing result files are refused. This is
an explicitly requested exploratory test evaluation; it does not create a new
untouched holdout or retroactively validate the inherited power settings.
