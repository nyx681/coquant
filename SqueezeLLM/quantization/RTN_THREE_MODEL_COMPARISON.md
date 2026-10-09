# RTN uniform / RTN-delta three-model completion

2026-09-19 run: `/mnt/dfq/rtn_three_model_comparison_20260919_105411`.
Supervisor log: same path with `.log` appended. The supervisor maintains
`comparison.md` and `comparison.json` as cells finish; `jobs.json` records
actual child PIDs, commands, logs and outputs. It marks the suite complete only
after every comparison cell exists. Failed child processes are recorded.

Protocol: original model, W4-G128, WikiText-2 test, AWQ-compatible tokenizer and
2048-token blocks, batch size 8, FP16 model weights and FP32 CE logits.

Reused:

- LLaMA-1 RTN-delta: `/mnt/dfq/llama1_rtn_gptq_nonuniform_20260916/rtn_delta_ppl.json`.
- OPT RTN uniform: `/mnt/dfq/opt67b_w4_g128_rtn_gptq/rtn_ppl.json`.

Missing jobs launched:

- LLaMA-1 RTN uniform and then LLaMA-2 RTN uniform (GPU evaluations sequential).
- LLaMA-2 RTN-delta and OPT RTN-delta (concurrent CPU KMeans pools, 4 workers
  each, alongside the ongoing 24-worker activation fusion run).

Uniform RTN uses the existing asymmetric min/max helper in
`opt_rtn_gptq_fusion.py`: scale=max(max-min,1e-5)/15, zero=clip(-round(min/scale),
0,15), code=clip(round(W/scale)+zero,0,15). Native RTN weights are copied directly
to the model without AWQ scaling or clipping.

RTN-delta exactly retains the prior nonuniform definition. First generate the
FP32 RTN reference in original weight coordinates. Compute module-wide input
channel inverse relative RMS delta (min_weight=.1, gamma=1), multiply
`normalize(AWQ_scale)**p * (1+normalize(inverse_delta))`, with p=3 for attention
output and p=4 for other modules. Then group each original weight row by 128
and fit weighted sklearn KMeans with 16 centers, seed=0, n_init=auto,
max_iter=50. Reassign original weights to nearest centers, as the prior RTN-delta
runner does. This is not the new activation-only fusion objective.

The two new nonuniform models persist FP32 centers for each completed module,
with source/importance fingerprints. Codes can be recovered from original
weights, avoiding multi-gigabyte label files. Combined codebook storage is about
6 GiB on /mnt/dfq; suite launch checks a 7 GiB space budget including reserve.
Original results and codebooks are untouched.

Individual interrupted children can resume with the same command plus
`--resume`. Configuration/source checks guard reuse. A supervisor failure may
require rebuilding the comparison from completed child `ppl.json` files after
the missing child resumes; no numerical result should be guessed for an absent
cell.

Validation before launch: synthetic RTN-delta reconstructed weights matched
the existing RTN/GPTQ dual-nonuniform runner exactly. Native RTN reconstruction
matched the original helper exactly. All existing group selection tests had
already passed for the concurrent activation run.
