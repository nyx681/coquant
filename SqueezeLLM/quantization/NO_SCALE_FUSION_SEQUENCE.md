# User-directed execution order (2026-09-19)

1. Finish both no-scale single-side experiments, including WikiText-2 PPL:
   AWQ-delta and GPTQ-delta, each using `1 + N(d)` within G128 K-means.
2. Reuse exactly those two endpoint codebooks in both fusion experiments:
   - ordinary SSE: every sample coefficient is 1;
   - activation-weighted SSE: coefficient is calibration mean absolute input
     activation for that input channel, `mean_t(abs(X_tj))`.
3. Report four matching PPLs: AWQ endpoint, GPTQ endpoint, coefficient-1
   fusion, activation fusion. Both fusions perform exact 32-to-16 selection;
   no center-refinement experiment was added to this sequence.

The older with-scale-candidate activation experiment was stopped on user
request; 198 completed module checkpoints and its calibration statistics
remain. The newer coefficient-1 fusion was paused after 4 complete modules,
then scheduled for resume only after **both endpoint runs are complete**.
The two endpoint runs continue and automatically increase to 12 CPU workers
each when the older activation process exits.

Activation statistics are reused from the original FP-model inputs:
`/home/zch/project/Quantification/llama1_activation_nonuniform_g128_20260919_102506/activation_mean_abs.pt`
These statistics do not depend on endpoint codebooks. Reusing them does not
reuse the old activation fusion's with-scale candidate centers or results.
The cache identifies model, statistic, calibration token counts and dataset.
The new fusion records its SHA-256 and validates every module's shape.

Both fusions use the same original weights and no-scale center candidates.
Neither fusion reads AWQ scales or delta-derived importance. Only the
activation version adds the independently collected activation vector.

Controller: `quantization/run_no_scale_fusion_sequence.py`.
It checks both endpoint `status=complete` and matching full PPL metadata
before launching either fusion. The coefficient-1 run resumes existing
checkpoints. The activation version writes a fresh result directory.
If an endpoint fails, the controller records failure and does not launch
fusions. Existing checkpoints/results are never deleted.

Validation: source code hashes for the paused coefficient-1 run match its
resume metadata. Real no-scale endpoint center groups passed activation
selection checks. A synthetic equivalence check verifies activation=1
returns exactly the same centers/objective as ordinary fusion; a varying
activation case verifies the weighted objective against both endpoints.
