# Activation-aware center fusion investigation

Objective: test whether calibration activation reconstruction can produce a useful
combination advantage. No universal superiority is assumed. Current work begins
with the completed OPT-6.7B GPTQ+RTN uniform-endpoint arm, where raw-weight MSE
fusion failed to improve test PPL. Existing AWQ/GPTQ delta jobs remain unchanged.

Evidence from saved OPT metrics: RTN SSE 19445.2820, selected SSE 19440.8108,
refined SSE 14591.2841. Test PPLs respectively 11.154899, 11.155668, 11.086960;
native GPTQ PPL 11.068875. More weight-MSE reduction did not beat GPTQ.

Pilot implementation: pilot_activation_center_fusion.py. This is a diagnostic,
not a full-model quantization or a PPL result. Layers 0,15,31, first 64 output
rows of all six OPT Linear types, preserving G128 and 16 centers/group.
Initialize from (a) native GPTQ centers and native codes, (b) RTN min/max centers
and codes, (c) the saved exact union-selection centers and codes. Optimize all
three with identical budget; do not credit fusion for an optimizer denied to
endpoints. This uniform arm contains no AWQ scale or delta.

Keep assignments fixed, move centers c=c0+std(c0)*offset. Adam lr=.01, 32 steps,
offset limited to [-.5,.5], regularizer 1e-3 mean(offset^2). Fit loss is activation
output SSE normalized by reference output energy plus the regularizer. Thus
input correlations and cross-group errors enter the objective. Evaluate using
FP16-rounded weights. Fixed assignments intentionally preserve native GPTQ codes.
Empty centers can move only through their zero-initialized regularizer gradient.

Data: WikiText train parsed into articles at top-level headings, shuffled with
seed 20260917, duplicates rejected by SHA256; 8 fit, 4 select, 4 guard articles,
first 512 tokens each. Capture every eighth input token at the selected modules.
The same articles/activations/rows are used for all starts. Select among steps
0,8,32 only on select output MSE, then score one chosen checkpoint on guard.
All article IDs/hashes and selection traces are recorded. Guard results are
reported, not used for repeated tuning on the same split. No WikiText test read.

Next stage only if the pilot warrants it: a frozen full-model algorithm with
fresh fit/select/confirmation partitions, independent validation NLL, and equal
endpoint optimization budgets on LLaMA-1, LLaMA-2, OPT. Record endpoint fallback
as lack of demonstrated fusion benefit, never as a fusion win. Existing test
scores are exploratory; do not search to improve already observed test values.

Limitations: local reconstruction may not predict PPL; FP inputs do not reflect
accumulated upstream quantization errors; fixed code assignments limit expressivity;
small article sample and first-64-row pilot cannot establish model-wide benefits.
All-model superiority and statistically meaningful gains remain unproven.

## Initial outcomes and follow-up

Adam pilot: fusion beat equally optimized endpoints on 1/18 guard module slices.
Ridge least-squares pilot (fresh articles excluding first pilot hashes): also
1/18. Neither supports a full-model superiority claim; these are output-MSE
results, not PPL. See /mnt/dfq/opt67b_activation_{fusion,ridge}_pilot_20260917.

Candidate-overlap audit on 55,296 groups (same selected layers/first 64 rows):
54,552 groups have bit-identical GPTQ and RTN 16-level FP32 codebooks. This
explains why union selection offers little candidate diversity in this arm.
It does not establish the same overlap for AWQ/GPTQ nonuniform codebooks.

Third pilot: pilot_activation_index_fusion.py, fresh articles excluding both
previous pilot manifests. Fix centers and use one Gauss-Seidel sweep over codes,
choosing each scalar code to minimize calibration output SSE given other codes.
The fusion starts from native GPTQ weights projected to its union-selected
codebook, retaining GPTQ assignments wherever levels coincide. Endpoint starts
receive the same coordinate sweep. Select original vs updated on select only,
then score guard once. This tests assignments rather than only centroid motion.
Synthetic correlated-objective updates checked to decrease fitting error.
Still a local diagnostic, not a claim of full-model PPL or all-model superiority.

Third pilot outcome: complete, 0/18 guard slices beat both equally optimized
endpoints. Do not repeat these three pilots as evidence of a successful method.

## Follow-up commitment (2026-09-17)

Continue proposing and testing alternatives regardless of the pending LLaMA-2
and OPT AWQ/GPTQ nonuniform results. Preserve all artifacts and reuse compatible
centers, reference weights, and completed evaluations. Do not stop research merely
because fusion's weight objective improved, or claim universal PPL superiority.

1. Whole-group routing (implemented in `pilot_group_routing.py`): retain native
   GPTQ or RTN centers AND codes for each output-row G128. Start separately from
   each endpoint. Flip a group's source only if it reduces total calibration
   output SSE, accounting for the residual from all other groups. Two sweeps;
   select from both starts and sweeps 0/1/2 on select data. Score guard after
   selection. Each final group has one 16-center book, not two summed books.
   The source indicator is only optimization metadata if the chosen book is
   materialized. FP inputs and first-64-output-row sampling remain limitations.
   Use fresh train articles excluding all three previous pilot manifests.
   Report comparisons against native endpoints and code-optimized controls.
   These controls have different search spaces/budgets: a pilot win is a lead,
   not evidence that fusion beats equal-budget optimization or full-model PPL.
2. Block reconstruction (design stage): optimize the final G128 codebooks and
   assignments jointly against FP transformer-block outputs, including nonlinear
   paths. Feed quantized upstream activations to the student. Test endpoint and
   fused initializations with identical optimization budgets. This addresses
   effects not captured by isolated-linear or weight-MSE objectives.
3. Complementary candidate construction (design stage): generate the second
   16-center proposal from the conditional output residual of the current
   quantized solution, rather than repeating the same original-weight fit.
   Union candidates can be selected or refined, but export only 16 centers/group.
   Do not add a second residual codebook at inference or silently increase bits.

Progression: diagnostic -> frozen algorithm -> LLaMA-1/LLaMA-2/OPT full evaluation.
Use independent validation NLL to select algorithm settings, not WikiText test.
Report all endpoint/fusion scores and absolute/relative PPL changes, including
failures. A return to an endpoint is a fallback, not a demonstrated fusion win.
Record storage/precision changes if a block method requires extra parameters.

Group-routing OPT pilot completed in
`/mnt/dfq/opt67b_group_routing_pilot_20260917_230828`: 4/18 held-out module
slices beat both native endpoints; 4/18 beat both code-optimized controls.
These are local output errors, not test PPL or a full-model benefit.

The same pilot supports `--target llama2` (21 slices across layers 0/15/31),
reusing the existing validated static G128 GPTQ artifact. On 2026-09-17 it was
launched as follows, alongside the existing nonuniform experiments:

```bash
PYTHONPATH=.:quantization /home/zch/miniconda3/envs/awq/bin/python -u \
  quantization/pilot_group_routing.py --target llama2 \
  --output /mnt/dfq/llama2_7b_group_routing_pilot_20260917_231519
```

## Nonuniform block and residual pilot (2026-09-18)

Implemented `pilot_nonuniform_block_residual.py` and launched the combined pilot
at `/mnt/dfq/llama2_nonuniform_block_residual_pilot_20260918_084508`.
It tests the actual AWQ-delta/GPTQ-delta nonuniform arms, not uniform GPTQ/RTN.
Scope: LLaMA-2 block 0, first 64 output rows of each of seven Linear modules;
all other rows remain FP16. Scores are partial-block output errors, not PPL.

Reuse AWQ LUT l0 from the completed experiment. The previous RAM-only GPTQ-delta
run saved no centers, so rebuild only these 64 rows using the original sklearn
KMeans seed/weights/settings. Persist the resulting small endpoint books for
reuse. Exact union selection uses the previous shared weighted objective.

Fresh WikiText train articles exclude hashes of all prior pilot manifests.
Fit/select/guard: 8/4/4 articles, 512 tokens each. Capture full block inputs and
teacher outputs, and subsample every fourth Linear input token for residual
fitting. Validate captured block replay matches teacher before proceeding.

Residual-directed generation: for each initial endpoint or fused book Q,
compute R=X(Q-W)^T and conditional unconstrained group correction
`dQ_g^T=-(X_g^T X_g + .01*mean(diag(X_g^T X_g))*I)^-1 X_g^T R`.
Form targets at alpha .25/.5/1, then update the initial 16 centers and assignments
using 30 weighted Lloyd steps with diagonal activation energy weights. Fusion
therefore starts from exact 32-to-16 selection, but updates toward conditional
residual targets instead of original weights. The corrections are simultaneous
(Jacobi), so full fitting SSE is not guaranteed to decrease. Include alpha=0
and choose on select linear output MSE. Do not use guard to choose alpha.
This is a residual-directed generated-codebook hypothesis; it does not yet
construct a new independent 16-center proposal and re-run exact union selection.

Block reconstruction: independently optimize each of original AWQ, GPTQ, fusion
and residual-updated AWQ, GPTQ, fusion starts. All receive 24 Adam steps, lr .02,
fixed codes, trainable center offsets normalized by initial center standard
deviation, clipped to +/-.5. Loss is full block output MSE normalized by initial
fit MSE plus 1e-4 mean squared offsets. Use FP16-rounded weights in forward,
select steps 0/8/24 only on select block MSE, then report guard MSE once for that
selected checkpoint. Save learned books and traces. This tests nonlinear block
interactions, while still quantizing only a small row subset. Upstream error
accumulation is not tested at block 0. No additional inference residual branch.

Synthetic checks passed for weighted Lloyd objective decrease, residual fallback,
16-center output, gradients through rounded LUT weights, and preservation of
untested rows. Real-data outcomes must be read from the run's result files.

Initial real-data run completed successfully. All 21 residual-generation choices
(three starts x seven modules) selected alpha=0; all six block-optimization runs
selected step=0. Neither new update gave a select-set improvement in this pilot.
Guard block MSE remained AWQ 8.38153036e-9, GPTQ 8.47673398e-9, exact fusion
8.21596791e-9. The existing fusion start was already slightly better locally;
this is not a gain produced by either new method, nor a PPL result. These
limited block-0/leading-row tests do not rule out other blocks or optimization
settings, but provide no evidence to claim a successful new fusion method.
