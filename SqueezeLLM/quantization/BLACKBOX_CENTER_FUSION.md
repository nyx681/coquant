# Black-box AWQ/GPTQ nonuniform center fusion

User-selected candidate sources: the newly started **no-AWQ-scale**
AWQ-delta and GPTQ-delta endpoints, not the older with-scale codebooks.

Endpoint source:
`/mnt/data/zch/quantization_experiments/llama1_delta_no_awq_scale_20260919_150656`

Fusion run:
`/mnt/data/zch/quantization_experiments/llama1_blackbox_no_scale_mse_20260919_153244`

Each endpoint internally clusters original weights with `1 + N(d)`.
The fusion treats its output codebook as a black-box interface: two arrays
of 16 centers for the same output row and the same contiguous 128 columns.
The fusion also receives those 128 original weights.

For union U of the two codebooks, select C with 16 slots to minimize
`sum_j min_{c in C} (W_j - c)**2`, subject to `C` drawn from `U`.
All sample weights are exactly 1. There is no activation, delta, AWQ-scale,
or method-specific importance in this fusion objective. Dividing by 128
yields MSE and does not change the minimizer.

The existing exact one-dimensional selector is reused. It sorts/deduplicates
candidate values and pads repeated slots if fewer than 16 unique values
exist. No center refinement is performed in this experiment. Assign each
original weight to its nearest selected center. The script checks candidate
membership and ordinary SSE no larger than either endpoint for every group.
PPL improvement is not guaranteed by these SSE checks.

Codebook production and fusion overlap: completed source modules can be
consumed immediately. A missing source module waits for its producer; a
failed producer is surfaced. The fusion never regenerates endpoint centers.
Candidate metadata must match model, bit width, group size, and no-scale
variant. Original weights must match the evaluated FP model. Module
fingerprints bind original data and both candidate books for checkpoint reuse.

Validation: a four-candidate/two-center case matches exhaustive enumeration;
four real groups from the no-scale endpoint files passed membership and
ordinary-SSE checks. Full evaluation uses the unchanged LLaMA-1-7B,
W4-G128, WikiText-2 test, block length 2048, batch size 8 protocol.

Outputs: `run.json`, per-module `metrics.json`, reusable `centers/*.npz`,
`ppl.json`, and `comparison.md/json` with the two matching no-scale endpoint
PPLs. If endpoint PPLs are still running after fusion evaluation, the
comparison waits for them; the fusion's own PPL is saved immediately.

Launch command and PID are recorded in:
`/home/zch/project/Quantification/llama1_blackbox_mse_launch_20260919_153244.json`

Resume after a stopped job using the same source mode, endpoint root,
output, and `--resume`. Existing checkpoints are validated and reused.
