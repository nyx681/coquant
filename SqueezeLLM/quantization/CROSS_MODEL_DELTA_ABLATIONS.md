# LLaMA-2-7B / OPT-6.7B delta importance ablations

User request, 2026-09-20: repeat the no-scale and no-additive-one comparisons
on LLaMA-2-7B and OPT-6.7B. Eight new endpoint evaluations:

| Model | Endpoint | Round 1 | Round 2 |
|---|---|---|---|
| LLaMA-2-7B | AWQ-delta | S=1+N(d) | S=N(d) |
| LLaMA-2-7B | GPTQ-delta | S=1+N(d) | S=N(d) |
| OPT-6.7B | AWQ-delta | S=1+N(d) | S=N(d) |
| OPT-6.7B | GPTQ-delta | S=1+N(d) | S=N(d) |

All runs use W4-G128, the existing original/reference chunks, the same
inverse channel-relative delta transform (gamma 1, floor 0.1), identical
sklearn KMeans settings, and WikiText-2 test PPL (2048-token blocks, batch 8).
Round 2 removes only the outer additive one; the internal d floor remains.
Neither round multiplies by AWQ scale. Original with-scale baselines and
completed LLaMA-1 ablations are reused in the combined comparison table.
No fusion is launched by this suite.

Resources: at most four concurrent endpoint processes, eight CPU workers
each. Both models start their first-round AWQ/GPTQ jobs concurrently. A
model's second-round jobs can start only after both of that model's
first-round jobs have completed full PPL evaluation. Center codebooks are
persisted module by module on /mnt/data for reuse and child-level resume.

The existing endpoint runners are reused without edits:
- run_delta_without_awq_scale.py
- run_delta_without_scale_or_offset.py

Supervisor: run_cross_model_delta_ablation_suite.py. It records child PIDs,
commands, logs, failed/blocked status, and writes comparison.json/.csv/.md
every 30 seconds. A failed first-round endpoint prevents that model's second
round from launching; other independent jobs can finish normally.

Preflight: all 32 original and AWQ reference layer chunks are present for
each model. GPTQ parameter shapes and W4-G128 metadata were checked against
all original module shapes. Both endpoint references, under both objectives,
passed four-group clustering/reconstruction checks for each model. Original
chunks are additionally checked against loaded FP-model weights per module
during full execution, and GPTQ exports undergo reconstruction validation.

Current output:
/mnt/data/zch/quantization_experiments/llama2_opt_delta_ablation_20260920_095209

Workspace link:
/home/zch/project/Quantification/llama2_opt_delta_ablation_20260920_095209

Launch record:
/home/zch/project/Quantification/llama2_opt_ablation_launch_20260920_095209.json
