# Current cross-model plan: run N(d) directly

User correction, 2026-09-20: do not run the keep-one (`1+N(d)`) ablation
on LLaMA-2-7B/OPT for now. Directly run no-scale/no-offset (`N(d)`).

Stopped the prior suite controller and its four keep-one process groups.
Their complete module checkpoints and partial progress remain in
`llama2_opt_delta_ablation_20260920_095209` with `paused_by_user` status.
No files were deleted, and none of the keep-one codebooks are reused as
N(d) codebooks because the objectives differ.

New suite has exactly four jobs, with no prerequisite on the stopped jobs:
- LLaMA-2-7B, AWQ-delta, S=N(d)
- LLaMA-2-7B, GPTQ-delta, S=N(d)
- OPT-6.7B, AWQ-delta, S=N(d)
- OPT-6.7B, GPTQ-delta, S=N(d)

All four run concurrently with eight CPU KMeans workers each; GPU is used
for final PPL. Models, reference matrices, delta transform, G128 groups,
16 centers, sklearn settings and WikiText-2 test evaluation are unchanged.
Existing with-scale baselines and completed LLaMA-1 results are reused.

Controller: run_cross_model_delta_only_suite.py
Endpoint runner (unchanged): run_delta_without_scale_or_offset.py

Current output:
/mnt/data/zch/quantization_experiments/llama2_opt_delta_only_20260920_095736

Workspace link:
/home/zch/project/Quantification/llama2_opt_delta_only_20260920_095736

The live comparison table retains the paused keep-one column and clearly
labels those cells as incomplete. Final PPLs fill the N(d) column. Fusion
remains unscheduled.
