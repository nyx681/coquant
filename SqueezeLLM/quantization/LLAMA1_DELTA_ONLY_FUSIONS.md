# LLaMA-1 N(d) endpoint fusions, started 2026-09-20

Two fusion jobs use the completed **S=N(d)** AWQ-delta/GPTQ-delta codebooks.
Neither uses the earlier S=1+N(d) books or the older with-scale books.

Reused endpoint source:
/mnt/data/zch/quantization_experiments/llama1_delta_no_scale_no_offset_20260919_172224

Reused baselines (W4-G128, WikiText-2 test):
- AWQ-delta S=N(d): 5.780082702636719
- GPTQ-delta S=N(d): 5.784216880798340

Fusion objectives on the same original 128 weights and same 32 candidates:
1. Every coefficient is 1: ordinary SSE, exact selection of 16 centers.
2. Coefficients are mean absolute original-model input activations collected
   on calibration data: activation-weighted SSE, exact selection of 16.

The second job reuses the existing original-FP activation cache; it does
not reuse the old activation fusion's candidate centers or quantized output.
No new endpoint KMeans or activation calibration is needed. No center
refinement has been added. Both fusions run concurrently with four CPU
workers each alongside the LLaMA-2/OPT N(d) experiments; PPL is evaluated
on GPU after each full model has been reconstructed.

The prior 4-module coefficient-1 checkpoint belongs to S=1+N(d) source
centers, so it remains preserved and is not used for this changed source.

Current comparison:
/home/zch/project/Quantification/llama1_delta_only_fusions_20260920_100141/comparison.md

Launch record:
/home/zch/project/Quantification/llama1_delta_only_fusions_launch_20260920_100141.json

The controller records both endpoint objective strings and absolute source
paths. It now resumes a coefficient-1 output only if an existing run.json
is present; a fresh output is launched without --resume.
