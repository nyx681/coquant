# AWQ/GPTQ-delta: remove the AWQ-scale prior

User-requested controlled ablation, 2026-09-19.

Keep `S_j = 1 + N(d)_j`; remove only `N(a)_j ** p`.

- The reference matrices remain the existing AWQ/GPTQ dequantized weights.
- Module input-channel relative RMS delta, inverse transform (`gamma=1`,
  `min_weight=0.1`), positive-median normalization and delta coefficient 1
  are unchanged. The `+1` is retained.
- Original weights are independently clustered per output row and contiguous
  G128 with 16 centers. The existing sklearn helper is reused unchanged:
  `random_state=0`, `n_init=auto`, `max_iter=50`.
- Neither variant reads an AWQ scale cache. AWQ's existing reference weights
  still reflect the original AWQ procedure; this is an importance-prior
  ablation, not removal of AWQ from the reference-generation procedure.
- Evaluation: the same model/tokenizer, WikiText-2 test token stream,
  nonoverlapping 2048-token blocks, batch size 8, original PPL evaluator.
- Existing with-scale PPL is reused and checked for matching protocol.
  The older whole-row inverse-delta experiment is not a G128 baseline and
  additionally did not use this `1+N(d)` objective.

First target: LLaMA-1-7B. The pair runner supports llama2 and opt as well.

Launch:

```bash
cd /home/zch/project/Quantification/SqueezeLLM
PYTHONPATH=.:quantization /home/zch/miniconda3/envs/awq/bin/python -u \
  quantization/run_no_scale_pair.py --target llama1 \
  --output /mnt/data/zch/quantization_experiments/NEW_UNIQUE_DIRECTORY \
  --workers 4 --boost-after-pid 1492649 --boost-workers 12
```

The two endpoints run concurrently. Initial K-means work uses four CPU
workers per endpoint; once the pre-existing activation job exits, each may
use twelve workers. PPL runs on GPU. Replacing sklearn K-means with a GPU
implementation is deliberately outside this controlled comparison.

Outputs per endpoint: atomic per-module FP32 codebooks (about 3 GiB),
`run.json`, `metrics.json`, reused `baseline_with_scale.json`, final
`ppl.json`, and `comparison.json`. The pair supervisor maintains a combined
`comparison.md/json`, including failed and pending status. A negative
`without_scale - with_scale` difference means removal improved PPL.

Use the child runner with identical target/variant/output and `--resume`
only after that child has stopped. Code hashes and source metadata are
checked, then module fingerprints validate checkpoint reuse. Do not run
two writers against the same output directory.

Validation before launch: real first-module AWQ/GPTQ references each passed
four-group clustering and reconstruction smoke checks. Synthetic checks
verified direct helper equivalence, retention of `+1`, a nonconstant scale
ablation, and finite all-zero-delta behavior.

Current launch record:
`/home/zch/project/Quantification/llama1_no_scale_launch_20260919_150656.json`

Workspace link:
`/home/zch/project/Quantification/llama1_delta_no_awq_scale_20260919_150656`
