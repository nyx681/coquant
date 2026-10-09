# Additional additive-offset ablation

Latest user instruction (2026-09-19): finish the no-scale AWQ/GPTQ
nonuniform experiments, then remove the `1+` term as well, inspect results,
and only then arrange fusion.

Execution order:

1. Finish `S = 1 + N(d)` AWQ-delta and GPTQ-delta, including both full PPLs.
2. Run `S = N(d)` AWQ-delta and GPTQ-delta as separate outputs.
3. Compare the two rounds against each other and the reused original
   with-scale baselines. Fusion remains deferred until this comparison
   determines which endpoint codebooks to use.

Both rounds omit the AWQ-scale multiplier. The inverse-delta transform is
unchanged: channel-relative RMS error, gamma=1, min_weight=0.1, module-wide
positive-median normalization. In particular, removing `1+` does not remove
the 0.1 floor inside d, change delta into a direct error weight, or make
all importance weights equal to one. Original references and original
weights are reused, and the G128 KMeans and full PPL protocol are identical.

The earlier fusion scheduler PID 1884627 was stopped before it launched
any children. The old activation fusion and the four-module coefficient-1
fusion checkpoint remain preserved. The first-round endpoint processes
were not stopped or restarted.

Second-round runner: `run_delta_without_scale_or_offset.py`, kept separate
from the active first-round source file to preserve its resume hashes.
Controller: `run_delta_offset_ablation_sequence.py`. It verifies both
first-round PPL artifacts and their objective before starting round two.
No fusion runner is launched by this controller.

The comparison table has these columns:
`method | original with scale | 1+N(d) | N(d) | second-round status`.

Validation checks verify exact `N(d)` helper output, the first-round
identity `S_first = 1 + S_second`, finite zero-error handling, and unchanged
KMeans output for the supplied importance. Small real-reference smoke runs
cover AWQ and GPTQ separately. Completed codebooks are checkpointed for
reuse on `/mnt/data/zch/quantization_experiments`.
