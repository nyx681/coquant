# Main Experiment Code

Code for 3-bit and 4-bit weight quantization experiments with AWQ, GPTQ,
EGConv-A, EGConv-G, and EGComp-AG. EGConv converts a source quantization;
EGComp combines the AWQ-derived and GPTQ-derived conversions.
Perplexity is evaluated on WikiText-2 and C4.

Supported models: LLaMA-1-7B, LLaMA-2-7B, OPT-6.7B, Qwen3-8B-Base,
and Mistral-7B-v0.1.

## Setup

Requires PyTorch, NumPy, scikit-learn, Transformers, and Datasets.
Baseline generation requires CUDA; conversion and composition run on CPU.
Provide local Hugging Face model weights, tokenizer files, and datasets.

## Usage

Run scripts from this directory. The workflow is:

1. Export original weights with `scripts/prepare_sources.py`.
2. Prepare tokens with `scripts/prepare_tokens.py` and generate AWQ/GPTQ
   weights with `scripts/generate_baselines.py`.
3. Collect Fisher statistics with `scripts/collect_fisher.py`.
4. Run conversion and composition with `scripts/run_quantization.py`.
5. Evaluate perplexity with `scripts/evaluate_ppl.py`.

Scripts accept `--model`, `--bits`, `--model-dir`, `--data-dir`, and
`--work-dir`. See each script's `--help` for options, for example:

```bash
python scripts/run_quantization.py --help
```

Core implementations are in `src/egquant/`; regression tests are in `tests/`.
Third-party license texts are in `src/egquant/_third_party.py`.
