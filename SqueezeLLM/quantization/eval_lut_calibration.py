"""Evaluate a dense SqueezeLLM fake LUT on deterministic WikiText train blocks."""

import argparse
import json
import os

import torch

from llama import awq_compatible_ppl_eval, get_model
from squeezellm.datautils import get_wikitext_calibration_tokens
from squeezellm.fake_quant import apply_squeezellm_fake_quant


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--lut_folder", required=True)
    parser.add_argument("--fallback_folder", default=None)
    parser.add_argument("--primary_modules", default=None)
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    model = get_model(args.model)
    model.eval()
    modules = (
        [item.strip() for item in args.primary_modules.split(",") if item.strip()]
        if args.primary_modules
        else None
    )
    stats = apply_squeezellm_fake_quant(
        model,
        args.lut_folder,
        include_sparse=False,
        chunk_rows=512,
        fallback_folder=args.fallback_folder,
        primary_modules=modules,
    )
    tokens, dataset_path = get_wikitext_calibration_tokens(
        args.model,
        nsamples=args.samples,
        seed=args.seed,
        block_size=model.seqlen,
    )
    ppl = awq_compatible_ppl_eval(
        model,
        tokens,
        torch.device("cuda:0"),
        dataset_name=f"wikitext-train-seed{args.seed}",
        block_size=model.seqlen,
        batch_size=args.batch_size,
    )
    result = {
        "format": "squeezellm_calibration_ppl_v1",
        "model": os.path.abspath(args.model),
        "dataset": "wikitext2_train",
        "dataset_path": dataset_path,
        "samples": args.samples,
        "seed": args.seed,
        "ppl": ppl,
        "lut_folder": os.path.abspath(args.lut_folder),
        "fallback_folder": (
            os.path.abspath(args.fallback_folder)
            if args.fallback_folder
            else None
        ),
        "primary_modules": modules,
        "fake_quant_stats": stats,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as handle:
        json.dump(result, handle, indent=2)
    print(f"Calibration PPL: {ppl:.12f}")
    print(f"Saved: {args.output}")
