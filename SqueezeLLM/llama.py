import time
import json
import os

import torch
import torch.nn as nn

from squeezellm.modelutils import *

from squeezellm.model_parse import (
    parse_model,
    get_layers,
    get_embedding,
    get_norm,
)


def get_model(model):
    import torch

    def skip(*args, **kwargs):
        pass

    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    from transformers import AutoModelForCausalLM

    try:
        model = AutoModelForCausalLM.from_pretrained(model, torch_dtype="auto")
    except FileNotFoundError as exc:
        if ".safetensors" not in str(exc):
            raise
        model = AutoModelForCausalLM.from_pretrained(
            model, torch_dtype="auto", use_safetensors=False
        )
    model.seqlen = 2048
    return model


@torch.no_grad()
def llama_eval(model, testenc, dev):
    print("Evaluating ...")
    model_type = parse_model(model)

    testenc = testenc.input_ids
    nsamples = testenc.numel() // model.seqlen

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = get_layers(model, model_type)
    embeddings = get_embedding(model, model_type)
    for i in range(len(embeddings)):
        embeddings[i] = embeddings[i].to(dev)

    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {"i": 0, "attention_mask": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp
            cache["i"] += 1
            cache["attention_mask"] = kwargs["attention_mask"]
            if "position_ids" in kwargs:
                cache["position_ids"] = kwargs["position_ids"]
            raise ValueError

    layers[0] = Catcher(layers[0])
    for i in range(nsamples):
        batch = testenc[:, (i * model.seqlen) : ((i + 1) * model.seqlen)].to(dev)
        try:
            model(batch)
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    for i in range(len(embeddings)):
        embeddings[i] = embeddings[i].cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache["attention_mask"]
    position_ids = cache.get("position_ids")

    for i in range(len(layers)):
        print("Layer", i)
        layer = layers[i].to(dev)

        for j in range(nsamples):
            if model_type == "opt":
                outs[j] = layer(
                    inps[j].unsqueeze(0),
                    attention_mask=attention_mask,
                )[0]
            else:
                assert model_type in ("llama", "mistral")
                outs[j] = layer(
                    inps[j].unsqueeze(0),
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                )[0]
        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()
        inps, outs = outs, inps

    norm = get_norm(model, model_type)
    if norm is not None:
        norm = norm.to(dev)
    model.lm_head = model.lm_head.to(dev)

    testenc = testenc.to(dev)
    nlls = []
    for i in range(nsamples):
        hidden_states = inps[i].unsqueeze(0)
        if norm is not None:
            hidden_states = norm(hidden_states)
        lm_logits = model.lm_head(hidden_states)
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = testenc[:, (i * model.seqlen) : ((i + 1) * model.seqlen)][:, 1:]
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
        )
        neg_log_likelihood = loss.float() * model.seqlen
        nlls.append(neg_log_likelihood)
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))
    print(ppl.item())

    model.config.use_cache = use_cache


@torch.no_grad()
def awq_compatible_ppl_eval(
    model, input_ids, dev, dataset_name, block_size=2048, batch_size=1
):
    """Evaluate with the same token stream and loss loop used by AWQ."""
    from tqdm.auto import tqdm

    print(f"Evaluating AWQ-compatible {dataset_name} PPL ...", flush=True)
    model = model.to(dev)
    model.eval()

    use_cache = model.config.use_cache
    model.config.use_cache = False
    nsamples = input_ids.numel() // block_size
    if nsamples == 0:
        raise ValueError(
            f"{dataset_name} has only {input_ids.numel()} tokens, fewer than "
            f"one {block_size}-token evaluation block"
        )

    if batch_size < 1:
        raise ValueError(f"batch_size must be positive, got {batch_size}")

    nlls = []
    progress = tqdm(
        range(0, nsamples, batch_size),
        total=(nsamples + batch_size - 1) // batch_size,
        desc=f"{dataset_name} PPL",
        unit="batch",
        dynamic_ncols=True,
        mininterval=1.0,
    )
    for i in progress:
        current_batch_size = min(batch_size, nsamples - i)
        start_idx = i * block_size
        end_idx = (i + current_batch_size) * block_size
        batch = input_ids[:, start_idx:end_idx].reshape(
            current_batch_size, block_size
        ).to(dev)
        outputs = model(batch)
        lm_logits = outputs.logits
        shift_logits = lm_logits[:, :-1, :].contiguous().float()
        shift_labels = batch[:, 1:].contiguous()
        loss = torch.nn.functional.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            reduction="mean",
        )
        nlls.append(loss.float() * (current_batch_size * block_size))

        del outputs, lm_logits, shift_logits, shift_labels, loss, batch
        if (i + current_batch_size) % (10 * batch_size) == 0:
            torch.cuda.empty_cache()

    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * block_size))
    result = float(ppl.item())
    print(f"{dataset_name} PPL: {result}", flush=True)
    model.config.use_cache = use_cache
    return result


# loading quantized checkpoint
def load_quant(model, checkpoint, wbits, include_sparse, topX):
    # The packed CUDA extension is unnecessary for float-expanded LUT
    # evaluation and may be built against a different PyTorch environment.
    # Import it only when a packed checkpoint is actually requested.
    from squeezellm.quant import make_quant_lut

    if (
        "xgen" in checkpoint
        or "opt" in checkpoint
        or ("vicuna" in checkpoint and "v1.3" in checkpoint)
        or "llama-2" in checkpoint
        or "mistral" in checkpoint
    ):
        # TODO: this is a hacky solution, will be preperly implemented after all the model checkpoints are updated with
        # the new packing scheme that includes the non-linear weights
        from transformers import AutoConfig, AutoModelForCausalLM

        config = AutoConfig.from_pretrained(model)
        model = AutoModelForCausalLM.from_config(config)
    else:
        from transformers import LlamaForCausalLM

        model = LlamaForCausalLM.from_pretrained(model, torch_dtype="auto")
    model = model.eval()
    layers = find_layers(model)

    state_dict = torch.load(checkpoint)

    # load sparse thresholds from checkpoint
    if include_sparse:
        num_vals = {}
        for k, v in state_dict.items():
            if "sparse_threshold." in k:
                key = k.replace("sparse_threshold.", "")
                num_vals[key] = v
        for k, v in num_vals.items():
            del state_dict["sparse_threshold." + k]
    else:
        num_vals = None

    # replace layers
    for name in ["lm_head"]:
        if name in layers:
            del layers[name]
    make_quant_lut(
        model, layers, wbits, include_sparse=include_sparse, numvals=num_vals, topX=topX
    )
    del layers

    print("Loading model ...")
    state_dict = torch.load(checkpoint)
    model.load_state_dict(state_dict, strict=False)
    model.seqlen = 2048
    print("Done.")

    return model


# function for benchmarking runtime
def benchmark(model, input_ids, check=False):
    model_type = parse_model(model)
    layers = get_layers(model, model_type)

    input_ids = input_ids.to(model.gpus[0] if hasattr(model, "gpus") else DEV)
    torch.cuda.synchronize()

    cache = {"past": None}

    def clear_past(i):
        def tmp(layer, inp, out):
            if cache["past"]:
                cache["past"][i] = None

        return tmp

    for i, layer in enumerate(layers):
        layer.register_forward_hook(clear_past(i))

    print("Benchmarking ...")

    if check:
        loss = nn.CrossEntropyLoss()
        tot = 0.0

    def sync():
        if hasattr(model, "gpus"):
            for gpu in model.gpus:
                torch.cuda.synchronize(gpu)
        else:
            torch.cuda.synchronize()

    max_memory = 0
    with torch.no_grad():
        attention_mask = torch.ones((1, input_ids.numel()), device=DEV)
        times = []
        for i in range(input_ids.numel()):
            tick = time.time()
            out = model(
                input_ids[:, i : i + 1],
                past_key_values=cache["past"],
                attention_mask=attention_mask[:, : (i + 1)].reshape((1, -1)),
            )
            sync()
            times.append(time.time() - tick)
            print(i, times[-1])
            max_memory = max(max_memory, torch.cuda.memory_allocated() / 1024 / 1024)
            if check and i != input_ids.numel() - 1:
                tot += loss(
                    out.logits[0].to(DEV), input_ids[:, (i + 1)].to(DEV)
                ).float()
            cache["past"] = list(out.past_key_values)
            del out
        sync()
        import numpy as np

        print("Median:", np.median(times))
        if check:
            print("PPL:", torch.exp(tot / (input_ids.numel() - 1)).item())
            print("max memory(MiB):", max_memory)


if __name__ == "__main__":
    import argparse
    from squeezellm.datautils import *

    parser = argparse.ArgumentParser()

    parser.add_argument("model", type=str, help="llama model to load")
    parser.add_argument(
        "dataset",
        type=str,
        choices=["wikitext2", "ptb", "c4"],
        help="Which dataset to use for benchmarking.",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Seed for sampling the calibration data."
    )
    parser.add_argument(
        "--wbits",
        type=int,
        default=16,
        choices=[3, 4, 16],
        help="#bits to use for quantization; use 16 for evaluating base model.",
    )
    parser.add_argument("--eval", action="store_true", help="evaluate quantized model.")
    parser.add_argument(
        "--awq_fair_eval",
        action="store_true",
        help=(
            "evaluate WikiText2 and C4 with the exact local dataset paths and "
            "token construction used by llm-awq_zch0810mixed-precision"
        ),
    )
    parser.add_argument(
        "--ppl_output",
        type=str,
        default=None,
        help="optional JSON file for AWQ-compatible PPL results",
    )
    parser.add_argument(
        "--awq_fair_datasets",
        type=str,
        default="wikitext,c4",
        help=(
            "comma-separated datasets for --awq_fair_eval; supported values: "
            "wikitext,c4 (default: both)"
        ),
    )
    parser.add_argument(
        "--awq_fair_batch_size",
        type=int,
        default=1,
        help=(
            "number of independent 2048-token blocks evaluated per forward "
            "pass in --awq_fair_eval (default: 1)"
        ),
    )
    parser.add_argument("--load", type=str, default="", help="Load quantized model.")
    parser.add_argument(
        "--fake_lut_folder",
        type=str,
        default="",
        help=(
            "expand a SqueezeLLM LUT folder into ordinary floating-point Linear "
            "weights for PPL simulation; mutually exclusive with --load"
        ),
    )
    parser.add_argument(
        "--fake_weight_chunk_rows",
        type=int,
        default=128,
        help="number of output rows expanded at once during fake LUT loading",
    )
    parser.add_argument(
        "--fake_lut_fallback_folder",
        type=str,
        default="",
        help="optional fallback LUT folder for modules not selected from --fake_lut_folder",
    )
    parser.add_argument(
        "--fake_lut_primary_modules",
        type=str,
        default="",
        help="comma-separated module names loaded from --fake_lut_folder when a fallback is set",
    )
    parser.add_argument(
        "--benchmark",
        type=int,
        default=0,
        help="Number of tokens to use for benchmarking.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Whether to compute perplexity during benchmarking for verification.",
    )
    parser.add_argument(
        "--nsamples", type=int, default=128, help="Number of calibration data samples."
    )
    parser.add_argument(
        "--torch_profile",
        action="store_true",
        help="Use CUDA profiling tool for timing runs.",
    )
    parser.add_argument(
        "--include_sparse",
        action="store_true",
        help="Whether loaded checkpoint has sparse matrix.",
    )
    parser.add_argument(
        "--num_dense_channels",
        type=int,
        default=10,
        help="Number of dense channel used for hybrid kernel.",
    )

    DEV = torch.device("cuda:0")

    args = parser.parse_args()

    if type(args.load) is not str:
        args.load = args.load.as_posix()
    if args.load and args.fake_lut_folder:
        parser.error("--load and --fake_lut_folder are mutually exclusive")

    if args.load:
        print(args.model)
        model = load_quant(
            args.model,
            args.load,
            args.wbits,
            args.include_sparse,
            args.num_dense_channels,
        )
    elif args.fake_lut_folder:
        from squeezellm.fake_quant import apply_squeezellm_fake_quant

        model = get_model(args.model)
        model.eval()
        fake_quant_stats = apply_squeezellm_fake_quant(
            model,
            args.fake_lut_folder,
            include_sparse=args.include_sparse,
            chunk_rows=args.fake_weight_chunk_rows,
            fallback_folder=args.fake_lut_fallback_folder or None,
            primary_modules=(
                [x.strip() for x in args.fake_lut_primary_modules.split(",") if x.strip()]
                if args.fake_lut_fallback_folder
                else None
            ),
        )
    else:
        model = get_model(args.model)
        model.eval()

    if args.benchmark or (args.eval and not args.awq_fair_eval):
        dataloader, testloader = get_loaders(
            args.dataset,
            nsamples=args.nsamples,
            seed=args.seed,
            model=args.model,
            seqlen=model.seqlen,
        )

    if args.benchmark:
        model = model.to(DEV)
        if args.benchmark:
            input_ids = next(iter(dataloader))[0][:, : args.benchmark]

            if args.torch_profile:
                from torch.profiler import profile, record_function, ProfilerActivity

                with torch.profiler.profile(
                    activities=[
                        torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA,
                    ]
                ) as p:
                    benchmark(model, input_ids, check=args.check)
                print(
                    p.key_averages().table(sort_by="self_cuda_time_total", row_limit=-1)
                )
            else:
                benchmark(model, input_ids, check=args.check)

    if args.awq_fair_eval:
        from squeezellm.datautils import get_awq_ppl_tokens

        results = {}
        dataset_paths = {}
        fair_datasets = []
        for raw_dataset in args.awq_fair_datasets.split(","):
            dataset = raw_dataset.strip().lower()
            if dataset in ("wiki", "wiki2", "wikitext2"):
                dataset = "wikitext"
            if dataset not in ("wikitext", "c4"):
                raise ValueError(
                    f"Unsupported --awq_fair_datasets value: {raw_dataset}"
                )
            if dataset not in fair_datasets:
                fair_datasets.append(dataset)
        if not fair_datasets:
            raise ValueError("--awq_fair_datasets must select at least one dataset")

        if args.ppl_output is not None and os.path.exists(args.ppl_output):
            with open(args.ppl_output, "r") as f:
                previous = json.load(f)
            quantization_mode = (
                "fake_lut" if args.fake_lut_folder else "packed" if args.load else "base"
            )
            quantization_artifact = args.fake_lut_folder or args.load
            expected_metadata = {
                "evaluation": "awq_compatible_ppl_v1",
                "model": args.model,
                "checkpoint": quantization_artifact,
                "wbits": args.wbits,
                "include_sparse": args.include_sparse,
                "block_size": 2048,
            }
            if args.fake_lut_folder:
                expected_metadata["quantization_mode"] = quantization_mode
            mismatches = {
                key: (previous.get(key), value)
                for key, value in expected_metadata.items()
                if previous.get(key) != value
            }
            if mismatches:
                raise ValueError(
                    f"Cannot resume incompatible PPL result {args.ppl_output}: "
                    f"{mismatches}"
                )
            results.update(previous.get("results", {}))
            dataset_paths.update(previous.get("dataset_paths", {}))
            print(
                f"Resuming PPL results from {args.ppl_output}; completed: "
                f"{sorted(results)}",
                flush=True,
            )

        def save_fair_results(completed):
            if args.ppl_output is None:
                return
            output_dir = os.path.dirname(os.path.abspath(args.ppl_output))
            os.makedirs(output_dir, exist_ok=True)
            with open(args.ppl_output, "w") as f:
                json.dump(
                    {
                        "evaluation": "awq_compatible_ppl_v1",
                        "status": (
                            "complete"
                            if len(completed) == len(fair_datasets)
                            else "partial"
                        ),
                        "requested_datasets": fair_datasets,
                        "completed_datasets": completed,
                        "model": args.model,
                        "checkpoint": args.fake_lut_folder or args.load,
                        "quantization_mode": (
                            "fake_lut"
                            if args.fake_lut_folder
                            else "packed"
                            if args.load
                            else "base"
                        ),
                        "wbits": args.wbits,
                        "include_sparse": args.include_sparse,
                        "block_size": 2048,
                        "eval_batch_size": args.awq_fair_batch_size,
                        "dataset_paths": dataset_paths,
                        "results": results,
                        "fake_quant_stats": (
                            fake_quant_stats if args.fake_lut_folder else None
                        ),
                    },
                    f,
                    indent=2,
                )
            print(f"Saved PPL results to {args.ppl_output}", flush=True)

        completed_datasets = [
            dataset for dataset in fair_datasets if dataset in results
        ]
        for dataset in fair_datasets:
            if dataset in results:
                print(
                    f"Skipping completed {dataset} PPL: "
                    f"{results[dataset]['ppl']}",
                    flush=True,
                )
                continue
            input_ids, dataset_path = get_awq_ppl_tokens(dataset, args.model)
            dataset_paths[dataset] = dataset_path
            print(
                f"{dataset}: path={dataset_path}, tokens={input_ids.numel()}, "
                f"blocks={input_ids.numel() // 2048}, block_size=2048",
                flush=True,
            )
            results[dataset] = {
                "ppl": awq_compatible_ppl_eval(
                    model,
                    input_ids,
                    DEV,
                    dataset,
                    block_size=2048,
                    batch_size=args.awq_fair_batch_size,
                )
            }
            del input_ids
            completed_datasets.append(dataset)
            save_fair_results(completed_datasets)

    elif args.eval:
        datasets = ["wikitext2", "c4"]
        for dataset in datasets:
            dataloader, testloader = get_loaders(
                dataset, seed=args.seed, model=args.model, seqlen=model.seqlen
            )
            llama_eval(model, testloader, DEV)
