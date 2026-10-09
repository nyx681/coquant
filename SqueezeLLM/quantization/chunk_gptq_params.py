import argparse
import json
import os
import re

import torch
from tqdm import tqdm


LLAMA_MODULES = {
    "q": "self_attn.q_proj",
    "k": "self_attn.k_proj",
    "v": "self_attn.v_proj",
    "o": "self_attn.o_proj",
    "gate": "mlp.gate_proj",
    "up": "mlp.up_proj",
    "down": "mlp.down_proj",
}

OPT_MODULES = {
    "q": "self_attn.q_proj",
    "k": "self_attn.k_proj",
    "v": "self_attn.v_proj",
    "o": "self_attn.out_proj",
    "up": "fc1",
    "down": "fc2",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert exported GPTQ fake-quant weights into SqueezeLLM layer chunks."
    )
    parser.add_argument("--gptq_params", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--original_chunks", required=True)
    parser.add_argument("--bit", required=True, type=int, choices=[2, 3, 4, 8])
    parser.add_argument("--group_size", required=True, type=int)
    parser.add_argument(
        "--model_type", choices=["llama", "opt"], default="llama"
    )
    return parser.parse_args()


def infer_group_size(record, in_features):
    if "group_size" in record:
        return int(record["group_size"])
    scale = record.get("scale")
    if not torch.is_tensor(scale) or scale.ndim != 2:
        return None
    if scale.shape[1] == 1:
        return -1
    if in_features % scale.shape[1] != 0:
        return None
    return in_features // scale.shape[1]


def main():
    args = parse_args()
    if not os.path.isfile(args.gptq_params):
        raise FileNotFoundError(args.gptq_params)
    if args.group_size == 0 or args.group_size < -1:
        raise ValueError("--group_size must be -1 or a positive integer")

    params = torch.load(args.gptq_params, map_location="cpu")
    if not isinstance(params, dict):
        raise TypeError("GPTQ parameter artifact must be a dictionary")

    if args.model_type == "opt":
        layer_prefix = "model.decoder.layers"
        modules = OPT_MODULES
    else:
        layer_prefix = "model.layers"
        modules = LLAMA_MODULES
    layer_pattern = re.compile(rf"^{re.escape(layer_prefix)}\.(\d+)\.")
    layer_indices = sorted(
        {
            int(match.group(1))
            for name in params
            for match in [layer_pattern.match(name)]
            if match is not None
        }
    )
    if not layer_indices or layer_indices != list(range(layer_indices[-1] + 1)):
        raise ValueError(f"GPTQ artifact has non-contiguous layer indices: {layer_indices}")

    expected_maxq = 2**args.bit - 1
    os.makedirs(args.output_path, exist_ok=True)
    for layer_idx in tqdm(layer_indices, desc="chunking GPTQ reference weights"):
        original_path = os.path.join(args.original_chunks, f"layer_{layer_idx}.pt")
        if not os.path.isfile(original_path):
            raise FileNotFoundError(original_path)
        original = torch.load(original_path, map_location="cpu")
        chunk = {}

        for short_name, suffix in modules.items():
            full_name = f"{layer_prefix}.{layer_idx}.{suffix}"
            if full_name not in params:
                raise KeyError(f"GPTQ artifact is missing {full_name}")
            record = params[full_name]
            if not isinstance(record, dict) or not torch.is_tensor(record.get("weight")):
                raise TypeError(f"GPTQ record {full_name} has no tensor weight")
            weight = record["weight"].detach().cpu().contiguous()
            if short_name not in original:
                raise KeyError(f"Original chunk {layer_idx} is missing {short_name}")
            if tuple(weight.shape) != tuple(original[short_name].shape):
                raise ValueError(
                    f"Shape mismatch for {full_name}: GPTQ={tuple(weight.shape)}, "
                    f"original={tuple(original[short_name].shape)}"
                )
            if not torch.isfinite(weight).all():
                raise ValueError(f"Non-finite GPTQ weight in {full_name}")

            maxq = int(record.get("maxq", expected_maxq))
            if maxq != expected_maxq:
                raise ValueError(
                    f"Bit-width mismatch for {full_name}: maxq={maxq}, "
                    f"expected {expected_maxq} for W{args.bit}"
                )
            inferred_group_size = infer_group_size(record, weight.shape[1])
            if inferred_group_size != args.group_size:
                raise ValueError(
                    f"Group-size mismatch for {full_name}: artifact={inferred_group_size}, "
                    f"requested={args.group_size}. Do not use an old full-row export as g128."
                )
            chunk[short_name] = weight

        output_path = os.path.join(args.output_path, f"layer_{layer_idx}.pt")
        temporary_path = output_path + ".incomplete"
        torch.save(chunk, temporary_path)
        os.replace(temporary_path, output_path)

    manifest = {
        "format": "squeezellm_gptq_reference_chunks_v1",
        "source": os.path.abspath(args.gptq_params),
        "bit": args.bit,
        "group_size": args.group_size,
        "layers": len(layer_indices),
        "model_type": args.model_type,
        "modules": list(modules),
    }
    with open(os.path.join(args.output_path, "gptq_reference.json"), "w") as handle:
        json.dump(manifest, handle, indent=2)
    print(f"Saved {len(layer_indices)} validated GPTQ reference chunks to {args.output_path}")


if __name__ == "__main__":
    main()
