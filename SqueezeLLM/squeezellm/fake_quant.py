"""Float simulation of SqueezeLLM LUT quantization.

This module expands the per-row K-means centroids and labels produced by
``quantization/nuq.py`` into ordinary floating-point ``nn.Linear`` weights.
It is intended for accuracy/PPL experiments; it does not provide compression
or the runtime characteristics of the packed CUDA kernels.
"""

import json
import os
import pickle

import numpy as np
import torch
from tqdm.auto import tqdm

from .model_parse import get_layers, get_module_names, get_modules, parse_model


def _dequantize_rows(lut_rows, shape, dtype, chunk_rows=128):
    if chunk_rows <= 0:
        raise ValueError("chunk_rows must be positive")
    if isinstance(lut_rows, dict):
        return _dequantize_groups(lut_rows, shape, dtype, chunk_rows)
    out_features, in_features = shape
    if len(lut_rows) != out_features:
        raise ValueError(
            f"LUT row count {len(lut_rows)} does not match output size "
            f"{out_features}"
        )
    result = torch.empty(shape, dtype=dtype, device="cpu")

    for start in range(0, out_features, chunk_rows):
        end = min(start + chunk_rows, out_features)
        centers_np = np.stack(
            [lut_rows[row][0][0] for row in range(start, end)], axis=0
        )
        labels_np = np.stack(
            [lut_rows[row][0][1] for row in range(start, end)], axis=0
        )
        if labels_np.shape != (end - start, in_features):
            raise ValueError(
                f"LUT label shape {labels_np.shape} does not match expected "
                f"{(end - start, in_features)}"
            )
        centers = torch.from_numpy(centers_np).float()
        labels = torch.from_numpy(labels_np.astype(np.int64, copy=False))
        result[start:end].copy_(
            torch.gather(centers, dim=1, index=labels).to(dtype=dtype)
        )
    return result


def _dequantize_groups(lut, shape, dtype, chunk_rows):
    """Expand compact G128 codebooks; this is not a packed-kernel format."""
    if lut.get("format") != "squeezellm_group_lut_v1" or lut.get("group_size") != 128:
        raise ValueError("Unsupported grouped LUT format")
    rows, columns = shape
    centers = np.asarray(lut["centers"])
    labels = np.asarray(lut["labels"])
    if (columns % 128 or centers.shape != (rows, columns // 128, 16)
            or labels.shape != (rows, columns // 128, 128)):
        raise ValueError("Grouped LUT shapes do not match model weights")
    if (not np.issubdtype(labels.dtype, np.integer) or not np.isfinite(centers).all()
            or np.any(labels < 0) or np.any(labels >= 16)):
        raise ValueError("Invalid grouped LUT centers or indices")
    result = torch.empty(shape, dtype=dtype, device="cpu")
    for start in range(0, rows, chunk_rows):
        end = min(start + chunk_rows, rows)
        block = np.take_along_axis(centers[start:end], labels[start:end].astype(np.int64), axis=2)
        result[start:end].copy_(torch.from_numpy(block.reshape(end-start, columns)).to(dtype))
    return result


def _restore_sparse_values(fake_weight, sparse_weight):
    if sparse_weight is None:
        return 0
    if tuple(sparse_weight.shape) != tuple(fake_weight.shape):
        raise ValueError(
            f"Sparse shape {tuple(sparse_weight.shape)} does not match fake "
            f"weight shape {tuple(fake_weight.shape)}"
        )

    if sparse_weight.layout == torch.strided:
        sparse_cpu = sparse_weight.detach().cpu()
        mask = sparse_cpu != 0
        fake_weight[mask] = sparse_cpu[mask].to(fake_weight.dtype)
        return int(mask.sum().item())

    sparse_coo = sparse_weight.to_sparse_coo().coalesce().cpu()
    indices = sparse_coo.indices()
    values = sparse_coo.values().to(fake_weight.dtype)
    if values.numel():
        fake_weight[indices[0], indices[1]] = values
    return int(values.numel())


@torch.no_grad()
def apply_squeezellm_fake_quant(
    model,
    folder,
    include_sparse=True,
    chunk_rows=128,
    fallback_folder=None,
    primary_modules=None,
):
    """Replace supported Linear weights with float-expanded SqueezeLLM LUTs."""
    if chunk_rows <= 0:
        raise ValueError("chunk_rows must be positive")

    kmeans_config = None
    kmeans_config_path = os.path.join(folder, "kmeans_config.json")
    if os.path.isfile(kmeans_config_path):
        with open(kmeans_config_path, "r") as f:
            kmeans_config = json.load(f)
        print(
            "Loading SqueezeLLM LUT with K-means weight source: "
            f"{kmeans_config.get('weight_source', 'unknown')}",
            flush=True,
        )

    model_type = parse_model(model)
    if model_type not in ("llama", "mistral", "opt"):
        raise NotImplementedError(
            "The current fake-LUT evaluator supports LLaMA/Mistral/OPT layouts"
        )
    layers = get_layers(model, model_type)
    module_names = get_module_names(model_type)
    if primary_modules is None:
        primary_modules = set(module_names)
    else:
        primary_modules = set(primary_modules)
        unknown_modules = primary_modules.difference(module_names)
        if unknown_modules:
            raise ValueError(
                f"Unknown primary LUT modules: {sorted(unknown_modules)}; "
                f"supported modules are {module_names}"
            )
    if fallback_folder is None and primary_modules != set(module_names):
        raise ValueError("A fallback_folder is required when selecting modules")
    total_linears = len(layers) * len(module_names)
    total_weights = 0
    total_sparse = 0

    progress = tqdm(
        total=total_linears,
        desc="expanding SqueezeLLM fake weights",
        unit="linear",
        dynamic_ncols=True,
    )
    for layer_idx, layer in enumerate(layers):
        lut_path = os.path.join(folder, "lut", f"l{layer_idx}.pkl")
        if not os.path.isfile(lut_path):
            raise FileNotFoundError(f"Missing LUT layer file: {lut_path}")
        with open(lut_path, "rb") as f:
            lut_layer = pickle.load(f)

        fallback_lut_layer = None
        if fallback_folder is not None:
            fallback_lut_path = os.path.join(
                fallback_folder, "lut", f"l{layer_idx}.pkl"
            )
            if not os.path.isfile(fallback_lut_path):
                raise FileNotFoundError(
                    f"Missing fallback LUT layer file: {fallback_lut_path}"
                )
            with open(fallback_lut_path, "rb") as f:
                fallback_lut_layer = pickle.load(f)

        sparse_layer = None
        if include_sparse:
            sparse_path = os.path.join(
                folder, "outliers", f"l{layer_idx}.pkl"
            )
            if not os.path.isfile(sparse_path):
                raise FileNotFoundError(
                    f"Missing sparse layer file: {sparse_path}"
                )
            with open(sparse_path, "rb") as f:
                sparse_layer = pickle.load(f)

        modules = get_modules(layer, model_type)
        if len(modules) != len(module_names):
            raise RuntimeError(
                f"Layer {layer_idx} module count mismatch: {len(modules)} vs "
                f"{len(module_names)}"
            )

        for module_idx, (module_name, module) in enumerate(
            zip(module_names, modules)
        ):
            progress.set_postfix_str(f"layer={layer_idx} module={module_name}")
            original_device = module.weight.device
            original_dtype = module.weight.dtype
            module_lut = (
                lut_layer[module_name]
                if module_name in primary_modules
                else fallback_lut_layer[module_name]
            )
            fake_weight = _dequantize_rows(
                module_lut,
                tuple(module.weight.shape),
                original_dtype,
                chunk_rows=chunk_rows,
            )
            sparse_count = 0
            if include_sparse:
                sparse_count = _restore_sparse_values(
                    fake_weight, sparse_layer[module_idx]
                )
            module.weight.data.copy_(fake_weight.to(original_device))
            total_weights += module.weight.numel()
            total_sparse += sparse_count
            del fake_weight
            progress.update(1)

        del lut_layer, fallback_lut_layer, sparse_layer

    progress.close()
    stats = {
        "linear_count": int(total_linears),
        "weight_count": int(total_weights),
        "sparse_count": int(total_sparse),
        "sparse_percentage": float(
            total_sparse / total_weights * 100.0 if total_weights else 0.0
        ),
        "kmeans_config": kmeans_config,
        "lut_composition": {
            "primary_folder": os.path.abspath(folder),
            "primary_modules": sorted(primary_modules),
            "fallback_folder": (
                os.path.abspath(fallback_folder)
                if fallback_folder is not None
                else None
            ),
        },
        "storage_note": (
            "fake quantization: all expanded weights use the source model tensor "
            "dtype; sparse values retain original numerical values"
        ),
    }
    print(
        "Applied SqueezeLLM fake quantization to "
        f"{total_linears} Linear layers; restored {total_sparse}/{total_weights} "
        f"sparse values ({stats['sparse_percentage']:.6f}%).",
        flush=True,
    )
    return stats
