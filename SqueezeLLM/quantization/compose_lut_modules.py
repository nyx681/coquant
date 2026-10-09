"""Compose a dense SqueezeLLM LUT from a base folder and module overrides."""

import argparse
import json
import os
import pickle


parser = argparse.ArgumentParser()
parser.add_argument("--base_folder", required=True)
parser.add_argument("--override_folder", required=True)
parser.add_argument("--override_modules", required=True)
parser.add_argument("--output_folder", required=True)


def load_lut(path):
    with open(path, "rb") as handle:
        return pickle.load(handle)


if __name__ == "__main__":
    args = parser.parse_args()
    modules = [item.strip() for item in args.override_modules.split(",") if item.strip()]
    if not modules:
        parser.error("--override_modules must contain at least one module")

    base_lut_folder = os.path.join(args.base_folder, "lut")
    override_lut_folder = os.path.join(args.override_folder, "lut")
    if not os.path.isdir(base_lut_folder):
        parser.error(f"Missing base LUT folder: {base_lut_folder}")
    if not os.path.isdir(override_lut_folder):
        parser.error(f"Missing override LUT folder: {override_lut_folder}")

    layer_names = sorted(
        (
            name
            for name in os.listdir(base_lut_folder)
            if name.startswith("l") and name.endswith(".pkl")
        ),
        key=lambda name: int(name[1:-4]),
    )
    if not layer_names:
        parser.error(f"No l*.pkl files in {base_lut_folder}")

    output_lut_folder = os.path.join(args.output_folder, "lut")
    os.makedirs(output_lut_folder, exist_ok=True)
    for layer_name in layer_names:
        output_path = os.path.join(output_lut_folder, layer_name)
        if os.path.isfile(output_path) and os.path.getsize(output_path) > 0:
            print(f"Skipping existing {output_path}")
            continue
        base_layer = load_lut(os.path.join(base_lut_folder, layer_name))
        override_layer = load_lut(os.path.join(override_lut_folder, layer_name))
        for module in modules:
            if module not in base_layer or module not in override_layer:
                raise KeyError(f"Module {module} is missing from {layer_name}")
            base_layer[module] = override_layer[module]
        with open(output_path, "wb") as handle:
            pickle.dump(base_layer, handle)
        print(f"Saved {output_path}")

    with open(os.path.join(args.output_folder, "kmeans_config.json"), "w") as handle:
        json.dump(
            {
                "format": "squeezellm_composed_lut_v1",
                "base_folder": os.path.abspath(args.base_folder),
                "override_folder": os.path.abspath(args.override_folder),
                "override_modules": modules,
                "gradient_chunks": None,
                "sparse": False,
            },
            handle,
            indent=4,
        )

