"""Replace one module in an existing dense LUT folder in place."""

import argparse
import os
import pickle


parser = argparse.ArgumentParser()
parser.add_argument("--target_folder", required=True)
parser.add_argument("--source_folder", required=True)
parser.add_argument("--module", required=True)


if __name__ == "__main__":
    args = parser.parse_args()
    target_lut = os.path.join(args.target_folder, "lut")
    source_lut = os.path.join(args.source_folder, "lut")
    if not os.path.isdir(target_lut) or not os.path.isdir(source_lut):
        parser.error("target/source folder must contain a lut subdirectory")

    layer_names = sorted(
        (
            name
            for name in os.listdir(target_lut)
            if name.startswith("l") and name.endswith(".pkl")
        ),
        key=lambda name: int(name[1:-4]),
    )
    for layer_name in layer_names:
        target_path = os.path.join(target_lut, layer_name)
        source_path = os.path.join(source_lut, layer_name)
        with open(target_path, "rb") as handle:
            target = pickle.load(handle)
        with open(source_path, "rb") as handle:
            source = pickle.load(handle)
        if args.module not in target or args.module not in source:
            raise KeyError(f"{args.module} missing from {layer_name}")
        target[args.module] = source[args.module]
        temporary = target_path + ".incomplete"
        with open(temporary, "wb") as handle:
            pickle.dump(target, handle)
        os.replace(temporary, target_path)
        print(f"Updated {layer_name}: {args.module}")
