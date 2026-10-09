import argparse
import json
import os

import numpy as np
import torch
from squeezellm.model_parse import get_module_names, parse_model
from tqdm import tqdm

parser = argparse.ArgumentParser()

parser.add_argument("--model", type=str, help="llama model to load")
parser.add_argument(
    "--model_type", type=str, default=None, help="model type", choices=["llama", "opt"]
)
parser.add_argument("--cache_dir", type=str, default=None, help="cache directory")
range_group = parser.add_mutually_exclusive_group(required=True)
range_group.add_argument(
    "--range",
    type=float,
    default=None,
    help="fixed IQR multiplier used for the outlier threshold, e.g. 1.8",
)
range_group.add_argument(
    "--target_percentage",
    type=float,
    default=None,
    help=(
        "automatically choose one global IQR multiplier whose aggregate "
        "magnitude-outlier percentage is this value"
    ),
)
parser.add_argument(
    "--output_folder", type=str, required=None, help="path to dump the output"
)

args = parser.parse_args()

model_type = args.model_type or parse_model(args.model)
threshold_range = args.range

if args.target_percentage is not None and not 0 < args.target_percentage < 100:
    parser.error("--target_percentage must be between 0 and 100")

nlayers = len([f for f in os.listdir(args.model)])

total_params = 0
total_outliers = 0
json_data = []
module_statistics = []
for l in tqdm(range(nlayers)):
    try:
        model_layer = torch.load(f"{args.model}/layer_{l}.pt")
    except:
        raise Exception(f"Needs chunked model weight file at {args.model}/layer_{l}.pt")
    layer_json = {}

    for name in get_module_names(model_type):
        module_weight = model_layer[name]
        # NumPy 2.x may overflow internally when computing quantiles directly
        # from float16 tensors, producing NaN thresholds and zero outliers.
        # SqueezeLLM's threshold statistics must therefore be accumulated in
        # float32; the stored model weights themselves are left unchanged.
        weights_np = module_weight.detach().float().numpy()
        q1 = np.quantile(weights_np, 0.25)
        q3 = np.quantile(weights_np, 0.75)
        num_params = weights_np.shape[0] * weights_np.shape[1]
        total_params += num_params
        module_statistics.append(
            {
                "layer": l,
                "name": name,
                # NumPy has no native BF16 dtype.  Keep the statistics array
                # in float32 for both threshold calibration and counting.
                "weights": weights_np,
                "q1": float(q1),
                "q3": float(q3),
                "iqr": float(q3 - q1),
                "num_params": num_params,
            }
        )

    json_data.append(layer_json)


def threshold_for(statistics, multiplier):
    minimum = statistics["q1"] - multiplier * statistics["iqr"]
    maximum = statistics["q3"] + multiplier * statistics["iqr"]
    return max(abs(minimum), abs(maximum))


def count_outliers(multiplier):
    count = 0
    for statistics in module_statistics:
        weights = statistics["weights"]
        threshold = threshold_for(statistics, multiplier)
        count += int((weights < -threshold).sum())
        count += int((weights > threshold).sum())
    return count


if args.target_percentage is not None:
    target_count = total_params * args.target_percentage / 100.0
    lower = 0.0
    upper = 2.0
    upper_count = count_outliers(upper)
    while upper_count > target_count:
        lower = upper
        upper *= 2.0
        upper_count = count_outliers(upper)
        if upper > 1024:
            raise RuntimeError(
                "Could not bracket target outlier percentage; check model weights"
            )

    # Outlier count decreases monotonically as the IQR multiplier grows.
    # Twenty-four iterations make multiplier error negligible relative to the
    # discrete element count and the two-decimal percentage stored on disk.
    for _ in range(24):
        midpoint = (lower + upper) / 2.0
        if count_outliers(midpoint) > target_count:
            lower = midpoint
        else:
            upper = midpoint
    threshold_range = (lower + upper) / 2.0
    print(
        "Auto-selected IQR range {:.8f} for target {:.6f}%".format(
            threshold_range, args.target_percentage
        )
    )

for statistics in module_statistics:
    threshold = threshold_for(statistics, threshold_range)
    weights = statistics.pop("weights")
    num_outliers = int((weights < -threshold).sum()) + int(
        (weights > threshold).sum()
    )
    total_outliers += num_outliers
    json_data[statistics["layer"]][statistics["name"]] = float(threshold)
    print(
        statistics["layer"],
        statistics["name"],
        f"% outlier: {num_outliers / statistics['num_params'] * 100:.3f}%",
    )

# Save the json data
outlier_percentage = total_outliers / total_params * 100

outlier_folder = f"{args.output_folder}"
if not os.path.exists(outlier_folder):
    os.makedirs(outlier_folder)

o = round(outlier_percentage, 2)
json_data = {
    "outlier_threshold": o,
    "threshold_range": threshold_range,
    "target_percentage": args.target_percentage,
    "outlier_config": json_data,
}

with open(f"{outlier_folder}/outlier_config_o{o}.json", "w") as f:
    json.dump(json_data, f, indent=4)
