import torch


def remove_outliers_by_mask(model, importance_masks):
    """Move externally selected weights into the full-precision sparse matrix."""
    module_names = list(model.keys())
    outlier_weights = []
    total_selected = 0
    total_weights = 0

    missing = [name for name in module_names if name not in importance_masks]
    if missing:
        raise KeyError(f"Importance mask is missing module(s): {missing}")

    for name in module_names:
        weight = model[name].to(torch.float32)
        selection = importance_masks[name]
        if isinstance(selection, dict):
            shape = tuple(selection.get("shape", ()))
            if shape != tuple(weight.shape):
                raise ValueError(
                    f"Importance selection shape mismatch for {name}: "
                    f"selection={shape}, weight={tuple(weight.shape)}"
                )
            flat_indices = selection.get("flat_indices")
            if not torch.is_tensor(flat_indices):
                raise TypeError(
                    f"Importance selection for {name} must contain flat_indices"
                )
            flat_indices = flat_indices.to(device=weight.device, dtype=torch.int64)
            if flat_indices.dim() != 1:
                raise ValueError(f"flat_indices for {name} must be one-dimensional")
            if flat_indices.numel() and (
                flat_indices.min().item() < 0
                or flat_indices.max().item() >= weight.numel()
            ):
                raise IndexError(f"flat_indices for {name} are out of bounds")
            if torch.unique(flat_indices).numel() != flat_indices.numel():
                raise ValueError(f"flat_indices for {name} contain duplicates")

            selected_weight = torch.zeros_like(weight)
            selected_weight.reshape(-1)[flat_indices] = weight.reshape(-1)[flat_indices]
            dense_weight = weight.clone()
            dense_weight.reshape(-1)[flat_indices] = 0
            selected_count = flat_indices.numel()
        else:
            # Retain support for early dense-bool artifacts.
            if not torch.is_tensor(selection):
                raise TypeError(f"Importance mask for {name} must be a tensor")
            if tuple(selection.shape) != tuple(weight.shape):
                raise ValueError(
                    f"Importance mask shape mismatch for {name}: "
                    f"mask={tuple(selection.shape)}, weight={tuple(weight.shape)}"
                )
            mask = selection.to(device=weight.device, dtype=torch.bool)
            selected_weight = weight * mask
            dense_weight = weight * ~mask
            selected_count = int(mask.sum().item())

        model[name] = dense_weight
        outlier_weights.append(selected_weight)
        total_selected += selected_count
        total_weights += weight.numel()

    print(
        "p external importance:",
        total_selected / total_weights * 100 if total_weights else 0.0,
    )
    return [outlier_weights]


def remove_outliers_by_sensitivity(
    model,
    gradients,
    sensitivity,
):
    module_names = list(model.keys())
    outlier_weights = []
    total_outliers = 0
    total_weights = 0

    def _body(gweight, weight):
        num_outliers = int(gweight.numel() * sensitivity / 100)
        thres = gweight.reshape(-1).topk(k=num_outliers).values[-1]

        t = gweight > thres

        outlier_weight = weight * t
        weight = weight * ~t
        # print((weight == 0).sum().item() / weight.numel())
        return weight.to(weight.dtype), outlier_weight, t.sum().item(), t.numel()

    for _name in module_names:
        weight = model[_name].to(torch.float)
        gweight = gradients[_name].to(torch.float)
        new_weight, outlier_weight, _total_outliers, _total_weights = _body(
            gweight, weight
        )
        model[_name] = new_weight
        total_outliers += _total_outliers
        total_weights += _total_weights
        outlier_weights.append(outlier_weight)

    print("p outlier:", total_outliers / total_weights * 100)
    return [outlier_weights]


def remove_outliers_by_threshold(
    model,
    outlier_config,
    outlier_weights=None,  # to avoid memory leak
):
    module_names = list(model.keys())
    if outlier_weights is None:
        outlier_weights = [[0 for _ in range(len(module_names))]]

    total_outliers = 0
    total_weights = 0

    def _body(weight, thres):
        t = torch.logical_or(
            weight >= thres,
            weight <= -thres,
        )

        outlier_weight = weight * t
        weight = weight * ~t
        return weight.to(weight.dtype), outlier_weight, t.sum().item(), t.numel()

    for i, name in enumerate(module_names):
        thres = outlier_config[name]
        weight = model[name].to(torch.float)
        new_weight, outlier_weight, _total_outliers, _total_weights = _body(
            weight, thres
        )
        model[name] = new_weight
        total_outliers += _total_outliers
        total_weights += _total_weights
        # import pdb; pdb.set_trace()
        outlier_weights[0][i] += outlier_weight

    print("p outlier:", total_outliers / total_weights * 100)
    return outlier_weights


def remove_outliers(
    model,
    sensitivity,
    outlier_config,
    gradients=None,
):
    # model and gradients are dictionary of a layer component
    # where the key is the layer name (e.g. q, k, v) and the value is the weight
    assert isinstance(model, dict)
    assert isinstance(gradients, dict) or gradients is None

    assert outlier_config is not None or sensitivity != 0
    if sensitivity != 0:
        assert gradients is not None

    if sensitivity != 0:
        print("removing outliers by sensitivity")
        outlier_weights = remove_outliers_by_sensitivity(
            model=model,
            gradients=gradients,
            sensitivity=sensitivity,
        )
    else:
        outlier_weights = None

    if outlier_config is not None:
        print("removing outliers by threshold")
        outlier_weights = remove_outliers_by_threshold(
            model=model,
            outlier_config=outlier_config,
            outlier_weights=outlier_weights,
        )

    return outlier_weights
