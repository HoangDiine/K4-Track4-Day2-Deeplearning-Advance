"""Validation-time inference methods; fit every choice on validation data only."""
from __future__ import annotations

import numpy as np


def predict_logits(model, loader, device, view=None, amp: bool = False):
    import torch
    model.eval()
    filenames, truths, outputs = [], [], []
    device = torch.device(device)
    with torch.inference_mode():
        for images, labels, names in loader:
            images = images.to(device, non_blocking=True)
            if view is not None:
                images = view(images)
            with torch.autocast(device_type=device.type, dtype=torch.float16,
                                enabled=bool(amp and device.type == "cuda")):
                logits = model(images)
            filenames.extend(list(names))
            truths.append(labels.cpu().numpy())
            outputs.append(logits.float().cpu().numpy())
    y_true = np.concatenate(truths) if truths else np.empty(0, dtype=np.int64)
    logits = np.concatenate(outputs) if outputs else np.empty((0, 9), dtype=np.float32)
    return filenames, y_true, logits


def view_identity(x):
    return x


def view_hflip(x):
    import torch
    return torch.flip(x, dims=(-1,))


def views_multicrop(x, crop: int):
    height, width = x.shape[-2:]
    if crop <= 0 or crop > min(height, width):
        raise ValueError(f"crop={crop} phải nằm trong 1..{min(height, width)}")
    offsets = [(0, 0), (0, width - crop), (height - crop, 0),
               (height - crop, width - crop), ((height - crop) // 2, (width - crop) // 2)]
    return [x[..., top:top + crop, left:left + crop] for top, left in offsets]


def views_multiscale(x, sizes):
    import torch.nn.functional as F
    views = []
    for size in sizes:
        if isinstance(size, int):
            shape = (size, size)
        else:
            shape = tuple(size)
        views.append(F.interpolate(x, size=shape, mode="bilinear", align_corners=False,
                                   antialias=True))
    return views


def _softmax(logits):
    values = np.asarray(logits, dtype=np.float64)
    values = values - values.max(axis=1, keepdims=True)
    exp = np.exp(values)
    return exp / exp.sum(axis=1, keepdims=True)


def aggregate_views(logits_per_view, space: str = "prob"):
    if not logits_per_view:
        raise ValueError("Cần ít nhất một view")
    logits = [np.asarray(value, dtype=np.float64) for value in logits_per_view]
    if any(value.shape != logits[0].shape for value in logits):
        raise ValueError("Mọi view cần có cùng kích thước và thứ tự ảnh")
    if space == "prob":
        probabilities = np.mean([_softmax(value) for value in logits], axis=0)
    elif space == "logit":
        probabilities = _softmax(np.mean(logits, axis=0))
    else:
        raise ValueError("space phải là 'prob' hoặc 'logit'")
    return probabilities / probabilities.sum(axis=1, keepdims=True)


def ensemble_probs(list_of_probs):
    if not list_of_probs:
        raise ValueError("Cần ít nhất một mô hình")
    probabilities = [np.asarray(value, dtype=np.float64) for value in list_of_probs]
    if any(value.shape != probabilities[0].shape for value in probabilities):
        raise ValueError("Ensemble cần cùng số ảnh, số lớp, thứ tự ảnh")
    mean = np.mean(probabilities, axis=0)
    if (mean < 0).any() or not np.isfinite(mean).all():
        raise ValueError("Xác suất ensemble không hợp lệ")
    return mean / np.clip(mean.sum(axis=1, keepdims=True), 1e-12, None)


def fit_temperature(val_logits, val_labels) -> float:
    logits = np.asarray(val_logits, dtype=np.float64)
    labels = np.asarray(val_labels, dtype=np.int64)
    if logits.ndim != 2 or logits.shape[0] != len(labels) or logits.shape[0] == 0:
        raise ValueError("val_logits và val_labels không khớp")

    def nll(log_t):
        scaled = logits / np.exp(float(log_t))
        shifted = scaled - scaled.max(axis=1, keepdims=True)
        log_norm = np.log(np.exp(shifted).sum(axis=1))
        return float(np.mean(log_norm - shifted[np.arange(len(labels)), labels]))

    try:
        from scipy.optimize import minimize_scalar
        result = minimize_scalar(nll, bounds=(-4.0, 4.0), method="bounded",
                                 options={"xatol": 1e-7})
        log_t = float(result.x) if result.success else 0.0
    except ImportError:
        grid = np.linspace(-4.0, 4.0, 401)
        log_t = float(min(grid, key=nll))
    return float(np.exp(log_t))


def apply_temperature(logits, T: float):
    if not np.isfinite(T) or T <= 0:
        raise ValueError("Nhiệt độ T phải là số dương hữu hạn")
    return _softmax(np.asarray(logits, dtype=np.float64) / float(T))


def fuse_conv_bn(model, example=None, tolerance: float = 1e-5):
    """Return an eval copy with adjacent Conv2d/BatchNorm2d pairs fused."""
    import copy
    import torch
    from torch.nn.utils.fusion import fuse_conv_bn_eval

    fused = copy.deepcopy(model).eval()
    original = copy.deepcopy(model).eval()

    fused_pairs = 0

    def recurse(parent):
        nonlocal fused_pairs
        for child in list(parent.children()):
            recurse(child)
        entries = list(parent._modules.items())
        for index in range(len(entries) - 1):
            name, first = entries[index]
            next_name, second = entries[index + 1]
            if isinstance(first, torch.nn.Conv2d) and isinstance(second, torch.nn.BatchNorm2d):
                parent._modules[name] = fuse_conv_bn_eval(first, second)
                parent._modules[next_name] = torch.nn.Identity()
                fused_pairs += 1

    recurse(fused)
    if fused_pairs == 0:
        raise ValueError("Model không có cặp Conv2d/BatchNorm2d liền kề để gộp")
    if example is None:
        parameter = next(original.parameters())
        example = torch.zeros(1, 3, 224, 224, device=parameter.device, dtype=parameter.dtype)
    else:
        example = example.to(next(original.parameters()).device)
    with torch.inference_mode():
        expected = original(example)
        actual = fused(example)
    difference = float((expected.float() - actual.float()).abs().max().item())
    fused._conv_bn_fuse_max_abs_error = difference
    print(f"Conv-BN fuse pairs={fused_pairs}; max |delta| = {difference:.3g}")
    if not torch.isfinite(torch.tensor(difference)) or difference > tolerance:
        raise RuntimeError(f"Fuse Conv-BN sai số vượt ngưỡng: {difference}")
    return fused
