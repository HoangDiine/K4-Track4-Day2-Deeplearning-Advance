"""Validation-time inference methods; fit every choice on validation data only."""
from __future__ import annotations

import copy
import json
from pathlib import Path

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


# Inference-only experiment runner (I00-I08).
def _save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, default=float), encoding="utf-8"
    )


def _load_eval_model(result: dict, device):
    import torch
    from model import build_model
    model = build_model(result["backbone"], pretrained=False, num_classes=9,
                        init="finetune").to(device)
    checkpoint = torch.load(result["checkpoint"], map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    return model.eval()


def _metrics(scores, labels):
    import eval as ev
    logits = np.asarray(scores, dtype=np.float64)
    shifted = logits - logits.max(axis=1, keepdims=True)
    probs = np.exp(shifted)
    probs /= probs.sum(axis=1, keepdims=True)
    return ev.compute_metrics(labels, probs.argmax(1), probs), probs


def _loader(df, images_dir, img_size, batch_size, workers):
    from dataset import build_transforms, make_loader
    return make_loader(df, images_dir, build_transforms(False, img_size), batch_size,
                       train=False, num_workers=workers)


def _predict(model, loader, device, view=None, amp=False):
    return predict_logits(model, loader, device, view=view, amp=amp)


def _record(method, description, scores, labels, latency, result, view_count=1, note=""):
    metrics, probs = _metrics(scores, labels)
    batch1, batch32 = latency["batch1"], latency["batch32"]
    return {
        "exp_id": method, "method": description, "model": result["exp_id"], "K (views/models)": view_count,
        "macro-F1 val": metrics["macro_f1"], "top-1 val": metrics["top1"], "ECE val": metrics["ece"],
        "p50 batch-1 (ms)": batch1["p50"], "p95 batch-1 (ms)": batch1["p95"],
        "p99 batch-1 (ms)": batch1["p99"], "throughput batch-1 (img/s)": batch1["images_per_s"],
        "p50 batch-32 (ms)": batch32["p50"], "p95 batch-32 (ms)": batch32["p95"],
        "p99 batch-32 (ms)": batch32["p99"], "throughput batch-32 (img/s)": batch32["images_per_s"],
        "GPU": batch1["gpu"], "dtype": batch1["dtype"], "input size": batch1["img_size"],
        "warmup": batch1["warmup"], "measurements": batch1["n"],
        "relative cost": batch1["p50"], "note": note,
        "_latency": latency, "_probabilities": probs, "_scores": np.asarray(scores),
    }


def _paired_latency(benchmark, **kwargs):
    """Measure the same inference path at robot batch 1 and throughput batch 32."""
    return {
        "batch1": benchmark(batch_size=1, **kwargs),
        "batch32": benchmark(batch_size=32, **kwargs),
    }


def sweep_inference(selected: dict, top_backbones: list[dict], images_dir: str | Path,
                    labels_dir: str | Path, output_dir: str | Path,
                    batch_size: int = 32, num_workers: int = 2) -> dict:
    """Compare I00-I08 on validation only; benchmark batch 1 and batch 32 separately."""
    import torch
    from benchmark import ensemble_latency as benchmark_ensemble_latency, latency_report, tta_latency
    from dataset import load_split
    output_dir = Path(output_dir)
    _train_df, val_df, _test_df = load_split(labels_dir, fold=0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_name = str(device)
    model = _load_eval_model(selected, device)
    val_loader = _loader(val_df, images_dir, 224, batch_size, num_workers)
    names, labels, base_logits = _predict(model, val_loader, device)
    base_latency = _paired_latency(latency_report, model=model, img_size=224, dtype="fp32",
                                   device=device_name, warmup=10, iters=50)
    rows = [_record("I00", "Standard 1-view FP32", base_logits, labels, base_latency, selected,
                    note="Resize 256 + CenterCrop 224; model.eval()")]

    flip_names, flip_labels, flip_logits = _predict(model, val_loader, device, view_hflip)
    if names != flip_names or not np.array_equal(labels, flip_labels):
        raise RuntimeError("TTA flip làm thay đổi thứ tự ảnh")
    flip_latency = _paired_latency(tta_latency, model=model, k_views=2, img_size=224,
                                   dtype="fp32", device=device_name, warmup=10, iters=50)
    flip_probs = aggregate_views([base_logits, flip_logits], "prob")
    rows.append(_record("I01", "Horizontal flip TTA, K=2, mean probabilities",
                        np.log(np.clip(flip_probs, 1e-12, 1)), labels, flip_latency,
                        selected, view_count=2, note="original + torch.flip(..., dims=(-1,))"))

    crop_loader = _loader(val_df, images_dir, 256, batch_size, num_workers)
    crop_views = [[] for _ in range(5)]
    crop_names, crop_label_batches = [], []
    model.eval()
    with torch.inference_mode():
        for images, targets, file_names in crop_loader:
            images = images.to(device, non_blocking=True)
            crop_names.extend(list(file_names))
            crop_label_batches.append(targets.cpu().numpy())
            for index, crop in enumerate(views_multicrop(images, 224)):
                crop_views[index].append(model(crop).float().cpu().numpy())
    crop_logits = [np.concatenate(batches) for batches in crop_views]
    crop_labels = np.concatenate(crop_label_batches)
    if crop_names != names or not np.array_equal(crop_labels, labels):
        raise RuntimeError("Multi-crop views làm thay đổi thứ tự ảnh hoặc nhãn")
    crop_latency = _paired_latency(tta_latency, model=model, k_views=5, img_size=224,
                                   dtype="fp32", device=device_name, warmup=10, iters=50)
    crop_prob = aggregate_views(crop_logits, "prob")
    rows.append(_record("I02", "Multi-crop TTA, K=5, mean probabilities",
                        np.log(np.clip(crop_prob, 1e-12, 1)), labels, crop_latency,
                        selected, view_count=5, note="4 corners + center; 224 crops from 256 input"))
    rows.append(_record("I03a", "I03 aggregation comparison: mean probabilities",
                        np.log(np.clip(crop_prob, 1e-12, 1)), labels, crop_latency,
                        selected, view_count=5, note="mean(softmax(z_k)) over I02 views"))
    rows.append(_record("I03b", "I03 aggregation comparison: mean logits",
                        np.mean(crop_logits, axis=0), labels, crop_latency,
                        selected, view_count=5, note="softmax(mean(z_k)) over the same I02 views"))

    temperature = fit_temperature(base_logits, labels)
    temp_prob = apply_temperature(base_logits, temperature)
    temp_scores = np.log(np.clip(temp_prob, 1e-12, 1))
    if not np.array_equal(np.argmax(base_logits, axis=1), np.argmax(temp_prob, axis=1)):
        raise RuntimeError("Temperature scaling changed Top-1; check the implementation")
    temp_metrics, _ = _metrics(temp_scores, labels)
    temp_row = _record("I07", f"Temperature scaling T={temperature:.4f}", temp_scores,
                       labels, base_latency, selected,
                       note=f"T fit on validation only; ECE {_metrics(base_logits, labels)[0]['ece']:.4f} -> {temp_metrics['ece']:.4f}")
    temp_row["temperature"] = temperature
    temp_row["ECE before"] = _metrics(base_logits, labels)[0]["ece"]
    temp_row["ECE after"] = temp_metrics["ece"]
    temp_row["Top-1 unchanged"] = True
    rows.append(temp_row)

    try:
        highres_loader = _loader(val_df, images_dir, 256, batch_size, num_workers)
        highres_names, highres_labels, highres_logits = _predict(model, highres_loader, device)
        highres_latency = _paired_latency(latency_report, model=model, img_size=256, dtype="fp32",
                                          device=device_name, warmup=10, iters=50)
        if highres_names == names and np.array_equal(highres_labels, labels):
            rows.append(_record("I04", "Test-time resolution / FixRes, 256", highres_logits,
                                labels, highres_latency, selected, note="Train 224; validation input 256"))
    except (RuntimeError, ValueError, TypeError) as exc:
        print(f"I04 resolution 256 skipped: {exc}")

    # Ensemble the selected recipe with the next one or two best different backbones.
    ensemble_models, ensemble_rows, ensemble_logits = [], [], []
    for candidate in top_backbones:
        if candidate["backbone"] == selected["backbone"]:
            continue
        candidate_model = None
        try:
            candidate_model = _load_eval_model(candidate, device)
            e_names, e_labels, e_logits = _predict(candidate_model, val_loader, device)
            if e_names != names or not np.array_equal(e_labels, labels):
                raise RuntimeError("Ensemble checkpoint validation order differs from I00")
            ensemble_models.append(candidate_model)
            ensemble_rows.append(candidate)
            ensemble_logits.append(e_logits)
            candidate_model = None  # Retain successful models until the ensemble benchmark ends.
            if len(ensemble_models) == 2:  # selected model + at most two others = K 2–3
                break
        except (RuntimeError, ValueError, TypeError, KeyError) as exc:
            print(f"I05 {candidate.get('backbone', 'unknown')} skipped: {exc}")
        finally:
            if candidate_model is not None:
                del candidate_model
                if device.type == "cuda":
                    torch.cuda.empty_cache()
    if ensemble_models:
        ensemble_prob = aggregate_views([base_logits, *ensemble_logits], "prob")
        ensemble_models_all = [model, *ensemble_models]
        ensemble_latency = _paired_latency(
            benchmark_ensemble_latency, models=ensemble_models_all, img_size=224, dtype="fp32",
            device=device_name, warmup=10, iters=50)
        ensemble_names = [selected["backbone"], *[row["backbone"] for row in ensemble_rows]]
        rows.append(_record("I05", f"{len(ensemble_models_all)}-model probability ensemble",
                            np.log(np.clip(ensemble_prob, 1e-12, 1)), labels,
                            ensemble_latency, selected, view_count=len(ensemble_models_all),
                            note=" + ".join(ensemble_names)))
        del ensemble_models_all
        del ensemble_models
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Try the chosen checkpoint first, then other trained checkpoints if its BN
    # layout cannot be fused within the required 1e-5 equivalence tolerance.
    bn_candidates = [(selected, model)]
    bn_candidates.extend((candidate, None) for candidate in top_backbones
                         if candidate.get("checkpoint") != selected.get("checkpoint"))
    i08_done = False
    for bn_result, candidate_model in bn_candidates:
        try:
            if candidate_model is None:
                candidate_model = _load_eval_model(bn_result, device)
            if not any(isinstance(module, torch.nn.BatchNorm2d) for module in candidate_model.modules()):
                continue
            sample = next(iter(val_loader))[0][:1].to(device, non_blocking=True)
            fused_model = fuse_conv_bn(candidate_model, example=sample, tolerance=1e-5).eval()
            fusion_error = fused_model._conv_bn_fuse_max_abs_error
            if device.type == "cuda":
                optimized_model = copy.deepcopy(fused_model).half().eval()
                dtype, amp = "fp16", True
                method = "Conv-BN fused + FP16 inference"
                note = f"max fusion error={fusion_error:.3g}; tolerance=1e-5"
            else:
                optimized_model = fused_model
                dtype, amp = "fp32", False
                method = "Conv-BN fused; FP16 unavailable on CPU"
                note = f"max fusion error={fusion_error:.3g}; tolerance=1e-5; FP16 skipped on CPU"
            fused_names, fused_labels, fused_logits = _predict(optimized_model, val_loader, device, amp=amp)
            fused_latency = _paired_latency(latency_report, model=optimized_model, img_size=224,
                                            dtype=dtype, device=device_name, warmup=10, iters=50)
            if fused_names != names or not np.array_equal(fused_labels, labels):
                raise RuntimeError("I08 validation order differs from I00")
            rows.append(_record("I08", method, fused_logits,
                                labels, fused_latency, bn_result, note=note))
            rows[-1]["fusion max abs error"] = fusion_error
            i08_done = True
            break
        except (RuntimeError, ValueError, TypeError) as exc:
            print(f"I08 {bn_result['backbone']} skipped: {exc}")
        finally:
            if candidate_model is not None and candidate_model is not model:
                del candidate_model
                if device.type == "cuda":
                    torch.cuda.empty_cache()
    if not i08_done:
        print("I08 skipped: no BatchNorm checkpoint passed the 1e-5 fusion check")

    baseline = next(row for row in rows if row["exp_id"] == "I00")
    for row in rows:
        row["relative cost"] = (row["p50 batch-1 (ms)"] / baseline["p50 batch-1 (ms)"]
                                if baseline["p50 batch-1 (ms)"] else np.nan)

    # F01's training runner can reproduce a single checkpoint with TTA/FixRes.
    # I05 and I08 remain fully measured validation comparisons because they alter the model set/dtype.
    final_candidates = [row for row in rows if row["exp_id"] in {"I00", "I01", "I02", "I03a", "I03b", "I04"}]
    chosen = max(final_candidates, key=lambda row: row["macro-F1 val"])
    realtime_candidates = [row for row in rows if row["p95 batch-1 (ms)"] <= 100]
    realtime_choice = max(realtime_candidates, key=lambda row: row["macro-F1 val"]) if realtime_candidates else None
    chosen_temperature = fit_temperature(chosen["_scores"], labels)
    calibrated_chosen = apply_temperature(chosen["_scores"], chosen_temperature)
    chosen_ece_after = _metrics(np.log(np.clip(calibrated_chosen, 1e-12, 1)), labels)[0]["ece"]
    chosen_ece_before = chosen["ECE val"]
    apply_temperature_final = chosen_ece_after < chosen_ece_before
    selection = {
        "inference_schema": 3,
        "training": selected, "inference_exp_id": chosen["exp_id"],
        "tta": "hflip" if chosen["exp_id"] == "I01" else
               "multicrop" if chosen["exp_id"] in {"I02", "I03a", "I03b"} else "none",
        "inference_space": "logit" if chosen["exp_id"] == "I03b" else "prob",
        "inference_img_size": 256 if chosen["exp_id"] == "I04" else 224,
        "tta_crop_size": 224, "temperature_scale": bool(apply_temperature_final),
        "temperature": chosen_temperature, "validation_macro_f1": chosen["macro-F1 val"],
        "validation_ece_before": chosen_ece_before,
        "validation_ece_after": chosen_ece_after,
        "realtime_candidate": ({"exp_id": realtime_choice["exp_id"],
                                "macro_f1_val": realtime_choice["macro-F1 val"],
                                "p95_batch1_ms": realtime_choice["p95 batch-1 (ms)"]}
                               if realtime_choice else None),
        "selection_scope_note": "I05 and I08 are measured on validation; the final runner selects a single-checkpoint method.",
        "rows": [{k: v for k, v in row.items() if not k.startswith("_")} for row in rows],
        "val_names": names,
    }
    _save_json(Path(output_dir) / "inference_study.json", selection)
    # Keep arrays out of JSON but persist all validation predictions for auditability.
    save_dir = Path(output_dir) / "inference_logits"
    save_dir.mkdir(parents=True, exist_ok=True)
    for row in rows:
        np.savez_compressed(save_dir / f"{row['exp_id']}.npz", filenames=np.asarray(names),
                            y_true=labels, probs=row["_probabilities"], scores=row["_scores"])
    return selection


def run_inference_only(checkpoint_path: str | Path, backbone: str,
                       images_dir: str | Path, labels_dir: str | Path,
                       output_dir: str | Path,
                       ensemble_checkpoints: list[dict] | None = None,
                       batch_size: int = 32, num_workers: int = 2) -> dict:
    """Run I00-I08 from existing checkpoint(s) without entering any training runner."""
    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint không tồn tại: {checkpoint_path}")
    if not backbone:
        raise ValueError("Cần tên backbone đúng với checkpoint đã huấn luyện")
    candidates = [{"exp_id": "I00", "backbone": backbone,
                   "checkpoint": str(checkpoint_path)}]
    ensemble_checkpoints = ensemble_checkpoints or []
    if len(ensemble_checkpoints) > 2:
        raise ValueError("I05 hỗ trợ tối đa hai checkpoint phụ (ensemble tổng 2–3 mô hình)")
    for index, item in enumerate(ensemble_checkpoints, start=1):
        candidate_path = Path(item["checkpoint"]).expanduser().resolve()
        if not candidate_path.is_file():
            raise FileNotFoundError(f"Ensemble checkpoint không tồn tại: {candidate_path}")
        if not item.get("backbone"):
            raise ValueError("Mỗi checkpoint ensemble cần trường backbone")
        candidates.append({"exp_id": item.get("exp_id", f"E{index:02d}"),
                           "backbone": item["backbone"],
                           "checkpoint": str(candidate_path)})
    return sweep_inference(candidates[0], candidates, images_dir, labels_dir,
                           output_dir, batch_size=batch_size, num_workers=num_workers)
