"""Reproducible orchestration and artifact generation for the DeepWeeds lab.

All model selection uses fold-0 validation data. Test predictions are created only
after the final model and inference method have been selected.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
import re
import shutil
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

BACKBONES = [
    ("B01", "resnet50"),
    ("B02", "resnext50_32x4d"),
    ("B03", "convnext_tiny"),
    ("B04", "deit_small_patch16_224"),
    ("B05", "mobilenetv3_large_100"),
]


def prepare_data_and_eda(images_dir: str | Path, labels_dir: str | Path,
                        output_dir: str | Path = ROOT) -> dict:
    """Run fold integrity checks and save train-only class/sample EDA figures."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from dataset import CLASS_NAMES, DeepWeedsDataset, build_transforms, check_split, load_split
    output_dir = Path(output_dir).resolve()
    train_df, val_df, test_df = load_split(labels_dir, fold=0)
    report = check_split(train_df, val_df, test_df, images_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _save_json(output_dir / "split_check.json", report)

    counts = pd.DataFrame({name: [int((frame["Label"] == label).sum()) for label in range(9)]
                           for name, frame in (("train", train_df), ("val", val_df), ("test", test_df))},
                          index=CLASS_NAMES)
    ax = counts.plot.bar(figsize=(13, 5), color=["#4472C4", "#ED7D31", "#70AD47"])
    ax.set(title="DeepWeeds fold 0 — class counts", xlabel="Class", ylabel="Images")
    ax.tick_params(axis="x", labelrotation=35)
    ax.legend(title="Split")
    plt.tight_layout()
    plt.savefig(output_dir / "eda_class_distribution.png", dpi=170)
    plt.close()
    counts.to_csv(output_dir / "eda_class_counts.csv", index_label="class")

    examples = train_df.groupby("Label", group_keys=False).head(3).reset_index(drop=True)
    dataset = DeepWeedsDataset(examples, images_dir, build_transforms(True, 160, "basic"))
    fig, axes = plt.subplots(9, 3, figsize=(9, 21))
    mean = np.array([0.485, 0.456, 0.406])[:, None, None]
    std = np.array([0.229, 0.224, 0.225])[:, None, None]
    for row in range(9):
        group = examples[examples["Label"] == row]
        for column, row_idx in enumerate(group.index[:3]):
            image, label, filename = dataset[int(row_idx)]
            image = (image.numpy() * std + mean).clip(0, 1).transpose(1, 2, 0)
            axes[row, column].imshow(image)
            axes[row, column].set_title(f"{CLASS_NAMES[label]}\n{filename}", fontsize=7)
            axes[row, column].axis("off")
    fig.suptitle("Training examples by class (random augmentation shown)")
    fig.tight_layout()
    fig.savefig(output_dir / "eda_train_examples.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return report


def pipeline_smoke(images_dir: str | Path, labels_dir: str | Path, seed: int = 0,
                   output_dir: str | Path = ROOT) -> dict:
    """Check initial head loss, tensor labels, gradient flow, and small-batch overfit."""
    import torch
    from dataset import DeepWeedsDataset, build_transforms, load_split
    from model import build_model
    from train import set_seed
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("Pipeline smoke check cần GPU Colab")
    set_seed(seed)
    train_df, _val_df, _test_df = load_split(labels_dir, fold=0)
    one_each = train_df.groupby("Label", group_keys=False).head(1).reset_index(drop=True)
    dataset = DeepWeedsDataset(one_each, images_dir, build_transforms(False, 160))
    images = torch.stack([dataset[i][0] for i in range(len(dataset))]).to(device)
    labels = torch.as_tensor(one_each["Label"].to_numpy(), device=device)
    model = build_model("resnet18", pretrained=True, num_classes=9, init="finetune").to(device)
    model.train()
    criterion = torch.nn.CrossEntropyLoss()
    with torch.no_grad():
        initial = float(criterion(model(images), labels))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    final = float("inf")
    steps = 0
    for step in range(1, 201):
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(images), labels)
        loss.backward()
        optimizer.step()
        steps = step
        if step % 10 == 0:
            model.eval()
            with torch.inference_mode():
                final = float(criterion(model(images), labels))
            model.train()
            if final < 0.1:
                break
    if abs(initial - np.log(9)) > 0.7:
        raise AssertionError(f"Initial head loss {initial:.3f} differs too much from ln(9)")
    if final >= 0.1:
        raise AssertionError(f"Small-batch did not overfit near zero after {steps} steps: {initial:.4f} -> {final:.4f}")
    result = {"seed": seed, "backbone": "resnet18", "n_images": len(one_each),
              "initial_ce": initial, "final_ce": final, "optimizer_steps": steps,
              "device": torch.cuda.get_device_name(0)}
    _save_json(Path(output_dir) / "pipeline_smoke.json", result)
    return result


def _json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _save_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=float), encoding="utf-8")


def _markdown_table(headers, rows):
    lines = ["| " + " | ".join(headers) + " |",
             "| " + " | ".join("---" for _ in headers) + " |"]
    lines.extend("| " + " | ".join(str(value).replace("|", "\\|") for value in row) + " |"
                 for row in rows)
    return "\n".join(lines)


def _cached_run(cfg):
    from train import run, run_dir, pred_path
    output = run_dir(cfg) / "result.json"
    test_done = pred_path(cfg, "test").is_file() if cfg.save_test_predictions else True
    if output.is_file() and test_done:
        return _json(output)
    return run(cfg)


def run_backbone_and_recipe_study(images_dir: str | Path, labels_dir: str | Path,
                                  output_dir: str | Path = ROOT, epochs: int = 10,
                                  batch_size: int = 32, num_workers: int = 2) -> dict:
    """Run the five-backbone comparison and controlled training recipe ablations."""
    import torch
    from train import Config
    if not torch.cuda.is_available():
        raise RuntimeError("Bài lab cần GPU CUDA; hãy chọn GPU runtime trong Colab.")
    output_dir = Path(output_dir).resolve()
    images_dir, labels_dir = str(Path(images_dir).resolve()), str(Path(labels_dir).resolve())
    common = dict(epochs=epochs, batch_size=batch_size, num_workers=num_workers,
                  images_dir=images_dir, labels_dir=labels_dir,
                  out_dir=str(output_dir / "runs"), pred_dir=str(output_dir / "predictions"),
                  curves_dir=str(output_dir / "curves"), amp=True, img_size=224)
    record_file = output_dir / "study.json"
    if record_file.is_file():
        cached = _json(record_file)
        if cached.get("epochs") == epochs and cached.get("batch_size") == batch_size:
            b_rows = cached.get("backbones", [])
            selected_checkpoint = cached.get("selected_recipe", {}).get("checkpoint")
            if (len(b_rows) == len(BACKBONES) and all(Path(r["checkpoint"]).is_file() for r in b_rows)
                    and selected_checkpoint and Path(selected_checkpoint).is_file()):
                if cached.get("inference", {}).get("inference_schema") == 3:
                    return cached
                cached["inference"] = sweep_inference(
                    cached["selected_recipe"], b_rows, images_dir, labels_dir,
                    output_dir, batch_size=batch_size, num_workers=num_workers)
                _save_json(record_file, cached)
                return cached

    backbone_rows = []
    for exp_id, backbone in BACKBONES:
        result = _cached_run(Config(exp_id=exp_id, backbone=backbone, seed=0, **common))
        backbone_rows.append(result)
        torch.cuda.empty_cache()
    backbone_rows.sort(key=lambda row: row["macro_f1_val"], reverse=True)
    selected_backbone = backbone_rows[0]["backbone"]

    baseline = _cached_run(Config(exp_id="T00", backbone=selected_backbone, seed=0, **common))
    variants = [
        ("T01", {"init": "scratch", "pretrained": False}, "A", "khởi tạo ImageNet -> ngẫu nhiên"),
        ("T02", {"aug": "color"}, "B", "basic -> color jitter"),
        ("T03", {"loss": "ls", "label_smoothing": 0.1}, "C", "CE -> label smoothing 0.1"),
        ("T04", {"sampler": "balanced"}, "D", "sampler thường -> balanced"),
        ("T05", {"ema_decay": 0.99}, "F", "không EMA -> EMA 0.99"),
    ]
    training_rows = [baseline]
    by_exp = {"T00": baseline}
    for exp_id, override, axis, difference in variants:
        result = _cached_run(Config(exp_id=exp_id, backbone=selected_backbone, seed=0, **common, **override))
        result["axis"] = axis
        result["difference"] = difference
        training_rows.append(result)
        by_exp[exp_id] = result
        torch.cuda.empty_cache()

    # Select each factor on validation independently, then measure one combined recipe.
    aug_choice = "color" if by_exp["T02"]["macro_f1_val"] > baseline["macro_f1_val"] else "basic"
    loss_choice, smoothing = (("ls", 0.1) if by_exp["T03"]["macro_f1_val"] > baseline["macro_f1_val"]
                              else ("ce", 0.0))
    sampler_choice = "balanced" if by_exp["T04"]["macro_f1_val"] > baseline["macro_f1_val"] else None
    ema_choice = 0.99 if by_exp["T05"]["macro_f1_val"] > baseline["macro_f1_val"] else None
    combination = _cached_run(Config(exp_id="T06", backbone=selected_backbone, seed=0, **common,
                                     aug=aug_choice, loss=loss_choice, label_smoothing=smoothing,
                                     sampler=sampler_choice, ema_decay=ema_choice))
    combination["axis"] = "Combination"
    combination["difference"] = f"aug={aug_choice}; loss={loss_choice}; sampler={sampler_choice}; EMA={ema_choice}"
    training_rows.append(combination)
    selected_recipe = max(training_rows, key=lambda row: row["macro_f1_val"])
    selected_inference = sweep_inference(selected_recipe, backbone_rows, images_dir, labels_dir,
                                         output_dir, batch_size=batch_size, num_workers=num_workers)
    result = {
        "epochs": epochs, "batch_size": batch_size,
        "backbones": backbone_rows, "selected_backbone": selected_backbone,
        "baseline": baseline, "training": training_rows,
        "selected_recipe": selected_recipe, "inference": selected_inference,
    }
    _save_json(record_file, result)
    return result


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
    from inference import predict_logits
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
    from inference import aggregate_views, apply_temperature, fit_temperature, fuse_conv_bn, views_multicrop, view_hflip
    output_dir = Path(output_dir)
    train_df, val_df, _test_df = load_split(labels_dir, fold=0)
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
    import torch
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


def run_final_and_score(study: dict, images_dir: str | Path, labels_dir: str | Path,
                        output_dir: str | Path = ROOT, seeds=(0, 1, 2),
                        num_workers: int = 2) -> dict:
    """Retrain selected and baseline configurations; touch test once per seed."""
    import torch
    from train import Config, run_dir
    if not torch.cuda.is_available():
        raise RuntimeError("Bài lab cần GPU CUDA; không chạy phần test trên CPU.")
    output_dir = Path(output_dir).resolve()
    selected = study["selected_recipe"]
    choice = study["inference"]
    selected_cfg = _json(run_dir(Config(exp_id=selected["exp_id"], seed=0,
                                       out_dir=str(output_dir / "runs"))) / "config.json")
    base_fields = dict(selected_cfg)
    for seed in seeds:
        final_args = {**base_fields, "exp_id": "F01", "seed": seed,
                      "images_dir": str(Path(images_dir).resolve()),
                      "labels_dir": str(Path(labels_dir).resolve()),
                      "out_dir": str(output_dir / "runs"), "pred_dir": str(output_dir / "predictions"),
                      "curves_dir": str(output_dir / "curves"), "save_test_predictions": True,
                      "tta": choice["tta"], "inference_space": choice["inference_space"],
                      "inference_img_size": choice["inference_img_size"],
                      "tta_crop_size": choice["tta_crop_size"],
                      "temperature_scale": choice["temperature_scale"]}
        _cached_run(Config(**final_args))

        baseline_args = {**base_fields, "exp_id": "T00", "seed": seed,
                         "images_dir": str(Path(images_dir).resolve()),
                         "labels_dir": str(Path(labels_dir).resolve()),
                         "out_dir": str(output_dir / "final_runs"),
                         "pred_dir": str(output_dir / "predictions"),
                         "curves_dir": str(output_dir / "curves"),
                         "init": "finetune", "pretrained": True, "aug": "basic",
                         "sampler": None, "mix": None, "loss": "ce", "label_smoothing": 0.0,
                         "class_weight_beta": None, "ema_decay": None,
                         "temperature_scale": False, "tta": "none", "inference_space": "prob",
                         "inference_img_size": 224, "save_test_predictions": True}
        _cached_run(Config(**baseline_args))

    results = score_and_write(output_dir, images_dir, labels_dir, study, seeds)
    return results


def _score_files(exp_id, base_dir, test_csv):
    import eval as ev
    rows, matrices = [], []
    for path in sorted(Path(base_dir).glob(f"{exp_id}_seed*_test.csv")):
        df = pd.read_csv(path)
        probability_cols = [f"p{i}" for i in range(9)]
        probs = df[probability_cols].to_numpy(dtype=float)
        metrics = ev.compute_metrics(df["y_true"].to_numpy(), df["y_pred"].to_numpy(), probs)
        metrics.update({"seed": int(path.stem.split("_seed")[-1].split("_")[0]), "path": str(path)})
        rows.append(metrics)
        matrices.append(metrics["confusion"])
    if len(rows) < 3:
        raise RuntimeError(f"{exp_id}: cần ít nhất 3 seed test; chỉ tìm thấy {len(rows)}")
    return rows, np.sum(matrices, axis=0)


def score_and_write(output_dir: str | Path, images_dir: str | Path, labels_dir: str | Path,
                    study: dict, seeds=(0, 1, 2)) -> dict:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import eval as ev
    from openpyxl.styles import Font, PatternFill
    output_dir = Path(output_dir).resolve()
    pred_dir = output_dir / "predictions"
    test_csv = str(Path(labels_dir) / "test_subset0.csv")
    labels = pd.read_csv(Path(labels_dir) / "labels.csv").sort_values("Label")["Species"].astype(str).tolist()
    final_rows, final_cm = _score_files("F01", pred_dir, test_csv)
    baseline_rows, baseline_cm = _score_files("T00", pred_dir, test_csv)

    # Run the supplied scorer and rubric calculator; never alter eval.py.
    out_eval = output_dir / "eval_out"
    rc_final = ev.main(["score", "--pred", str(pred_dir / "F01_seed*_test.csv"),
                        "--test-csv", test_csv, "--labels", str(Path(labels_dir) / "labels.csv"),
                        "--tag", "F01", "--out", str(out_eval)])
    rc_base = ev.main(["score", "--pred", str(pred_dir / "T00_seed*_test.csv"),
                       "--test-csv", test_csv, "--labels", str(Path(labels_dir) / "labels.csv"),
                       "--tag", "T00", "--out", str(out_eval)])
    if rc_final or rc_base:
        raise RuntimeError(f"eval.py score returned {rc_final}/{rc_base}")
    uncal = list(pred_dir.glob("F01uncal_seed*_test.csv"))
    final_val = list(pred_dir.glob("F01_seed*_val.csv"))
    grade_args = ["grade", "--final", str(pred_dir / "F01_seed*_test.csv"),
                  "--baseline", str(pred_dir / "T00_seed*_test.csv"),
                  "--test-csv", test_csv, "--labels", str(Path(labels_dir) / "labels.csv"),
                  "--out", str(out_eval)]
    if uncal:
        grade_args.extend(["--uncal", str(pred_dir / "F01uncal_seed*_test.csv")])
    if final_val:
        grade_args.extend(["--final-val", str(pred_dir / "F01_seed*_val.csv")])
    best_latency = max((row.get("p95 batch-1 (ms)") or 0) for row in study["inference"]["rows"]
                       if row.get("exp_id") == study["inference"]["inference_exp_id"])
    grade_args.extend(["--latency-p95-ms", str(best_latency), "--latency-method", "proper"])
    if ev.main(grade_args) != 0:
        raise RuntimeError("eval.py grade failed")

    names_final = pd.read_csv(final_rows[0]["path"])["Filename"].tolist()
    test_df = pd.read_csv(test_csv)
    if names_final != test_df["Filename"].astype(str).tolist():
        raise RuntimeError("Predictions do not follow the test CSV order")
    summary_rows, per_class_rows = [], []
    for label, rows, confusion, tag in (("F01", final_rows, final_cm, "Final"),
                                        ("T00", baseline_rows, baseline_cm, "Baseline")):
        for key, column in (("macro_f1", "macro-F1 test"), ("top1", "top-1 test"), ("ece", "ECE test")):
            values = [row[key] for row in rows]
            summary_rows.append({"exp_id": label, "seed": "mean ± std", "metric": column,
                                 "mean": float(np.mean(values)), "std (ddof=1)": float(np.std(values, ddof=1))})
        for index, name in enumerate(labels):
            precision, recall, f1_values = [], [], []
            for row in rows:
                frame = pd.read_csv(row["path"])
                values = ev.compute_metrics(frame["y_true"].to_numpy(), frame["y_pred"].to_numpy(),
                                            frame[[f"p{i}" for i in range(9)]].to_numpy())
                precision.append(values["precision"][index])
                recall.append(values["recall"][index])
                f1_values.append(values["f1"][index])
            per_class_rows.append({"configuration": label, "class": name,
                                   "test support": int((test_df["Label"] == index).sum()),
                                   "precision mean": float(np.mean(precision)),
                                   "recall mean": float(np.mean(recall)), "F1 mean": float(np.mean(f1_values)),
                                   "precision std": float(np.std(precision, ddof=1)),
                                   "recall std": float(np.std(recall, ddof=1)),
                                   "F1 std": float(np.std(f1_values, ddof=1))})

    final_mean = float(np.mean([row["macro_f1"] for row in final_rows]))
    final_std = float(np.std([row["macro_f1"] for row in final_rows], ddof=1))
    base_mean = float(np.mean([row["macro_f1"] for row in baseline_rows]))
    base_std = float(np.std([row["macro_f1"] for row in baseline_rows], ddof=1))
    delta = final_mean - base_mean
    noise = max(final_std, base_std)
    evidence = ("lớn hơn độ lệch chuẩn lớn nhất giữa hai nhóm" if delta > noise
                else "chưa vượt độ lệch chuẩn lớn nhất giữa hai nhóm")
    matrix = final_cm
    fig, ax = plt.subplots(figsize=(10, 8))
    image = ax.imshow(matrix, cmap="Blues")
    ax.set(xticks=range(9), yticks=range(9), xticklabels=labels, yticklabels=labels,
           xlabel="Predicted label", ylabel="True label", title="F01 — summed test confusion matrix (3 seeds)")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(output_dir / "confusion_matrix_test.png", dpi=170)
    plt.close(fig)

    from dataset import _image_lookup
    lookup = _image_lookup(Path(images_dir))
    final_seed0 = next((row for row in final_rows if row["seed"] == 0), final_rows[0])
    prediction_frame = pd.read_csv(final_seed0["path"])
    mistakes = prediction_frame[prediction_frame["y_true"] != prediction_frame["y_pred"]].head(12)
    fig, axes = plt.subplots(3, 4, figsize=(14, 10))
    for ax in axes.flat:
        ax.axis("off")
    for ax, (_, mistake) in zip(axes.flat, mistakes.iterrows()):
        filename = str(mistake["Filename"])
        image_path = lookup.get(filename)
        if image_path is not None:
            from PIL import Image
            with Image.open(image_path) as image:
                ax.imshow(image.convert("RGB"))
            ax.set_title(f"true: {labels[int(mistake['y_true'])]}\npred: {labels[int(mistake['y_pred'])]}",
                         fontsize=8)
            ax.axis("off")
    if mistakes.empty:
        axes.flat[0].text(0.1, 0.5, "No errors in seed 0", fontsize=12)
    fig.suptitle("F01 test errors — seed 0 examples")
    fig.tight_layout()
    fig.savefig(output_dir / "misclassified_test.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    offdiag = matrix.copy()
    np.fill_diagonal(offdiag, 0)
    i, j = np.unravel_index(np.argmax(offdiag), offdiag.shape)
    best_backbone = max(study["backbones"], key=lambda row: row["macro_f1_val"])
    best_recipe = study["selected_recipe"]
    inference = study["inference"]
    selected_latency = next((row.get("p95 batch-1 (ms)") for row in inference["rows"]
                             if row.get("exp_id") == inference["inference_exp_id"]), None)
    latency_statement = (f"p95 batch-1 = {selected_latency:.2f} ms; "
                        f"{'đạt' if selected_latency is not None and selected_latency <= 100 else 'chưa đạt'} ngưỡng 100 ms")
    realtime = inference.get("realtime_candidate")
    realtime_statement = (f"Cấu hình đáp ứng p95 ≤ 100 ms ở batch 1: `{realtime['exp_id']}` "
                          f"(p95 {realtime['p95_batch1_ms']:.2f} ms; val macro-F1 {realtime['macro_f1_val']:.4f})."
                          if realtime else "Chưa có cấu hình nào đạt p95 ≤ 100 ms ở batch 1 trên GPU đo được.")
    backbone_table = _markdown_table(
        ["Backbone", "Params (M)", "GMAC", "Val macro-F1", "Val top-1", "sec/epoch"],
        [[r["backbone"], f"{r['params_m']:.2f}", f"{r['gmac']:.3f}",
          f"{r['macro_f1_val']:.4f}", f"{r['top1_val']:.4f}",
          f"{r['train_seconds_per_epoch']:.2f}"] for r in study["backbones"]])
    training_table = _markdown_table(
        ["ID", "Axis", "Change", "Val macro-F1", "Δ vs T00"],
        [[r["exp_id"], r.get("axis", "Baseline"), r.get("difference", "baseline"),
          f"{r['macro_f1_val']:.4f}", f"{r['macro_f1_val']-study['baseline']['macro_f1_val']:+.4f}"]
         for r in study["training"]])
    inference_table = _markdown_table(
        ["ID", "Method", "Val macro-F1", "ECE", "p95 ms (B1)", "p95 ms (B32)",
         "throughput (B32 img/s)", "Cost/I00"],
        [[r["exp_id"], r["method"], f"{r['macro-F1 val']:.4f}", f"{r['ECE val']:.4f}",
          f"{r['p95 batch-1 (ms)']:.2f}", f"{r['p95 batch-32 (ms)']:.2f}",
          f"{r['throughput batch-32 (img/s)']:.1f}", f"{r['relative cost']:.2f}×"]
         for r in inference["rows"]])
    difficult_rows = [row for row in per_class_rows
                      if row["configuration"] == "F01" and row["class"] in {"Chinee Apple", "Snake Weed"}]
    perclass_table = _markdown_table(
        ["Class", "Support", "Precision", "Recall", "F1"],
        [[r["class"], r["test support"], f"{r['precision mean']:.4f}",
          f"{r['recall mean']:.4f}", f"{r['F1 mean']:.4f}"] for r in difficult_rows])
    smoke_path = output_dir / "pipeline_smoke.json"
    smoke_line = ""
    if smoke_path.is_file():
        smoke = _json(smoke_path)
        smoke_line = (f"- Pipeline overfit: CE đầu {smoke['initial_ce']:.4f}, cuối "
                      f"{smoke['final_ce']:.4f} sau {smoke['optimizer_steps']} bước trên "
                      f"{smoke['n_images']} ảnh ({smoke['device']}).\n")
    report = f"""# DeepWeeds — kết quả Lab Day 2

## 1. Tóm tắt

- Dữ liệu dùng fold 0, lựa chọn dựa trên validation; test được chấm một lần cho mỗi seed.
- Backbone được chọn: `{best_backbone['backbone']}` (macro-F1 val {best_backbone['macro_f1_val']:.4f}).
- Công thức cuối: `{best_recipe['exp_id']}`; suy luận `{inference['inference_exp_id']}`.
- Macro-F1 test của F01: **{final_mean:.4f} ± {final_std:.4f}** (n=3); baseline T00: {base_mean:.4f} ± {base_std:.4f}.
- Chênh lệch F01−T00: {final_mean-base_mean:+.4f}; coi là bằng chứng rõ khi vượt độ phân tán giữa các seed.

## 2. Dữ liệu và thiết lập

- DeepWeeds, split tác giả fold 0; {int(test_df.shape[0]):,} ảnh test; 9 lớp.
- Train epochs mỗi lần: {study['epochs']}; batch size: {study['batch_size']}; checkpoint chọn bằng macro-F1 val.
- Python/PyTorch/GPU: xem `environment.json` và `pipeline_smoke.json`.
{smoke_line}- Số ảnh/lớp, overlap và kiểm tra checksum được lưu cùng EDA.
- Top-1 và macro-F1 được tính bằng `eval.py`; ECE dùng 15 bin. Độ trễ đo riêng ở batch 1 và batch 32, sau warmup 10 lượt và 50 phép đo, đồng bộ CUDA; không tính đọc ảnh/tiền xử lý.
- Phân bố lớp: `eda_class_distribution.png`. Split ngẫu nhiên, không theo địa điểm; test có thể lạc quan khi triển khai ở khu vực mới.

## 3. So sánh backbone

{backbone_table}

## 4. Công thức huấn luyện

{training_table}

T01–T05 lần lượt thay đổi một yếu tố so với T00. T06 kết hợp các yếu tố cho macro-F1 val cao hơn nền. Mỗi ablation có một seed, vì vậy chênh lệch nhỏ chỉ mang tính gợi ý; không khẳng định khác biệt nếu nó chưa được kiểm tra qua nhiều seed.

## 5. Suy luận và độ trễ

{inference_table}

Phương pháp được chọn trên validation: `{inference['inference_exp_id']}` (macro-F1 {inference['validation_macro_f1']:.4f}). {inference['selection_scope_note']} {realtime_statement} Temperature scaling {'được dùng' if inference['temperature_scale'] else 'không được dùng'}; T={inference['temperature']:.4f} fit trên validation; ECE val {inference['validation_ece_before']:.4f} → {inference['validation_ece_after']:.4f}. Độ trễ phương pháp được chọn: {latency_statement}. p50/p95/p99 được báo cáo cho batch 1 và batch 32, với warmup 10 lượt, 50 phép đo và CUDA synchronize; không gồm đọc ảnh.

## 6. Cấu hình cuối và lỗi

| Cấu hình | Macro-F1 test mean ± std | Top-1 test mean ± std | ECE test mean ± std |
|---|---:|---:|---:|
| F01 | {final_mean:.4f} ± {final_std:.4f} | {np.mean([r['top1'] for r in final_rows]):.4f} ± {np.std([r['top1'] for r in final_rows], ddof=1):.4f} | {np.mean([r['ece'] for r in final_rows]):.4f} ± {np.std([r['ece'] for r in final_rows], ddof=1):.4f} |
| T00 | {base_mean:.4f} ± {base_std:.4f} | {np.mean([r['top1'] for r in baseline_rows]):.4f} ± {np.std([r['top1'] for r in baseline_rows], ddof=1):.4f} | {np.mean([r['ece'] for r in baseline_rows]):.4f} ± {np.std([r['ece'] for r in baseline_rows], ddof=1):.4f} |

Lớp khó nhất theo nhầm chéo gộp: **{labels[i]} → {labels[j]}** ({int(matrix[i,j])} ảnh). Ma trận: `confusion_matrix_test.png`; ví dụ dự đoán sai từ seed 0: `misclassified_test.png`. Precision, recall và F1 trung bình qua seed của Chinee Apple và Snake Weed nằm trong sheet `PerClass`.

{perclass_table}

## 7. Kết luận và khuyến nghị

Macro-F1 F01 tăng {delta:+.4f} so với baseline. Chênh lệch này {evidence} ({noise:.4f}). Khi triển khai thời gian thực, cấu hình được đo có p95 batch 1 {latency_statement}; nếu chưa đạt 100 ms, chọn cấu hình nhanh hơn sau khi cân nhắc mức giảm macro-F1.

## 8. Hạn chế và việc tiếp theo

Kết quả dùng một fold và mỗi ablation chỉ có một seed; ảnh chia ngẫu nhiên chứ không theo địa điểm. Kết quả vì thế chưa chứng minh khả năng tổng quát sang mùa, ánh sáng hay địa hình khác. Không dùng điểm test để chọn lại cấu hình.

## 9. Phụ lục tái lập

- Thí nghiệm backbone: {', '.join(r['exp_id'] for r in study['backbones'])}.
- Thí nghiệm huấn luyện: {', '.join(r['exp_id'] for r in study['training'])}.
- Thí nghiệm suy luận: {', '.join(r['exp_id'] for r in inference['rows'])}.
- Chung kết: F01 và T00, seeds 0, 1, 2. Notebook nằm trong `code/lab_day2.ipynb`.
"""
    (output_dir / "report.md").write_text(report, encoding="utf-8")

    backbones_sheet = []
    for row in study["backbones"]:
        backbones_sheet.append({"exp_id": row["exp_id"], "backbone": row["backbone"],
                                "weight tag": row["pretrained_tag"], "parameters (M)": row["params_m"],
                                "GMAC": row["gmac"], "resolution": row["img_size"],
                                "epochs": study["epochs"], "seed": row["seed"],
                                "macro-F1 val": row["macro_f1_val"], "top-1 val": row["top1_val"],
                                "train sec/epoch": row["train_seconds_per_epoch"], "notes": "T00 recipe"})
    training_sheet = []
    baseline_val = study["baseline"]["macro_f1_val"]
    for row in study["training"]:
        training_sheet.append({"exp_id": row["exp_id"], "backbone": row["backbone"],
                               "axis": row.get("axis", "Baseline"), "change from T00": row.get("difference", "baseline"),
                               "seed": row["seed"], "macro-F1 val": row["macro_f1_val"],
                               "top-1 val": row["top1_val"], "delta vs T00": row["macro_f1_val"]-baseline_val,
                               "notes": "single-seed exploratory ablation"})
    inference_sheet = inference["rows"]
    latency_sheet = []
    for row in inference_sheet:
        for batch_size in (1, 32):
            suffix = f"batch-{batch_size}"
            latency_sheet.append({
                "configuration": row["exp_id"], "GPU": row.get("GPU", "unknown"),
                "dtype": row.get("dtype", "fp32"), "batch": batch_size,
                "input size": row.get("input size"),
                "BN fused": "yes" if row["exp_id"] == "I08" else "no",
                "p50 (ms)": row.get(f"p50 {suffix} (ms)"),
                "p95 (ms)": row.get(f"p95 {suffix} (ms)"),
                "p99 (ms)": row.get(f"p99 {suffix} (ms)"),
                "images/s": row.get(f"throughput {suffix} (img/s)"),
                "warmup": row.get("warmup"), "measurements": row.get("measurements"),
            })
    final_sheet = []
    for tag, rows in (("F01", final_rows), ("T00", baseline_rows)):
        for row in rows:
            results_parent = output_dir / ("runs" if tag == "F01" else "final_runs")
            train_result_path = results_parent / tag / f"seed{row['seed']}" / "result.json"
            val_macro_f1 = _json(train_result_path).get("macro_f1_val", np.nan) if train_result_path.is_file() else np.nan
            final_sheet.append({"exp_id": tag, "seed": row["seed"],
                                "macro-F1 test": row["macro_f1"], "top-1 test": row["top1"],
                                "ECE test": row["ece"], "macro-F1 val": val_macro_f1,
                                "configuration": best_recipe["backbone"] if tag == "F01" else "T00 baseline"})
        final_sheet.append({"exp_id": tag, "seed": "mean ± std",
                            "macro-F1 test": f"{np.mean([r['macro_f1'] for r in rows]):.4f} ± {np.std([r['macro_f1'] for r in rows], ddof=1):.4f}",
                            "top-1 test": f"{np.mean([r['top1'] for r in rows]):.4f} ± {np.std([r['top1'] for r in rows], ddof=1):.4f}",
                            "ECE test": f"{np.mean([r['ece'] for r in rows]):.4f} ± {np.std([r['ece'] for r in rows], ddof=1):.4f}"})
    top_summary = [{"group": "Backbone", "exp_id": r["exp_id"], "configuration": r["backbone"],
                    "macro-F1 val": r["macro_f1_val"], "top-1 val": r["top1_val"],
                    "params (M)": r["params_m"], "GMAC": r["gmac"]}
                   for r in study["backbones"]]
    top_summary += [{"group": "Training", "exp_id": r["exp_id"], "configuration": r.get("difference", "baseline"),
                     "macro-F1 val": r["macro_f1_val"], "top-1 val": r["top1_val"]}
                    for r in study["training"]]
    top_summary += [{"group": "Inference", "exp_id": r["exp_id"], "configuration": r["method"],
                     "macro-F1 val": r["macro-F1 val"], "top-1 val": r["top-1 val"],
                     "p95 batch-1 (ms)": r["p95 batch-1 (ms)"]}
                    for r in inference_sheet]
    top_summary.sort(key=lambda row: row["macro-F1 val"], reverse=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    workbook_path = output_dir / "results.xlsx"
    sheets = {
        "Backbones": pd.DataFrame(backbones_sheet), "Training": pd.DataFrame(training_sheet),
        "Inference": pd.DataFrame(inference_sheet), "Final": pd.DataFrame(final_sheet),
        "PerClass": pd.DataFrame(per_class_rows), "Latency": pd.DataFrame(latency_sheet),
        "Summary": pd.DataFrame(top_summary[:10]),
    }
    with pd.ExcelWriter(workbook_path, engine="openpyxl") as writer:
        for name, frame in sheets.items():
            frame.to_excel(writer, sheet_name=name, index=False)
        for sheet in writer.book.worksheets:
            sheet.freeze_panes = "A2"
            sheet.auto_filter.ref = sheet.dimensions
            for cell in sheet[1]:
                cell.font = Font(bold=True, color="FFFFFF")
                cell.fill = PatternFill("solid", fgColor="1F4E78")
            for column in sheet.columns:
                width = min(48, max(12, max(len(str(cell.value or "")) for cell in column) + 2))
                sheet.column_dimensions[column[0].column_letter].width = width

    return {"final_macro_f1_mean": final_mean, "final_macro_f1_std": final_std,
            "baseline_macro_f1_mean": base_mean, "baseline_macro_f1_std": base_std,
            "eval_out": str(out_eval), "workbook": str(workbook_path)}


def export_submission(output_dir: str | Path, student_slug: str, environment: dict | None = None,
                      submissions_dir: str | Path | None = None) -> Path:
    """Assemble the six required local submission products without dataset/checkpoints."""
    output_dir = Path(output_dir).resolve()
    if not re.fullmatch(r"[A-Za-z0-9]+_[a-z0-9]+(?:_[a-z0-9]+)*", student_slug):
        raise ValueError("student_slug phải có dạng MSSV_ho_ten_khong_dau, chỉ gồm chữ/số và gạch dưới")
    dest_root = Path(submissions_dir).resolve() if submissions_dir else output_dir / "submissions"
    dest = dest_root / student_slug
    dest.mkdir(parents=True, exist_ok=True)
    code = dest / "code"
    code.mkdir(exist_ok=True)
    for source in list((ROOT / "starter").glob("*.py")) + [ROOT / "starter" / "lab_day2.ipynb", ROOT / "eval.py"]:
        shutil.copy2(source, code / source.name)
    for folder in ("curves", "predictions"):
        source = output_dir / folder
        target = dest / folder
        if source.exists():
            shutil.copytree(source, target, dirs_exist_ok=True)
        else:
            target.mkdir()
    for name in ("results.xlsx", "report.md", "confusion_matrix_test.png", "misclassified_test.png",
                 "eda_class_distribution.png", "eda_train_examples.png", "eda_class_counts.csv",
                 "split_check.json", "pipeline_smoke.json", "environment.json"):
        source = output_dir / name
        if source.is_file():
            shutil.copy2(source, dest / name)
    if environment:
        _save_json(dest / "environment.json", environment)
    final = _json(output_dir / "runs" / "F01" / "seed0" / "result.json")
    readme = f"""# DeepWeeds Lab Day 2

## Nộp bài

- MSSV/tên: `{student_slug}`
- Notebook tái lập: [`code/lab_day2.ipynb`](code/lab_day2.ipynb) (đính kèm trong thư mục code; chưa đăng/chia sẻ công khai).
- Mã nguồn: thư mục `code/`; `eval.py` được giữ nguyên.
- Dataset: DeepWeeds, fold 0, CSV gốc; MD5 ảnh đã kiểm tra theo notebook.
- Seeds chung kết và baseline: 0, 1, 2.

## Môi trường đã đo

- GPU: {final.get('gpu', 'Colab GPU')}
- PyTorch: {final.get('torch', 'ghi trong environment.json')}
- Python/timm: xem `environment.json`.
- AMP: {'bật' if final.get('gpu') != 'CPU' else 'không'}; ảnh train {final.get('img_size')} px; inference {final.get('inference_img_size')} px.

## Chạy lại

1. Mở `code/lab_day2.ipynb` bằng Kaggle hoặc Google Colab, bật GPU và Internet.
2. Chạy cài thư viện, tải dữ liệu, EDA và kiểm tra pipeline theo thứ tự ô.
3. Chạy bước backbone/công thức/suy luận, sau đó bước chung kết và tự chấm `eval.py`.
4. Mỗi số trong báo cáo và `results.xlsx` được tạo từ lần chạy thật; predictions test lưu ở `predictions/`.

Không kèm ảnh dataset, checkpoint hoặc file zip dữ liệu.
"""
    (dest / "README.md").write_text(readme, encoding="utf-8")
    return dest
