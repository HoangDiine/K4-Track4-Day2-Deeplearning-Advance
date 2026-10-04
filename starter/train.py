"""Shared train/evaluate entry point for backbone, recipe, and final experiments."""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, fields
import json
import math
from pathlib import Path
import random
import sys
import time
import typing
from typing import get_type_hints

import numpy as np


@dataclass
class Config:
    exp_id: str = "T00"
    seed: int = 0
    fold: int = 0
    backbone: str = "resnet50"
    init: str = "finetune"                 # scratch | frozen | finetune
    pretrained: bool = True
    drop_rate: float = 0.0
    img_size: int = 224
    aug: str = "basic"                     # basic | color | trivial | randaug | none
    sampler: str | None = None
    mix: str | None = None                 # None | mixup | cutmix
    mix_alpha: float = 1.0
    loss: str = "ce"                       # ce | ls | focal | ce_weighted
    label_smoothing: float = 0.0
    focal_gamma: float = 2.0
    class_weight_beta: float | None = None
    epochs: int = 12
    batch_size: int = 64
    lr_backbone: float = 1e-4
    lr_head: float = 1e-3
    weight_decay: float = 0.05
    warmup_epochs: float = 1.0
    ema_decay: float | None = None
    temperature_scale: bool = False        # fit T on val; apply to test after final choice
    tta: str = "none"                     # none | hflip | multicrop
    inference_space: str = "prob"          # prob | logit
    inference_img_size: int | None = None
    tta_crop_size: int = 192
    amp: bool = True
    num_workers: int = 2
    images_dir: str = "data/images"
    labels_dir: str = "data/labels"
    out_dir: str = "runs"
    pred_dir: str = "predictions"
    curves_dir: str = "curves"
    save_test_predictions: bool = False


def run_dir(cfg: Config) -> Path:
    return Path(cfg.out_dir) / cfg.exp_id / f"seed{cfg.seed}"


def pred_path(cfg: Config, split: str) -> Path:
    return Path(cfg.pred_dir) / f"{cfg.exp_id}_seed{cfg.seed}_{split}.csv"


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Không thể ghi kiểu {type(value).__name__} vào JSON")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def build_optimizer(model, cfg: Config):
    import torch
    from model import param_groups
    return torch.optim.AdamW(param_groups(model, cfg.lr_backbone, cfg.lr_head, cfg.weight_decay))


def build_scheduler(optimizer, cfg: Config, steps_per_epoch: int):
    import torch
    total_steps = max(1, cfg.epochs * steps_per_epoch)
    warmup_steps = min(total_steps - 1, max(0, int(cfg.warmup_epochs * steps_per_epoch)))

    def scale(step):
        if warmup_steps and step < warmup_steps:
            return max(1e-3, (step + 1) / warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.01, 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress))))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, scale)


class EMA:
    def __init__(self, model, decay: float):
        import copy
        if not 0.0 < decay < 1.0:
            raise ValueError("ema_decay phải thuộc (0, 1)")
        self.decay = float(decay)
        self.model = copy.deepcopy(model).eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    @property
    def ema(self):
        return self.model

    def update(self, model) -> None:
        import torch
        source = model.state_dict()
        target = self.model.state_dict()
        with torch.no_grad():
            for name, value in target.items():
                incoming = source[name].detach()
                if value.is_floating_point():
                    value.mul_(self.decay).add_(incoming, alpha=1.0 - self.decay)
                else:
                    value.copy_(incoming)

    def copy_to(self, model) -> None:
        model.load_state_dict(self.model.state_dict())


def train_one_epoch(model, loader, criterion, optimizer, scheduler, scaler, cfg: Config,
                    device, ema: EMA | None = None) -> dict:
    import torch
    from losses import mix_batch, mixed_loss
    model.train()
    if cfg.init == "frozen":
        for module in model.modules():
            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                module.eval()
    amp_enabled = bool(cfg.amp and device.type == "cuda")
    total_loss, seen = 0.0, 0
    start = time.perf_counter()
    for images, labels, _filenames in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        targets = labels
        if cfg.mix:
            images, targets = mix_batch(images, labels, cfg.mix_alpha, cfg.mix)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
            logits = model(images)
            loss = mixed_loss(criterion, logits, targets) if cfg.mix else criterion(logits, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        if ema is not None:
            ema.update(model)
        total_loss += float(loss.detach()) * len(labels)
        seen += len(labels)
    return {"train_loss": total_loss / max(1, seen), "lr": optimizer.param_groups[0]["lr"],
            "train_seconds": time.perf_counter() - start}


def evaluate(model, loader, criterion, device):
    import torch
    model.eval()
    names, truths, outputs = [], [], []
    loss_sum, seen = 0.0, 0
    with torch.inference_mode():
        for images, labels, filenames in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(images)
            loss_sum += float(criterion(logits, labels)) * len(labels)
            seen += len(labels)
            names.extend(list(filenames))
            truths.append(labels.cpu().numpy())
            outputs.append(logits.float().cpu().numpy())
    y_true = np.concatenate(truths) if truths else np.empty((0,), dtype=np.int64)
    logits = np.concatenate(outputs) if outputs else np.empty((0, 9), dtype=np.float32)
    return names, y_true, logits, loss_sum / max(1, seen)


def plot_curves(history: list[dict], path: str | Path, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    frame = __import__("pandas").DataFrame(history)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    axes[0].plot(frame["epoch"], frame["train_loss"], label="Train")
    axes[0].plot(frame["epoch"], frame["val_loss"], label="Validation")
    axes[0].set(title="Loss", xlabel="Epoch", ylabel="Cross-entropy loss")
    axes[0].legend()
    axes[1].plot(frame["epoch"], frame["val_macro_f1"], label="Val macro-F1")
    axes[1].plot(frame["epoch"], frame["val_top1"], label="Val top-1")
    axes[1].set(title="Validation metrics", xlabel="Epoch", ylabel="Score", ylim=(0, 1))
    axes[1].legend()
    fig.suptitle(title)
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _softmax(logits):
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def _evaluate_tta(model, loader, device, mode: str, crop_size: int, space: str):
    from inference import aggregate_views, predict_logits, views_multicrop, view_hflip
    if mode not in {"none", "hflip", "multicrop"}:
        raise ValueError("tta phải là none, hflip hoặc multicrop")
    if space not in {"prob", "logit"}:
        raise ValueError("inference_space phải là prob hoặc logit")
    names, truths, logits = predict_logits(model, loader, device)
    if mode == "none":
        return names, truths, logits
    if mode == "hflip":
        h_names, h_truths, h_logits = predict_logits(model, loader, device, view_hflip)
        if names != h_names or not np.array_equal(truths, h_truths):
            raise RuntimeError("TTA views returned different validation/test ordering")
        views = [logits, h_logits]
    else:
        import torch
        model.eval()
        per_view_batches = [[] for _ in range(5)]
        with torch.inference_mode():
            for images, _labels, _filenames in loader:
                images = images.to(device, non_blocking=True)
                for index, crop in enumerate(views_multicrop(images, crop_size)):
                    device_type = torch.device(device).type
                    with torch.autocast(device_type=device_type, dtype=torch.float16, enabled=False):
                        per_view_batches[index].append(model(crop).float().cpu().numpy())
        views = [np.concatenate(parts, axis=0) for parts in per_view_batches]
    probabilities = aggregate_views(views, space=space)
    scores = np.log(np.clip(probabilities, 1e-12, 1.0)) if space == "prob" else np.mean(views, axis=0)
    return names, truths, scores


def run(cfg: Config) -> dict:
    import pandas as pd
    import torch
    from dataset import build_transforms, check_split, load_split, make_loader
    from losses import build_criterion, class_weights
    from model import build_model, count_gmacs, count_params

    if cfg.fold != 0:
        raise ValueError("Bài chính chỉ sử dụng fold 0")
    if cfg.epochs < 1 or cfg.batch_size < 1:
        raise ValueError("epochs và batch_size cần >= 1")
    if cfg.mix not in (None, "mixup", "cutmix"):
        raise ValueError("mix phải là None, mixup hoặc cutmix")
    if cfg.tta not in {"none", "hflip", "multicrop"}:
        raise ValueError("tta phải là none, hflip hoặc multicrop")
    if cfg.save_test_predictions:
        destination = pred_path(cfg, "test")
        if destination.exists():
            raise FileExistsError(f"Test output đã tồn tại; không chạy lại test: {destination}")

    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_path = run_dir(cfg)
    run_path.mkdir(parents=True, exist_ok=True)
    Path(cfg.pred_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.curves_dir).mkdir(parents=True, exist_ok=True)
    with (run_path / "config.json").open("w", encoding="utf-8") as stream:
        json.dump(asdict(cfg), stream, indent=2, ensure_ascii=False)

    train_df, val_df, test_df = load_split(cfg.labels_dir, cfg.fold)
    split_report = check_split(train_df, val_df, test_df, cfg.images_dir)
    train_loader = make_loader(train_df, cfg.images_dir, build_transforms(True, cfg.img_size, cfg.aug),
                               cfg.batch_size, train=True, sampler=cfg.sampler,
                               num_workers=cfg.num_workers)
    val_loader = make_loader(val_df, cfg.images_dir, build_transforms(False, cfg.img_size),
                             cfg.batch_size, train=False, num_workers=cfg.num_workers)

    model = build_model(cfg.backbone, pretrained=cfg.pretrained, num_classes=9,
                        drop_rate=cfg.drop_rate, init=cfg.init).to(device)
    n_params = count_params(model)
    gmacs = count_gmacs(model, cfg.img_size)
    counts = np.bincount(train_df["Label"].astype(int), minlength=9)
    weights = class_weights(counts, cfg.class_weight_beta) if cfg.class_weight_beta is not None else None
    if cfg.loss == "ce_weighted" and weights is None:
        weights = class_weights(counts, beta=0)
    if weights is not None:
        weights = weights.to(device)
    criterion = build_criterion(cfg.loss, smoothing=cfg.label_smoothing,
                                gamma=cfg.focal_gamma, weight=weights)
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg, len(train_loader))
    amp_enabled = bool(cfg.amp and device.type == "cuda")
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    except (AttributeError, TypeError):
        scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    ema = EMA(model, cfg.ema_decay) if cfg.ema_decay is not None else None

    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    import eval as eval_module

    history, best_f1, best_epoch = [], -float("inf"), 0
    best_state = None
    epoch_times = []
    for epoch in range(1, cfg.epochs + 1):
        epoch_start = time.perf_counter()
        train_stats = train_one_epoch(model, train_loader, criterion, optimizer, scheduler,
                                      scaler, cfg, device, ema)
        evaluator = ema.model if ema is not None else model
        val_names, val_y, val_logits, val_loss = evaluate(evaluator, val_loader, criterion, device)
        val_prob = _softmax(val_logits)
        metrics = eval_module.compute_metrics(val_y, val_prob.argmax(1), val_prob)
        item = {"epoch": epoch, **train_stats, "val_loss": val_loss,
                "val_macro_f1": metrics["macro_f1"], "val_top1": metrics["top1"],
                "val_ece": metrics["ece"]}
        history.append(item)
        epoch_times.append(time.perf_counter() - epoch_start)
        print(f"{cfg.exp_id} seed={cfg.seed} epoch={epoch}/{cfg.epochs} "
              f"loss={item['train_loss']:.4f}/{val_loss:.4f} "
              f"val_macro_f1={metrics['macro_f1']:.4f} top1={metrics['top1']:.4f}")
        if metrics["macro_f1"] > best_f1:
            best_f1, best_epoch = float(metrics["macro_f1"]), epoch
            best_state = {key: value.detach().cpu().clone()
                          for key, value in evaluator.state_dict().items()}
    if best_state is None:
        raise RuntimeError("Không tạo được checkpoint tốt nhất")
    checkpoint_path = run_path / "best.pth"
    torch.save({"state_dict": best_state, "best_epoch": best_epoch, "best_val_macro_f1": best_f1},
               checkpoint_path)
    evaluator = ema.model if ema is not None else model
    evaluator.load_state_dict(best_state)
    selected_resolution = cfg.inference_img_size or cfg.img_size
    if cfg.tta == "multicrop":
        selected_resolution = max(selected_resolution, cfg.tta_crop_size + 32)
    selected_val_loader = val_loader if selected_resolution == cfg.img_size else make_loader(
        val_df, cfg.images_dir, build_transforms(False, selected_resolution), cfg.batch_size,
        train=False, num_workers=cfg.num_workers)
    val_names, val_y, val_logits = _evaluate_tta(evaluator, selected_val_loader, device,
                                                cfg.tta, cfg.tta_crop_size, cfg.inference_space)
    np.savez_compressed(run_path / "val_logits.npz", filenames=np.asarray(val_names),
                        y_true=val_y, logits=val_logits)

    from inference import apply_temperature, fit_temperature
    val_probs_uncal = apply_temperature(val_logits, 1.0)
    temperature = fit_temperature(val_logits, val_y) if cfg.temperature_scale else 1.0
    val_probs = apply_temperature(val_logits, temperature)
    if cfg.temperature_scale:
        raw_path = pred_path(cfg, "val").with_name(f"{cfg.exp_id}uncal_seed{cfg.seed}_val.csv")
        eval_module.save_predictions(raw_path, val_names, val_y, val_probs_uncal)
    eval_module.save_predictions(pred_path(cfg, "val"), val_names, val_y, val_probs)

    test_metrics = None
    if cfg.save_test_predictions:
        test_loader = make_loader(test_df, cfg.images_dir, build_transforms(False, selected_resolution),
                                  cfg.batch_size, train=False, num_workers=cfg.num_workers)
        test_names, test_y, test_logits = _evaluate_tta(evaluator, test_loader, device,
                                                        cfg.tta, cfg.tta_crop_size, cfg.inference_space)
        test_probs_uncal = apply_temperature(test_logits, 1.0)
        test_probs = apply_temperature(test_logits, temperature)
        if cfg.temperature_scale:
            raw_path = pred_path(cfg, "test").with_name(f"{cfg.exp_id}uncal_seed{cfg.seed}_test.csv")
            eval_module.save_predictions(raw_path, test_names, test_y, test_probs_uncal)
        eval_module.save_predictions(pred_path(cfg, "test"), test_names, test_y, test_probs)
        test_metrics = eval_module.compute_metrics(test_y, test_probs.argmax(1), test_probs)
        np.savez_compressed(run_path / "test_logits.npz", filenames=np.asarray(test_names),
                            y_true=test_y, logits=test_logits)

    pd.DataFrame(history).to_csv(run_path / "history.csv", index=False)
    plot_curves(history, Path(cfg.curves_dir) / f"{cfg.exp_id}_{cfg.backbone}_seed{cfg.seed}.png",
                f"{cfg.exp_id} — {cfg.backbone} — seed {cfg.seed}")
    result = {
        "exp_id": cfg.exp_id, "seed": cfg.seed, "backbone": cfg.backbone,
        "pretrained_tag": getattr(model, "pretrained_tag", "unknown"),
        "params_m": n_params, "gmac": gmacs, "img_size": cfg.img_size,
        "best_epoch": best_epoch, "macro_f1_val": float(eval_module.compute_metrics(
            val_y, val_probs.argmax(1), val_probs)["macro_f1"]),
        "top1_val": float(eval_module.compute_metrics(val_y, val_probs.argmax(1), val_probs)["top1"]),
        "ece_val": float(eval_module.compute_metrics(val_y, val_probs.argmax(1), val_probs)["ece"]),
        "temperature": temperature, "train_seconds_per_epoch": float(np.mean(epoch_times)),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU",
        "torch": torch.__version__, "split": split_report, "test_metrics": test_metrics,
        "checkpoint": str(checkpoint_path), "tta": cfg.tta,
        "inference_img_size": selected_resolution,
    }
    with (run_path / "result.json").open("w", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False, default=_json_default)
    return result


def _parse_value(value: str, annotation, default):
    if value.lower() in {"none", "null"}:
        return None
    if annotation is bool or isinstance(default, bool):
        if value.lower() not in {"true", "false", "1", "0", "yes", "no"}:
            raise ValueError(f"Boolean không hợp lệ: {value}")
        return value.lower() in {"true", "1", "yes"}
    choices = typing.get_args(annotation)
    if choices:
        concrete = next((arg for arg in choices if arg is not type(None)), str)
        return _parse_value(value, concrete, default)
    if annotation is int or isinstance(default, int) and not isinstance(default, bool):
        return int(value)
    if annotation is float or isinstance(default, float):
        return float(value)
    return value


def parse_overrides(pairs: list[str]) -> dict:
    defaults = Config()
    valid = {item.name: item for item in fields(Config)}
    annotations = get_type_hints(Config)
    parsed = {}
    for pair in pairs:
        if "=" not in pair:
            raise ValueError(f"Override cần dạng key=value: {pair}")
        key, value = pair.split("=", 1)
        if key not in valid:
            raise ValueError(f"Tham số Config không tồn tại: {key}")
        parsed[key] = _parse_value(value, annotations[key], getattr(defaults, key))
    return parsed


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate one DeepWeeds configuration")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    args = parser.parse_args(argv)
    cfg = Config(**parse_overrides(args.set))
    print(json.dumps(run(cfg), indent=2, ensure_ascii=False, default=float))


if __name__ == "__main__":
    main()
