"""Classification losses, class weights, and batch-level Mixup/CutMix."""
from __future__ import annotations


def build_criterion(kind: str = "ce", **kw):
    import torch
    if kind == "ce":
        return torch.nn.CrossEntropyLoss(weight=kw.get("weight"))
    if kind == "ls":
        return LabelSmoothingCE(kw.get("smoothing", 0.1), weight=kw.get("weight"))
    if kind == "focal":
        return FocalLoss(kw.get("gamma", 2.0), alpha=kw.get("alpha"))
    if kind == "ce_weighted":
        if kw.get("weight") is None:
            raise ValueError("ce_weighted cần tensor weight lấy từ số mẫu TRAIN")
        return torch.nn.CrossEntropyLoss(weight=kw["weight"])
    raise ValueError(f"Loss không hỗ trợ: {kind}")


class LabelSmoothingCE:
    def __new__(cls, smoothing: float = 0.1, weight=None):
        import torch

        class _LabelSmoothingCE(torch.nn.Module):
            def __init__(self):
                super().__init__()
                if not 0 <= smoothing < 1:
                    raise ValueError("smoothing phải thuộc [0, 1)")
                self.smoothing = float(smoothing)
                self.register_buffer("weight", weight if weight is None else torch.as_tensor(weight).float())

            def forward(self, logits, target):
                return torch.nn.functional.cross_entropy(
                    logits, target, weight=self.weight, label_smoothing=self.smoothing)

        return _LabelSmoothingCE()


class FocalLoss:
    def __new__(cls, gamma: float = 2.0, alpha=None):
        import torch

        class _FocalLoss(torch.nn.Module):
            def __init__(self):
                super().__init__()
                if gamma < 0:
                    raise ValueError("gamma phải >= 0")
                self.gamma = float(gamma)
                self.register_buffer("alpha", None if alpha is None else torch.as_tensor(alpha).float())

            def forward(self, logits, target):
                log_probs = torch.nn.functional.log_softmax(logits, dim=1)
                log_pt = log_probs.gather(1, target.long().unsqueeze(1)).squeeze(1)
                pt = log_pt.exp()
                loss = -(1.0 - pt).clamp_min(0).pow(self.gamma) * log_pt
                if self.alpha is not None:
                    loss = loss * self.alpha.to(logits.device)[target.long()]
                return loss.mean()

        return _FocalLoss()


def class_weights(counts, beta: float = 0.0):
    import torch
    values = torch.as_tensor(counts, dtype=torch.float64)
    if values.ndim != 1 or len(values) != 9 or (values <= 0).any():
        raise ValueError("counts cần có 9 số dương, chỉ đếm từ train")
    if not 0 <= beta < 1:
        raise ValueError("beta phải thuộc [0, 1)")
    if beta == 0:
        weights = values.reciprocal()
        weights = weights / weights.mean()
    else:
        weights = (1 - beta) / (1 - torch.pow(torch.tensor(beta, dtype=values.dtype), values))
        weights = weights / weights.sum() * len(values)
    return weights.float()


def mix_batch(x, y, alpha: float = 1.0, mode: str = "cutmix"):
    import numpy as np
    import torch
    if alpha <= 0:
        raise ValueError("alpha phải > 0")
    if mode not in {"mixup", "cutmix"}:
        raise ValueError("mode phải là mixup hoặc cutmix")
    lam = float(np.random.beta(alpha, alpha))
    perm = torch.randperm(x.size(0), device=x.device)
    y_a, y_b = y, y[perm]
    mixed = x.clone()
    if mode == "mixup":
        mixed = lam * x + (1.0 - lam) * x[perm]
    else:
        height, width = x.shape[-2:]
        ratio = float(np.sqrt(1.0 - lam))
        cut_w, cut_h = int(width * ratio), int(height * ratio)
        cx = int(torch.randint(width, (1,), device=x.device).item())
        cy = int(torch.randint(height, (1,), device=x.device).item())
        x1, x2 = max(cx - cut_w // 2, 0), min(cx + cut_w // 2, width)
        y1, y2 = max(cy - cut_h // 2, 0), min(cy + cut_h // 2, height)
        mixed[:, :, y1:y2, x1:x2] = x[perm, :, y1:y2, x1:x2]
        lam = 1.0 - ((x2 - x1) * (y2 - y1) / float(width * height))
    return mixed, (y_a, y_b, lam)


def mixed_loss(criterion, logits, targets):
    y_a, y_b, lam = targets
    return float(lam) * criterion(logits, y_a) + (1.0 - float(lam)) * criterion(logits, y_b)
