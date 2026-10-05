"""Synchronized p50/p95/p99 inference timing for CPU and CUDA."""
from __future__ import annotations

import time
import numpy as np


def bench(fn, warmup: int = 10, iters: int = 100, sync=None) -> dict:
    if warmup < 0 or iters < 1:
        raise ValueError("warmup >= 0 và iters >= 1")
    for _ in range(max(10, warmup)):
        fn()
    samples = []
    for _ in range(max(50, iters)):
        if sync:
            sync()
        started = time.perf_counter()
        fn()
        if sync:
            sync()
        samples.append((time.perf_counter() - started) * 1000.0)
    values = np.asarray(samples, dtype=np.float64)
    return {"p50": float(np.percentile(values, 50)),
            "p95": float(np.percentile(values, 95)),
            "p99": float(np.percentile(values, 99)),
            "mean": float(values.mean()), "n": len(samples),
            "warmup": max(10, warmup)}


def _prepare(model, batch_size, img_size, dtype, device):
    import copy
    import torch
    device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA được yêu cầu nhưng không có GPU")
    if dtype not in {"fp32", "amp", "fp16"}:
        raise ValueError("dtype phải là fp32, amp hoặc fp16")
    if dtype == "fp16" and device.type != "cuda":
        raise ValueError("FP16 chỉ benchmark trên CUDA")
    model = copy.deepcopy(model).to(device).eval()
    if dtype == "fp16":
        model = model.half()
    x = torch.randn(batch_size, 3, img_size, img_size, device=device)
    if dtype == "fp16":
        x = x.half()

    def forward():
        if dtype == "amp":
            with torch.autocast(device_type=device.type, dtype=torch.float16,
                                enabled=device.type == "cuda"):
                model(x)
        else:
            model(x)

    sync = (lambda: torch.cuda.synchronize(device)) if device.type == "cuda" else None
    return model, x, forward, sync, device


def latency_report(model, batch_size: int, img_size: int, dtype: str = "fp32", device: str = "cuda",
                   warmup: int = 10, iters: int = 100) -> dict:
    import torch
    _model, _x, forward, sync, device_obj = _prepare(model, batch_size, img_size, dtype, device)
    with torch.inference_mode():
        timing = bench(forward, warmup=warmup, iters=iters, sync=sync)
    p50 = timing["p50"]
    return {
        "gpu": torch.cuda.get_device_name(device_obj) if device_obj.type == "cuda" else "CPU",
        "dtype": dtype, "batch": int(batch_size), "img_size": int(img_size),
        **timing, "images_per_s": float(batch_size / max(p50, 1e-12) * 1000),
        "torch": torch.__version__, "includes_preprocessing": False,
        "batch1_p95_ms": timing["p95"] if batch_size == 1 else None,
    }


def tta_latency(model, k_views: int, **kw) -> dict:
    import torch
    if k_views < 1:
        raise ValueError("k_views phải >= 1")
    batch_size = kw.pop("batch_size", 1)
    img_size = kw.pop("img_size", 224)
    dtype = kw.pop("dtype", "fp32")
    device = kw.pop("device", "cuda")
    warmup = kw.pop("warmup", 10)
    iters = kw.pop("iters", 100)
    if kw:
        raise TypeError(f"Tham số không hỗ trợ: {sorted(kw)}")
    _model, x, forward, sync, device_obj = _prepare(model, batch_size, img_size, dtype, device)

    def repeated_views():
        for _ in range(k_views):
            forward()

    with torch.inference_mode():
        timing = bench(repeated_views, warmup=warmup, iters=iters, sync=sync)
    return {
        "gpu": torch.cuda.get_device_name(device_obj) if device_obj.type == "cuda" else "CPU",
        "dtype": dtype, "batch": batch_size, "img_size": img_size,
        "k_views": k_views, **timing,
        "images_per_s": float(batch_size / max(timing["p50"], 1e-12) * 1000),
        "torch": torch.__version__, "includes_preprocessing": False,
    }


def ensemble_latency(models, batch_size: int, img_size: int, dtype: str = "fp32", device: str = "cuda",
                     warmup: int = 10, iters: int = 100) -> dict:
    """Measure a probability ensemble by timing one forward through every model per input batch."""
    import torch

    if not models:
        raise ValueError("Ensemble cần ít nhất một model")
    device_obj = torch.device(device)
    if device_obj.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA được yêu cầu nhưng không có GPU")
    if dtype not in {"fp32", "amp", "fp16"}:
        raise ValueError("dtype phải là fp32, amp hoặc fp16")
    if dtype == "fp16" and device_obj.type != "cuda":
        raise ValueError("FP16 chỉ benchmark trên CUDA")

    ensemble_models = [model.to(device_obj).eval() for model in models]
    if dtype == "fp16":
        ensemble_models = [model.half() for model in ensemble_models]
    images = torch.randn(batch_size, 3, img_size, img_size, device=device_obj)
    if dtype == "fp16":
        images = images.half()

    def forward():
        if dtype == "amp":
            with torch.autocast(device_type=device_obj.type, dtype=torch.float16,
                                enabled=device_obj.type == "cuda"):
                for model in ensemble_models:
                    model(images)
        else:
            for model in ensemble_models:
                model(images)

    sync = (lambda: torch.cuda.synchronize(device_obj)) if device_obj.type == "cuda" else None
    with torch.inference_mode():
        timing = bench(forward, warmup=warmup, iters=iters, sync=sync)
    p50 = timing["p50"]
    return {
        "gpu": torch.cuda.get_device_name(device_obj) if device_obj.type == "cuda" else "CPU",
        "dtype": dtype, "batch": int(batch_size), "img_size": int(img_size),
        "k_models": len(ensemble_models), **timing,
        "images_per_s": float(batch_size / max(p50, 1e-12) * 1000),
        "torch": torch.__version__, "includes_preprocessing": False,
    }
