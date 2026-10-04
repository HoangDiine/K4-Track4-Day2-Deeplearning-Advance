"""Timm backbones and optimizer parameter grouping for DeepWeeds."""
from __future__ import annotations

SUGGESTED_BACKBONES = {
    "resnet50": "resnet50",
    "resnext50": "resnext50_32x4d",
    "convnext_tiny": "convnext_tiny",
    "deit_small": "deit_small_patch16_224",
    "swin_tiny": "swin_tiny_patch4_window7_224",
    "efficientnet_b0": "efficientnet_b0",
    "mobilenetv3": "mobilenetv3_large_100",
}


def build_model(name: str, pretrained: bool = True, num_classes: int = 9,
                drop_rate: float = 0.0, init: str = "finetune"):
    import timm
    if init not in {"scratch", "frozen", "finetune"}:
        raise ValueError("init phải là scratch, frozen hoặc finetune")
    model = timm.create_model(name, pretrained=pretrained and init != "scratch",
                              num_classes=num_classes, drop_rate=drop_rate)
    model.pretrained_tag = getattr(model, "pretrained_cfg", {}).get("tag", "default")
    if init == "frozen":
        freeze_backbone(model)
    return model


def _head_parameters(model):
    classifier = model.get_classifier() if hasattr(model, "get_classifier") else None
    if classifier is None:
        raise ValueError("Backbone không cung cấp get_classifier(); không thể nhận diện head.")
    return {id(p) for p in classifier.parameters()}


def freeze_backbone(model) -> None:
    head_ids = _head_parameters(model)
    for parameter in model.parameters():
        parameter.requires_grad_(id(parameter) in head_ids)


def param_groups(model, lr_backbone: float, lr_head: float, weight_decay: float):
    head_ids = _head_parameters(model)
    backbone_decay, backbone_no_decay, head = [], [], []
    for parameter in model.parameters():
        if not parameter.requires_grad:
            continue
        if id(parameter) in head_ids:
            head.append(parameter)
        elif parameter.ndim <= 1:
            backbone_no_decay.append(parameter)
        else:
            backbone_decay.append(parameter)
    groups = []
    for params, lr, decay in (
        (backbone_decay, lr_backbone, weight_decay),
        (backbone_no_decay, lr_backbone, 0.0),
        (head, lr_head, weight_decay),
    ):
        if params:
            groups.append({"params": params, "lr": lr, "weight_decay": decay})
    if not groups:
        raise ValueError("Mô hình không có tham số đang huấn luyện")
    return groups


def count_params(model) -> float:
    return sum(p.numel() for p in model.parameters()) / 1_000_000


def count_gmacs(model, img_size: int = 224) -> float:
    """Estimate MACs with fvcore when available; report a module-hook fallback otherwise."""
    import torch
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    example = torch.zeros(1, 3, img_size, img_size, device=device)
    try:
        try:
            from fvcore.nn import FlopCountAnalysis
            macs = float(FlopCountAnalysis(model, example).total())
        except (ImportError, RuntimeError, TypeError, ValueError, AttributeError):
            # Fallback counts convolution and linear MACs. It can undercount attention's
            # tensor-matrix products, so the report labels its counting method.
            counts = {"macs": 0}
            handles = []

            def conv_hook(module, _inputs, output):
                kh, kw = module.kernel_size
                counts["macs"] += output.numel() * (module.in_channels // module.groups) * kh * kw

            def linear_hook(module, _inputs, output):
                counts["macs"] += output.numel() * module.in_features

            for module in model.modules():
                if isinstance(module, torch.nn.Conv2d):
                    handles.append(module.register_forward_hook(conv_hook))
                elif isinstance(module, torch.nn.Linear):
                    handles.append(module.register_forward_hook(linear_hook))
            with torch.inference_mode():
                model(example)
            for handle in handles:
                handle.remove()
            macs = float(counts["macs"])
        return macs / 1_000_000_000
    finally:
        model.train(was_training)
