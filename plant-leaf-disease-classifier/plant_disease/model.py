from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models

SUPPORTED_ARCHS = ("resnet18", "resnet50", "efficientnet_b0", "mobilenet_v3_large")


def build_model(arch: str, num_classes: int, pretrained: bool = True, dropout: float = 0.2) -> nn.Module:
    if arch not in SUPPORTED_ARCHS:
        raise ValueError(f"unknown arch '{arch}', expected one of {SUPPORTED_ARCHS}")

    weights = "DEFAULT" if pretrained else None
    model = getattr(models, arch)(weights=weights)

    if arch.startswith("resnet"):
        in_features = model.fc.in_features
        model.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_features, num_classes))
    else:
        # efficientnet / mobilenet keep the classifier as a Sequential with
        # the Linear layer last.
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
    return model


def head_parameters(model: nn.Module):
    return model.fc.parameters() if hasattr(model, "fc") else model.classifier.parameters()


def freeze_backbone(model: nn.Module) -> None:
    for p in model.parameters():
        p.requires_grad = False
    for p in head_parameters(model):
        p.requires_grad = True


def unfreeze_top(model: nn.Module, arch: str) -> None:
    """Unfreeze the last couple of stages for fine-tuning."""
    if arch.startswith("resnet"):
        blocks = [model.layer3, model.layer4]
    else:
        blocks = list(model.features)[-3:]
    for block in blocks:
        for p in block.parameters():
            p.requires_grad = True


def target_layer(model: nn.Module, arch: str) -> nn.Module:
    """Last convolutional stage, used for Grad-CAM."""
    if arch.startswith("resnet"):
        return model.layer4[-1]
    return model.features[-1]


def save_checkpoint(path: Path, model: nn.Module, arch: str, classes: list[str],
                    image_size: int, extra: dict | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "arch": arch,
        "classes": list(classes),
        "image_size": image_size,
        "model_state_dict": model.state_dict(),
        **(extra or {}),
    }, path)


def load_checkpoint(path: Path, device: torch.device | str = "cpu"):
    ckpt = torch.load(path, map_location=device, weights_only=True)

    # Checkpoints from the first version of this project only stored the
    # weights and class names of a plain resnet18.
    arch = ckpt.get("arch", "resnet18")
    model = build_model(arch, len(ckpt["classes"]), pretrained=False)
    state = ckpt["model_state_dict"]
    if "fc.weight" in state and isinstance(model.fc, nn.Sequential):
        state = {k.replace("fc.", "fc.1.", 1) if k.startswith("fc.") else k: v for k, v in state.items()}
    model.load_state_dict(state)
    model.to(device).eval()
    return model, ckpt
