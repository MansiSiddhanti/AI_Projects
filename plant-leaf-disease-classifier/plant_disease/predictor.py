from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .data import eval_transforms
from .gradcam import GradCAM
from .model import load_checkpoint, target_layer


def pretty_label(class_name: str) -> str:
    """'Tomato___Early_blight' -> 'Early blight'."""
    name = class_name.split("___")[-1].replace("_", " ").strip()
    return name[:1].upper() + name[1:]


@dataclass
class Prediction:
    label: str
    class_name: str
    confidence: float
    probabilities: dict[str, float]
    heatmap: np.ndarray | None = None


class Predictor:
    def __init__(self, checkpoint: str | Path, device: str | None = None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model, ckpt = load_checkpoint(checkpoint, self.device)
        self.arch = ckpt.get("arch", "resnet18")
        self.classes: list[str] = ckpt["classes"]
        self.image_size: int = ckpt.get("image_size", 224)
        self.metrics: dict = ckpt.get("metrics", {})
        self.transform = eval_transforms(self.image_size)

    def _prepare(self, image: Image.Image) -> torch.Tensor:
        return self.transform(image.convert("RGB")).unsqueeze(0).to(self.device)

    def predict(self, image: Image.Image, explain: bool = False) -> Prediction:
        x = self._prepare(image)
        heatmap = None

        if explain:
            cam = GradCAM(self.model, target_layer(self.model, self.arch))
            try:
                heatmap, logits = cam(x)
            finally:
                cam.remove()
        else:
            with torch.no_grad():
                logits = self.model(x)

        probs = torch.softmax(logits, dim=1)[0].cpu()
        idx = int(probs.argmax())
        return Prediction(
            label=pretty_label(self.classes[idx]),
            class_name=self.classes[idx],
            confidence=float(probs[idx]),
            probabilities={pretty_label(c): float(p) for c, p in zip(self.classes, probs)},
            heatmap=heatmap,
        )
