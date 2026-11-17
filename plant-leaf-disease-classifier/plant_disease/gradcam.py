from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image


class GradCAM:
    """Grad-CAM (Selvaraju et al., 2017) for a single conv layer.

    Hooks the layer's activations on the forward pass and its gradients on the
    backward pass, then weights each channel by its mean gradient.
    """

    def __init__(self, model: nn.Module, layer: nn.Module):
        self.model = model
        self._acts = None
        self._grads = None
        self._handles = [
            layer.register_forward_hook(self._save_acts),
            layer.register_full_backward_hook(self._save_grads),
        ]

    def _save_acts(self, module, inputs, output):
        self._acts = output.detach()

    def _save_grads(self, module, grad_input, grad_output):
        self._grads = grad_output[0].detach()

    def __call__(self, x: torch.Tensor, class_idx: int | None = None) -> tuple[np.ndarray, torch.Tensor]:
        """Return a heatmap in [0, 1] with the input's spatial size, plus the logits."""
        self.model.zero_grad(set_to_none=True)
        x = x.clone().requires_grad_(True)
        logits = self.model(x)
        if class_idx is None:
            class_idx = int(logits.argmax(dim=1))
        logits[0, class_idx].backward()

        weights = self._grads.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * self._acts).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
        cam -= cam.min()
        if cam.max() > 0:
            cam /= cam.max()
        return cam.cpu().numpy(), logits.detach()

    def remove(self):
        for h in self._handles:
            h.remove()


def overlay(image: Image.Image, heatmap: np.ndarray, alpha: float = 0.45) -> Image.Image:
    """Blend a [0, 1] heatmap on top of an RGB image using the jet colormap."""
    from matplotlib import colormaps

    heat = Image.fromarray(np.uint8(heatmap * 255)).resize(image.size, Image.BILINEAR)
    colored = colormaps["jet"](np.asarray(heat) / 255.0)[..., :3]
    colored = Image.fromarray(np.uint8(colored * 255))
    return Image.blend(image.convert("RGB"), colored, alpha)
