from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def run_epoch(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device,
              optimizer: torch.optim.Optimizer | None = None) -> tuple[float, float]:
    """One pass over `loader`. Trains if an optimizer is given, otherwise evaluates."""
    training = optimizer is not None
    model.train(training)

    total_loss, correct, seen = 0.0, 0, 0
    with torch.set_grad_enabled(training):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * labels.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            seen += labels.size(0)

    return total_loss / seen, correct / seen


@torch.no_grad()
def collect_predictions(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    all_probs, all_labels = [], []
    for images, labels in loader:
        logits = model(images.to(device))
        all_probs.append(torch.softmax(logits, dim=1).cpu())
        all_labels.append(labels)
    return torch.cat(all_probs), torch.cat(all_labels)
