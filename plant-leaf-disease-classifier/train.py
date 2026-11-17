"""Fine-tune an ImageNet model on the split created by split_dataset.py.

Training runs in two stages:
  1. only the new classifier head is trained, backbone frozen
  2. the last stages of the backbone are unfrozen and trained with a lower LR

    python train.py
    python train.py --arch efficientnet_b0 --head-epochs 2 --finetune-epochs 8
"""
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from plant_disease.data import build_loaders, class_weights
from plant_disease.engine import run_epoch
from plant_disease.model import SUPPORTED_ARCHS, build_model, freeze_backbone, save_checkpoint, unfreeze_top


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=Path, default=Path("data"))
    p.add_argument("--arch", choices=SUPPORTED_ARCHS, default="resnet18")
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--head-epochs", type=int, default=2)
    p.add_argument("--finetune-epochs", type=int, default=6)
    p.add_argument("--head-lr", type=float, default=1e-3)
    p.add_argument("--finetune-lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--label-smoothing", type=float, default=0.1)
    p.add_argument("--patience", type=int, default=3, help="early stopping patience (fine-tune stage)")
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=Path, default=Path("models/best_plant_disease_model.pth"))
    return p.parse_args()


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def main():
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds, train_loader, val_loader = build_loaders(args.data, args.image_size, args.batch_size, args.workers)
    classes = train_ds.classes
    print(f"device={device}  classes={len(classes)}  train={len(train_ds)}  val={len(val_loader.dataset)}")

    model = build_model(args.arch, len(classes)).to(device)

    # Early blight has roughly half as many images as the other classes.
    weights = class_weights(train_ds).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=args.label_smoothing)

    history = []
    best_val_loss = float("inf")
    best_epoch = 0
    epoch = 0

    def step(stage, optimizer, scheduler=None):
        nonlocal best_val_loss, best_epoch, epoch
        epoch += 1
        start = time.time()
        train_loss, train_acc = run_epoch(model, train_loader, criterion, device, optimizer)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, device)
        if scheduler is not None:
            scheduler.step()

        history.append({
            "epoch": epoch, "stage": stage,
            "train_loss": train_loss, "train_acc": train_acc,
            "val_loss": val_loss, "val_acc": val_acc,
            "lr": optimizer.param_groups[0]["lr"],
        })
        improved = val_loss < best_val_loss
        print(
            f"[{stage:8s}] epoch {epoch:2d}  "
            f"train {train_loss:.4f} / {train_acc:.2%}   val {val_loss:.4f} / {val_acc:.2%}   "
            f"{time.time() - start:.0f}s{'  *' if improved else ''}"
        )
        if improved:
            best_val_loss, best_epoch = val_loss, epoch
            save_checkpoint(
                args.out, model, args.arch, classes, args.image_size,
                extra={"metrics": {"val_loss": val_loss, "val_acc": val_acc, "epoch": epoch}},
            )
        return improved

    # Stage 1: train only the classifier head.
    freeze_backbone(model)
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad), lr=args.head_lr, weight_decay=args.weight_decay
    )
    for _ in range(args.head_epochs):
        step("head", optimizer)

    # Stage 2: unfreeze the top of the backbone. The head gets a slightly
    # higher LR than the pretrained layers.
    unfreeze_top(model, args.arch)
    head_ids = {id(p) for p in (model.fc if hasattr(model, "fc") else model.classifier).parameters()}
    backbone = [p for p in model.parameters() if p.requires_grad and id(p) not in head_ids]
    head = [p for p in model.parameters() if id(p) in head_ids]
    optimizer = torch.optim.AdamW(
        [{"params": backbone, "lr": args.finetune_lr}, {"params": head, "lr": args.finetune_lr * 5}],
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.finetune_epochs, 1))

    stale = 0
    for _ in range(args.finetune_epochs):
        stale = 0 if step("finetune", optimizer, scheduler) else stale + 1
        if stale >= args.patience:
            print(f"no improvement for {args.patience} epochs, stopping early")
            break

    reports = Path("reports")
    reports.mkdir(exist_ok=True)
    with open(reports / "history.json", "w") as f:
        json.dump({"args": {k: str(v) for k, v in vars(args).items()}, "history": history}, f, indent=2)
    plot_history(history, reports / "training_curves.png")

    print(f"\nbest val loss {best_val_loss:.4f} at epoch {best_epoch}, saved to {args.out}")


def plot_history(history, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = [h["epoch"] for h in history]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.8))
    ax1.plot(epochs, [h["train_loss"] for h in history], marker="o", label="train")
    ax1.plot(epochs, [h["val_loss"] for h in history], marker="o", label="val")
    ax1.set_title("Loss")
    ax2.plot(epochs, [h["train_acc"] for h in history], marker="o", label="train")
    ax2.plot(epochs, [h["val_acc"] for h in history], marker="o", label="val")
    ax2.set_title("Accuracy")

    head_end = max((h["epoch"] for h in history if h["stage"] == "head"), default=None)
    for ax in (ax1, ax2):
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.3)
        if head_end:
            ax.axvline(head_end + 0.5, color="grey", linestyle="--", linewidth=1)
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
