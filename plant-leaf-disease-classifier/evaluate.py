"""Evaluate a checkpoint on the held-out test split.

Writes metrics and figures to reports/:
  metrics.json, confusion_matrix.png, misclassified.png, robustness.json

    python evaluate.py
    python evaluate.py --robustness     # also test under blur/noise/lighting changes
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader
from torchvision import datasets

from plant_disease.data import eval_transforms
from plant_disease.engine import collect_predictions
from plant_disease.model import load_checkpoint
from plant_disease.predictor import pretty_label


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=Path, default=Path("models/best_plant_disease_model.pth"))
    p.add_argument("--data", type=Path, default=Path("data/test"))
    p.add_argument("--reports", type=Path, default=Path("reports"))
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--robustness", action="store_true")
    return p.parse_args()


# Simple corruptions that roughly mimic phone photos taken in a field.
def _jpeg(img, quality=20):
    from io import BytesIO
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def _noise(img, sigma=20):
    arr = np.asarray(img).astype(np.float32)
    rng = np.random.default_rng(0)
    arr += rng.normal(0, sigma, arr.shape)
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


CORRUPTIONS = {
    "clean": lambda img: img,
    "gaussian_blur": lambda img: img.filter(ImageFilter.GaussianBlur(radius=2)),
    "gaussian_noise": _noise,
    "dark": lambda img: ImageEnhance.Brightness(img).enhance(0.5),
    "bright": lambda img: ImageEnhance.Brightness(img).enhance(1.6),
    "low_contrast": lambda img: ImageEnhance.Contrast(img).enhance(0.5),
    "jpeg_q20": _jpeg,
}


def plot_confusion(cm, labels, path):
    cm_norm = cm / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    im = ax.imshow(cm_norm, cmap="Greens", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)), labels, rotation=30, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    for i in range(len(labels)):
        for j in range(len(labels)):
            color = "white" if cm_norm[i, j] > 0.6 else "black"
            ax.text(j, i, f"{cm[i, j]}\n{cm_norm[i, j]:.1%}", ha="center", va="center", fontsize=8, color=color)
    fig.colorbar(im, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_mistakes(dataset, probs, labels, classes, path, limit=12):
    preds = probs.argmax(1)
    wrong = torch.nonzero(preds != labels).flatten().tolist()
    if not wrong:
        return 0
    # Most confident mistakes first; those are the interesting ones.
    wrong.sort(key=lambda i: -probs[i].max().item())
    wrong = wrong[:limit]

    cols = 4
    rows = (len(wrong) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3.3), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, i in zip(axes.flat, wrong):
        path_i, _ = dataset.samples[i]
        with Image.open(path_i) as img:
            ax.imshow(img.convert("RGB"))
        ax.set_title(
            f"true: {pretty_label(classes[labels[i]])}\n"
            f"pred: {pretty_label(classes[preds[i]])} ({probs[i].max():.0%})",
            fontsize=8,
        )
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return len(wrong)


def main():
    args = parse_args()
    args.reports.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, ckpt = load_checkpoint(args.checkpoint, device)
    classes = ckpt["classes"]
    image_size = ckpt.get("image_size", 224)

    dataset = datasets.ImageFolder(args.data, transform=eval_transforms(image_size))
    if dataset.classes != classes:
        raise ValueError(f"test classes {dataset.classes} don't match checkpoint classes {classes}")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    probs, labels = collect_predictions(model, loader, device)
    preds = probs.argmax(1)

    accuracy = (preds == labels).float().mean().item()
    macro_f1 = f1_score(labels, preds, average="macro")
    cm = confusion_matrix(labels, preds, labels=list(range(len(classes))))
    report = classification_report(
        labels, preds, labels=list(range(len(classes))),
        target_names=[pretty_label(c) for c in classes], digits=4, zero_division=0, output_dict=True,
    )

    print(f"test images: {len(dataset)}")
    print(f"accuracy:    {accuracy:.4f}")
    print(f"macro F1:    {macro_f1:.4f}\n")
    print(classification_report(
        labels, preds, labels=list(range(len(classes))),
        target_names=[pretty_label(c) for c in classes], digits=4, zero_division=0,
    ))

    with open(args.reports / "metrics.json", "w") as f:
        json.dump({
            "checkpoint": str(args.checkpoint),
            "arch": ckpt.get("arch", "resnet18"),
            "test_images": len(dataset),
            "accuracy": accuracy,
            "macro_f1": macro_f1,
            "per_class": report,
            "confusion_matrix": cm.tolist(),
        }, f, indent=2)

    plot_confusion(cm, [pretty_label(c) for c in classes], args.reports / "confusion_matrix.png")
    n = plot_mistakes(dataset, probs, labels, classes, args.reports / "misclassified.png")
    print(f"figures written to {args.reports}/ ({n} misclassified examples shown)")

    if args.robustness:
        base_tf = eval_transforms(image_size)
        results = {}
        for name, corrupt in CORRUPTIONS.items():
            ds = datasets.ImageFolder(args.data, transform=lambda img, c=corrupt: base_tf(c(img.convert("RGB"))))
            p, y = collect_predictions(model, DataLoader(ds, batch_size=args.batch_size, num_workers=args.workers), device)
            results[name] = {
                "accuracy": (p.argmax(1) == y).float().mean().item(),
                "macro_f1": f1_score(y, p.argmax(1), average="macro"),
            }
            print(f"  {name:15s} acc {results[name]['accuracy']:.4f}   macro F1 {results[name]['macro_f1']:.4f}")
        with open(args.reports / "robustness.json", "w") as f:
            json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
