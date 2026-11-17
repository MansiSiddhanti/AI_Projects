from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def train_transforms(image_size: int = 224) -> transforms.Compose:
    # PlantVillage photos are taken on a plain background under similar
    # lighting, so the colour/crop augmentation is there to make the model
    # a bit less dependent on those conditions.
    return transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.2, hue=0.02),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def eval_transforms(image_size: int = 224) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def build_loaders(data_dir: Path, image_size: int, batch_size: int, num_workers: int = 2):
    data_dir = Path(data_dir)
    train_ds = datasets.ImageFolder(data_dir / "train", transform=train_transforms(image_size))
    val_ds = datasets.ImageFolder(data_dir / "val", transform=eval_transforms(image_size))

    if train_ds.classes != val_ds.classes:
        raise ValueError(f"train/val class folders differ: {train_ds.classes} vs {val_ds.classes}")

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, persistent_workers=num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, persistent_workers=num_workers > 0,
    )
    return train_ds, train_loader, val_loader


def class_weights(dataset: datasets.ImageFolder) -> torch.Tensor:
    """Inverse-frequency weights, normalised so they average to 1."""
    counts = torch.bincount(torch.tensor(dataset.targets), minlength=len(dataset.classes)).float()
    weights = counts.sum() / (len(counts) * counts)
    return weights


# --- leaf grouping -----------------------------------------------------------
#
# PlantVillage contains several photos of the same physical leaf. The dataset
# authors published leaf-map.json, which maps an image identifier to a leaf id.
# If photos of one leaf end up in both train and test, the test score mostly
# measures memorisation, so the split is done per leaf rather than per image.

def load_leaf_map(path: Path) -> dict[str, list[str]]:
    with open(path) as f:
        return json.load(f)


def image_key(filename: str) -> str:
    """Turn a file name into the lookup key used by leaf-map.json.

    Mirrors the logic in the official PlantVillage loader, e.g.
    '0a1b..___RS_HL 0105.JPG' -> 'rs_hl 0105'.
    """
    stem = Path(filename).stem.replace("_final_masked", "")
    stem = stem.split("___")[-1]
    stem = stem.split("copy")[0]
    return stem.strip().lower()


def leaf_id(class_name: str, filename: str, leaf_map: dict[str, list[str]]) -> str:
    candidates = leaf_map.get(image_key(filename), [])
    for candidate in candidates:
        if candidate.split(":::")[0] == class_name:
            return candidate
    # No usable mapping: treat the image as its own group.
    return f"{class_name}:::single:::{filename}"


def grouped_split(
    files: list[str],
    groups: list[str],
    ratios: dict[str, float],
    seed: int = 42,
) -> dict[str, list[str]]:
    """Assign whole groups to splits so that each split gets roughly its share of images."""
    by_group: dict[str, list[str]] = defaultdict(list)
    for f, g in zip(files, groups):
        by_group[g].append(f)

    group_ids = sorted(by_group)
    random.Random(seed).shuffle(group_ids)

    total = len(files)
    names = list(ratios)
    targets = {name: ratios[name] * total for name in names}
    out: dict[str, list[str]] = {name: [] for name in names}

    # Fill the smaller splits first and put everything that is left into the
    # first (largest) one. Filling greedily up to the target keeps the sizes
    # within one group of the requested ratio.
    order = sorted(names, key=lambda n: ratios[n])
    largest = order[-1]
    idx = 0
    for name in order[:-1]:
        while idx < len(group_ids) and len(out[name]) < targets[name]:
            out[name].extend(by_group[group_ids[idx]])
            idx += 1
    for g in group_ids[idx:]:
        out[largest].extend(by_group[g])
    return out


def count_leaf_overlap(split_dir: Path, leaf_map: dict[str, list[str]]) -> dict[str, int]:
    """How many val/test images share a leaf with some training image."""
    split_dir = Path(split_dir)

    def ids(split: str) -> list[str]:
        result = []
        for class_dir in sorted((split_dir / split).iterdir()):
            for p in class_dir.iterdir():
                if p.suffix.lower() in IMAGE_EXTENSIONS:
                    result.append(leaf_id(class_dir.name, p.name, leaf_map))
        return result

    train_ids = {i for i in ids("train") if ":::single:::" not in i}
    report = {}
    for split in ("val", "test"):
        split_ids = ids(split)
        report[f"{split}_images"] = len(split_ids)
        report[f"{split}_leaked"] = sum(i in train_ids for i in split_ids)
    return report
