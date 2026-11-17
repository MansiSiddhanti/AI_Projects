"""Split the raw PlantVillage images into train/val/test folders.

By default images are grouped by physical leaf (using leaf-map.json from the
PlantVillage repo) so no leaf appears in more than one split.

    python split_dataset.py
    python split_dataset.py --classes Tomato___healthy Tomato___Late_blight
    python split_dataset.py --no-group          # old behaviour, random per image
"""
import argparse
import csv
import shutil
from pathlib import Path

from plant_disease.data import (
    IMAGE_EXTENSIONS,
    count_leaf_overlap,
    grouped_split,
    leaf_id,
    load_leaf_map,
)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", type=Path, default=Path("plantvillage-source/raw/color"))
    p.add_argument("--leaf-map", type=Path, default=Path("plantvillage-source/leaf-map.json"))
    p.add_argument("--out", type=Path, default=Path("data"))
    p.add_argument("--classes", nargs="*", help="class folders to use (default: every folder in --source)")
    p.add_argument("--val", type=float, default=0.15)
    p.add_argument("--test", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-group", action="store_true", help="split per image instead of per leaf")
    return p.parse_args()


def main():
    args = parse_args()
    leaf_map = load_leaf_map(args.leaf_map)
    ratios = {"train": 1 - args.val - args.test, "val": args.val, "test": args.test}

    classes = args.classes or sorted(p.name for p in args.source.iterdir() if p.is_dir())

    # Start from a clean output folder so stale images from a previous split
    # can't leak into the new one.
    for split in ratios:
        if (args.out / split).exists():
            shutil.rmtree(args.out / split)

    manifest = []
    for class_name in classes:
        class_dir = args.source / class_name
        if not class_dir.is_dir():
            raise FileNotFoundError(f"missing class folder: {class_dir}")

        files = sorted(p.name for p in class_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
        if args.no_group:
            groups = files
        else:
            groups = [leaf_id(class_name, f, leaf_map) for f in files]

        assignment = grouped_split(files, groups, ratios, seed=args.seed)
        group_of = dict(zip(files, groups))

        print(f"{class_name}: {len(files)} images, {len(set(groups))} leaves")
        for split, names in assignment.items():
            dest = args.out / split / class_name
            dest.mkdir(parents=True, exist_ok=True)
            for name in names:
                shutil.copy2(class_dir / name, dest / name)
                manifest.append((split, class_name, name, group_of[name]))
            print(f"  {split:5s} {len(names)}")

    args.out.mkdir(parents=True, exist_ok=True)
    with open(args.out / "split_manifest.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["split", "class", "file", "leaf_id"])
        writer.writerows(manifest)

    overlap = count_leaf_overlap(args.out, leaf_map)
    print(
        f"\nval images sharing a leaf with train:  {overlap['val_leaked']}/{overlap['val_images']}\n"
        f"test images sharing a leaf with train: {overlap['test_leaked']}/{overlap['test_images']}"
    )


if __name__ == "__main__":
    main()
