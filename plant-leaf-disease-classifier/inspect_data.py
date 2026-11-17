"""Print image counts per split/class and save a grid of sample images.

    python inspect_data.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from plant_disease.data import IMAGE_EXTENSIONS
from plant_disease.predictor import pretty_label

DATA_DIR = Path("data")
OUT = Path("reports/samples.png")
PER_CLASS = 4


def images_in(folder: Path):
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)


def main():
    classes = sorted(p.name for p in (DATA_DIR / "train").iterdir() if p.is_dir())

    header = f"{'class':28s}" + "".join(f"{s:>8s}" for s in ("train", "val", "test"))
    print(header)
    print("-" * len(header))
    for c in classes:
        counts = [len(images_in(DATA_DIR / s / c)) if (DATA_DIR / s / c).exists() else 0
                  for s in ("train", "val", "test")]
        print(f"{c:28s}" + "".join(f"{n:8d}" for n in counts))

    fig, axes = plt.subplots(len(classes), PER_CLASS, figsize=(PER_CLASS * 2.2, len(classes) * 2.3))
    for row, c in zip(axes, classes):
        paths = images_in(DATA_DIR / "train" / c)[:: max(1, len(images_in(DATA_DIR / "train" / c)) // PER_CLASS)]
        for ax, path in zip(row, paths):
            with Image.open(path) as img:
                ax.imshow(img.convert("RGB"))
            ax.set_xticks([])
            ax.set_yticks([])
        row[0].set_ylabel(pretty_label(c), fontsize=9)
    fig.tight_layout()
    OUT.parent.mkdir(exist_ok=True)
    fig.savefig(OUT, dpi=110)
    print(f"\nsample grid saved to {OUT}")


if __name__ == "__main__":
    main()
