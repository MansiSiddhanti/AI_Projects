"""Classify one or more leaf images from the command line.

    python predict.py path/to/leaf.jpg
    python predict.py leaves/*.jpg --gradcam out/
"""
import argparse
from pathlib import Path

from PIL import Image

from plant_disease.gradcam import overlay
from plant_disease.predictor import Predictor


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("images", nargs="+", type=Path)
    p.add_argument("--checkpoint", type=Path, default=Path("models/best_plant_disease_model.pth"))
    p.add_argument("--gradcam", type=Path, metavar="DIR", help="save Grad-CAM overlays to this folder")
    p.add_argument("--top", type=int, default=3, help="number of classes to list per image")
    args = p.parse_args()

    predictor = Predictor(args.checkpoint)
    if args.gradcam:
        args.gradcam.mkdir(parents=True, exist_ok=True)

    for path in args.images:
        try:
            with Image.open(path) as img:
                image = img.convert("RGB")
        except (FileNotFoundError, OSError) as err:
            print(f"{path}: could not read image ({err})")
            continue

        result = predictor.predict(image, explain=args.gradcam is not None)
        print(f"{path}: {result.label} ({result.confidence:.1%})")
        ranked = sorted(result.probabilities.items(), key=lambda kv: -kv[1])[: args.top]
        for label, prob in ranked[1:]:
            print(f"    {label:20s} {prob:.1%}")

        if args.gradcam:
            out = args.gradcam / f"{path.stem}_gradcam.png"
            overlay(image, result.heatmap).save(out)
            print(f"    grad-cam -> {out}")


if __name__ == "__main__":
    main()
