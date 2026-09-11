"""Convert an RDD2022 Pascal VOC subset into an Ultralytics YOLO dataset."""

from __future__ import annotations

import argparse
import random
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


CLASSES = ("D00", "D10", "D20", "D40")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="RDD2022 country folder or extracted archive")
    parser.add_argument("--output", default="data/rdd2022", help="YOLO dataset output folder")
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-images", type=int, default=0, help="Optional limit for a quick experiment")
    return parser.parse_args()


def find_images(source: Path) -> Dict[str, Path]:
    images: Dict[str, Path] = {}
    for path in source.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            images.setdefault(path.stem, path)
    return images


def read_boxes(annotation: Path, image_width: int, image_height: int) -> List[str]:
    root = ET.parse(annotation).getroot()
    labels: List[str] = []
    for obj in root.findall("object"):
        class_name = (obj.findtext("name") or "").strip()
        if class_name not in CLASSES:
            continue
        box = obj.find("bndbox")
        if box is None:
            continue
        xmin = float(box.findtext("xmin", "0"))
        ymin = float(box.findtext("ymin", "0"))
        xmax = float(box.findtext("xmax", "0"))
        ymax = float(box.findtext("ymax", "0"))
        xmin = max(0.0, min(xmin, image_width))
        xmax = max(0.0, min(xmax, image_width))
        ymin = max(0.0, min(ymin, image_height))
        ymax = max(0.0, min(ymax, image_height))
        if xmax <= xmin or ymax <= ymin:
            continue
        center_x = ((xmin + xmax) / 2.0) / image_width
        center_y = ((ymin + ymax) / 2.0) / image_height
        width = (xmax - xmin) / image_width
        height = (ymax - ymin) / image_height
        labels.append(f"{CLASSES.index(class_name)} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}")
    return labels


def image_size(image: Path) -> Tuple[int, int]:
    from PIL import Image

    with Image.open(image) as opened:
        return opened.size


def prepare(source: Path, output: Path, val_ratio: float, seed: int, max_images: int) -> None:
    if not source.is_dir():
        raise FileNotFoundError(f"Source folder does not exist: {source}")
    if not 0.0 < val_ratio < 1.0:
        raise ValueError("val-ratio must be between 0 and 1")

    images = find_images(source)
    samples: List[Tuple[Path, Path]] = []
    for annotation in source.rglob("*.xml"):
        image = images.get(annotation.stem)
        if image is not None:
            samples.append((image, annotation))
    if not samples:
        raise ValueError("No image/XML pairs found in source folder")

    random.Random(seed).shuffle(samples)
    if max_images > 0:
        samples = samples[:max_images]
    split_at = max(1, int(len(samples) * (1.0 - val_ratio)))
    splits = {"train": samples[:split_at], "val": samples[split_at:]}
    if not splits["val"]:
        raise ValueError("Dataset is too small for the requested validation split")

    if output.exists():
        raise FileExistsError(f"Output already exists; remove it manually before rerunning: {output}")
    for split in splits:
        (output / "images" / split).mkdir(parents=True)
        (output / "labels" / split).mkdir(parents=True)

    for split, split_samples in splits.items():
        for index, (image, annotation) in enumerate(split_samples):
            target_name = f"{index:06d}{image.suffix.lower()}"
            target_image = output / "images" / split / target_name
            shutil.copy2(image, target_image)
            width, height = image_size(image)
            labels = read_boxes(annotation, width, height)
            (output / "labels" / split / f"{target_image.stem}.txt").write_text(
                "\n".join(labels) + ("\n" if labels else ""), encoding="utf-8"
            )

    yaml_text = "\n".join(
        (
            "path: .",
            "train: images/train",
            "val: images/val",
            "names:",
            *[f"  {index}: {name}" for index, name in enumerate(CLASSES)],
            "",
        )
    )
    (output / "data.yaml").write_text(yaml_text, encoding="utf-8")
    print(f"Prepared {len(samples)} images: {len(splits['train'])} train, {len(splits['val'])} val")
    print(f"Dataset YAML: {output / 'data.yaml'}")


if __name__ == "__main__":
    arguments = parse_args()
    prepare(
        Path(arguments.source).expanduser().resolve(),
        Path(arguments.output).expanduser().resolve(),
        arguments.val_ratio,
        arguments.seed,
        arguments.max_images,
    )