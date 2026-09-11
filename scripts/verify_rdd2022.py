"""Validate an RDD2022 Pascal VOC subset and create a factual dataset report."""

from __future__ import annotations

import argparse
import random
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import cv2


CLASSES = ("D00", "D10", "D20", "D40")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="Extracted RDD2022 folder")
    parser.add_argument("--output", default="data/verification")
    parser.add_argument("--samples", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def find_images(source: Path) -> Dict[str, Path]:
    images: Dict[str, Path] = {}
    for path in source.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            images.setdefault(path.stem, path)
    return images


def inspect_xml(annotation: Path, width: int, height: int) -> Tuple[List[dict], List[str]]:
    objects: List[dict] = []
    errors: List[str] = []
    try:
        root = ET.parse(annotation).getroot()
    except (ET.ParseError, OSError) as exc:
        return objects, [f"{annotation}: invalid XML ({exc})"]
    for index, item in enumerate(root.findall("object")):
        name = (item.findtext("name") or "").strip()
        box = item.find("bndbox")
        if box is None:
            errors.append(f"{annotation} object {index}: missing bndbox")
            continue
        try:
            xmin = float(box.findtext("xmin", "0"))
            ymin = float(box.findtext("ymin", "0"))
            xmax = float(box.findtext("xmax", "0"))
            ymax = float(box.findtext("ymax", "0"))
        except ValueError:
            errors.append(f"{annotation} object {index}: non-numeric coordinates")
            continue
        if not (0 <= xmin < xmax <= width and 0 <= ymin < ymax <= height):
            errors.append(f"{annotation} object {index}: box outside image bounds")
            continue
        objects.append({"name": name, "box": (int(xmin), int(ymin), int(xmax), int(ymax))})
    return objects, errors


def verify(source: Path, output: Path, sample_count: int, seed: int) -> None:
    if not source.is_dir():
        raise FileNotFoundError(f"Source folder does not exist: {source}")
    output.mkdir(parents=True, exist_ok=True)
    images = find_images(source)
    annotations = {path.stem: path for path in source.rglob("*.xml")}
    errors: List[str] = []
    class_counts: Counter[str] = Counter()
    box_sizes: List[Tuple[float, float]] = []
    valid_samples: List[Tuple[Path, List[dict]]] = []
    missing_xml = 0
    missing_image = 0
    checked_images = 0

    for stem, image in sorted(images.items()):
        checked_images += 1
        frame = cv2.imread(str(image))
        if frame is None:
            errors.append(f"{image}: unreadable image")
            continue
        height, width = frame.shape[:2]
        annotation = annotations.get(stem)
        if annotation is None:
            missing_xml += 1
            continue
        objects, object_errors = inspect_xml(annotation, width, height)
        errors.extend(object_errors)
        filtered = [item for item in objects if item["name"] in CLASSES]
        for item in filtered:
            class_counts[item["name"]] += 1
            x1, y1, x2, y2 = item["box"]
            box_sizes.append(((x2 - x1) / width, (y2 - y1) / height))
        if filtered:
            valid_samples.append((image, filtered))

    for stem, annotation in annotations.items():
        if stem not in images:
            missing_image += 1
            errors.append(f"{annotation}: matching image not found")

    rng = random.Random(seed)
    rng.shuffle(valid_samples)
    sample_dir = output / "samples"
    sample_dir.mkdir(exist_ok=True)
    saved_samples: List[str] = []
    for index, (image, objects) in enumerate(valid_samples[:sample_count]):
        frame = cv2.imread(str(image))
        for item in objects:
            x1, y1, x2, y2 = item["box"]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 220, 0), 2)
            cv2.putText(frame, item["name"], (x1, max(20, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 0), 2)
        target = sample_dir / f"sample_{index + 1:02d}.png"
        cv2.imwrite(str(target), frame)
        saved_samples.append(str(target))

    mean_width = sum(item[0] for item in box_sizes) / len(box_sizes) if box_sizes else 0.0
    mean_height = sum(item[1] for item in box_sizes) / len(box_sizes) if box_sizes else 0.0
    total_objects = sum(class_counts.values())
    lines = [
        "# RDD2022 dataset verification report",
        "",
        f"- Source: `{source}`",
        f"- Images scanned: {checked_images}",
        f"- XML files scanned: {len(annotations)}",
        f"- Images without XML: {missing_xml}",
        f"- XML files without image: {missing_image}",
        f"- Valid annotated images used for sampling: {len(valid_samples)}",
        f"- Invalid or suspicious records: {len(errors)}",
        "",
        "## Object statistics",
        "",
        "| Class | Objects | Share |",
        "|---|---:|---:|",
    ]
    for name in CLASSES:
        share = class_counts[name] / total_objects * 100 if total_objects else 0.0
        lines.append(f"| {name} | {class_counts[name]} | {share:.2f}% |")
    lines.extend(
        [
            "",
            f"Mean normalized box width: {mean_width:.6f}",
            f"Mean normalized box height: {mean_height:.6f}",
            "",
            "## Control visualizations",
            "",
        ]
    )
    lines.extend(f"- `{path}`" for path in saved_samples)
    lines.extend(["", "## Validation errors", ""])
    lines.extend(f"- {error}" for error in errors[:200])
    if len(errors) > 200:
        lines.append(f"- ... {len(errors) - 200} more errors omitted")
    (output / "dataset_verification_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved report: {output / 'dataset_verification_report.md'}")
    print(f"Scanned {checked_images} images and {total_objects} supported objects")
    print(f"Validation errors: {len(errors)}")


if __name__ == "__main__":
    arguments = parse_args()
    verify(
        Path(arguments.source).expanduser().resolve(),
        Path(arguments.output).expanduser().resolve(),
        arguments.samples,
        arguments.seed,
    )