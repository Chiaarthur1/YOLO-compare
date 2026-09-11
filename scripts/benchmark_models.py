"""Train and compare YOLOv8n and YOLOv10n on one YOLO-format dataset."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Path to data.yaml")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="cpu", help="cpu, 0, 1, ...")
    parser.add_argument("--output", default="runs/comparison")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def train_and_validate(model_name: str, args: argparse.Namespace) -> Dict[str, Any]:
    model = YOLO(f"{model_name}.pt")
    started = time.perf_counter()
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.output,
        name=model_name,
        exist_ok=True,
        pretrained=True,
        seed=args.seed,
        deterministic=True,
        verbose=False,
    )
    training_seconds = time.perf_counter() - started
    save_dir = Path(model.trainer.save_dir)
    metrics = model.val(
        data=args.data,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        split="val",
        plots=True,
    )
    box = metrics.box
    precision = float(box.mp)
    recall = float(box.mr)
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    info = model.info(verbose=False)
    parameters = None
    gflops = None
    if isinstance(info, dict):
        parameters = info.get("parameters") or info.get("params")
        gflops = info.get("GFLOPs") or info.get("gflops")
    elif isinstance(info, (tuple, list)) and len(info) >= 4:
        parameters = info[1]
        gflops = info[3]
    result: Dict[str, Any] = {
        "model": model_name,
        "weights": str(save_dir / "weights" / "best.pt"),
        "training_seconds": round(training_seconds, 2),
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
        "map50": round(float(box.map50), 6),
        "map50_95": round(float(box.map), 6),
        "parameters": int(parameters) if parameters is not None else None,
        "parameters_millions": round(float(parameters) / 1_000_000, 6) if parameters is not None else None,
        "gflops": round(float(gflops), 6) if gflops is not None else None,
        "fitness": round(float(metrics.fitness), 6),
        "speed_ms": {key: round(float(value), 4) for key, value in metrics.speed.items()},
        "per_class_map50": [round(float(value), 6) for value in box.maps],
    }
    return result


def main() -> int:
    args = parse_args()
    results = [train_and_validate(name, args) for name in ("yolov8n", "yolov10n")]
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    report = output / "comparison.json"
    report.write_text(json.dumps(results, indent=2), encoding="utf-8")
    config = {
        "data": str(Path(args.data).resolve()),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": args.device,
        "seed": args.seed,
        "models": ["yolov8n", "yolov10n"],
    }
    (output / "experiment_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))
    print(f"Saved comparison report to {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())