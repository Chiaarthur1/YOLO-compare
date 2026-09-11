"""Production-oriented RDD2022 inference wrapper for Ultralytics models."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np
import torch
from ultralytics import YOLO


RDD_CLASSES: Mapping[int, str] = {0: "D00", 1: "D10", 2: "D20", 3: "D40"}


@dataclass(frozen=True)
class Detection:
    class_id: int
    class_name: str
    confidence: float
    box: Tuple[int, int, int, int]


@dataclass(frozen=True)
class DetectionFrame:
    detections: List[Detection]
    latency_ms: float
    total_ms: float


class RoadDamageDetector:
    """Load YOLO weights with TensorRT preference and profile inference."""

    def __init__(
        self,
        yolov8_weights: Optional[str] = None,
        yolov10_weights: Optional[str] = None,
        model_type: str = "yolov8",
        device: Optional[str] = None,
        image_size: int = 640,
        confidence: float = 0.25,
        iou: float = 0.45,
        prefer_tensorrt: bool = True,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        if model_type not in {"yolov8", "yolov10", "both"}:
            raise ValueError("model_type must be yolov8, yolov10, or both")
        if image_size <= 0 or not 0.0 <= confidence <= 1.0:
            raise ValueError("image_size must be positive and confidence in [0, 1]")
        if not 0.0 <= iou <= 1.0:
            raise ValueError("iou must be in [0, 1]")

        self.logger = logger or logging.getLogger(__name__)
        self.device = device
        self.image_size = image_size
        self.confidence = confidence
        self.iou = iou
        self.models: Dict[str, YOLO] = {}
        configured = {"yolov8": yolov8_weights, "yolov10": yolov10_weights}
        names = ("yolov8", "yolov10") if model_type == "both" else (model_type,)

        for name in names:
            selected = self._select_weights(configured[name], prefer_tensorrt)
            try:
                self.models[name] = YOLO(str(selected))
            except Exception as exc:
                raise RuntimeError(f"Could not load {name} weights: {selected}") from exc
            self.logger.info(
                "model_loaded",
                extra={"event": "model_loaded", "model": name, "weights": str(selected)},
            )

    def detect(self, frame: np.ndarray) -> DetectionFrame:
        """Run inference and measure model time with CUDA synchronization."""
        if frame is None or frame.size == 0:
            raise ValueError("received an empty video frame")

        all_detections: List[Detection] = []
        model_latencies: List[float] = []
        total_started = perf_counter()

        for model_name, model in self.models.items():
            self._synchronize_cuda()
            started = perf_counter()
            try:
                results = model.predict(
                    source=frame,
                    imgsz=self.image_size,
                    conf=self.confidence,
                    iou=self.iou,
                    device=self.device,
                    verbose=False,
                )
                self._synchronize_cuda()
            except Exception as exc:
                raise RuntimeError(f"Inference failed for {model_name}") from exc

            measured_ms = (perf_counter() - started) * 1000.0
            reported_ms = self._reported_inference_ms(results)
            model_latencies.append(reported_ms if reported_ms > 0 else measured_ms)
            for result in results:
                all_detections.extend(self._extract_detections(result))

        total_ms = (perf_counter() - total_started) * 1000.0
        return DetectionFrame(all_detections, sum(model_latencies), total_ms)

    @staticmethod
    def _select_weights(weights: Optional[str], prefer_tensorrt: bool) -> Path:
        if not weights:
            raise ValueError("Model weights must be configured")
        configured = Path(weights).expanduser()
        suffix = configured.suffix.lower()
        candidates: List[Path] = []
        if prefer_tensorrt and suffix != ".engine":
            candidates.append(configured.with_suffix(".engine"))
        candidates.append(configured)
        if suffix == ".engine":
            candidates.extend(
                (configured.with_suffix(".pt"), configured.with_suffix(".onnx"))
            )
        elif suffix == ".pt":
            candidates.append(configured.with_suffix(".onnx"))
        elif suffix == ".onnx":
            candidates.append(configured.with_suffix(".pt"))
        for candidate in candidates:
            if candidate.is_file():
                return candidate
        searched = ", ".join(str(item) for item in candidates)
        raise FileNotFoundError(f"No model weights found. Checked: {searched}")

    @staticmethod
    def _synchronize_cuda() -> None:
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    @staticmethod
    def _reported_inference_ms(results: Any) -> float:
        return sum(
            float((getattr(result, "speed", {}) or {}).get("inference", 0.0))
            for result in results
        )

    @staticmethod
    def _extract_detections(result: Any) -> List[Detection]:
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            return []
        names = getattr(result, "names", {}) or {}
        coordinates = boxes.xyxy.cpu().numpy()
        confidences = boxes.conf.cpu().numpy()
        class_ids = boxes.cls.cpu().numpy().astype(int)
        detections: List[Detection] = []
        for row, confidence, class_id in zip(coordinates, confidences, class_ids):
            class_id = int(class_id)
            class_name = str(names.get(class_id, RDD_CLASSES.get(class_id, class_id)))
            if class_name not in RDD_CLASSES.values():
                continue
            x1, y1, x2, y2 = (int(round(value)) for value in row)
            detections.append(Detection(class_id, class_name, float(confidence), (x1, y1, x2, y2)))
        return detections