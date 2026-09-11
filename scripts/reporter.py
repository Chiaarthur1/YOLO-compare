"""Structured JSON reporting for road damage detections."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, Mapping, Optional

from detector import Detection, DetectionFrame


class RoadDamageReporter:
    def __init__(
        self,
        class_thresholds: Optional[Mapping[str, float]] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.thresholds = dict(class_thresholds or {"D00": 0.25, "D10": 0.25, "D20": 0.70, "D40": 0.70})
        self.logger = logger or logging.getLogger(__name__)

    def build_report(self, frame_result: DetectionFrame, fps: float) -> Dict[str, Any]:
        detections = frame_result.detections
        critical = any(
            detection.class_name in {"D20", "D40"}
            and detection.confidence > self.thresholds.get(detection.class_name, 0.70)
            for detection in detections
        )
        warning = any(detection.class_name in {"D00", "D10"} for detection in detections)
        status = "CRITICAL" if critical else "WARNING" if warning else "OK"
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "status": status,
            "fps": round(max(0.0, fps), 2),
            "latency_ms": round(frame_result.latency_ms, 3),
            "objects": [self._serialize_detection(item) for item in detections],
        }

    def emit(self, frame_result: DetectionFrame, fps: float) -> Dict[str, Any]:
        report = self.build_report(frame_result, fps)
        self.logger.info("detection_report", extra={"event": "detection_report", "report": report})
        if report["status"] == "CRITICAL":
            self.logger.critical("critical_road_damage", extra={"event": "critical_road_damage", "report": report})
        return report

    @staticmethod
    def _serialize_detection(detection: Detection) -> Dict[str, Any]:
        return {
            "class": detection.class_name,
            "confidence": round(detection.confidence, 4),
            "bbox": {"x1": detection.box[0], "y1": detection.box[1], "x2": detection.box[2], "y2": detection.box[3]},
        }