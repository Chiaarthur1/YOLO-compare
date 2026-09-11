"""Threaded real-time RDD2022 road damage pipeline."""

from __future__ import annotations

import argparse
import json
import logging
import queue
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Union

import cv2
import numpy as np
import yaml

from detector import Detection, RoadDamageDetector
from reporter import RoadDamageReporter


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: Dict[str, Any] = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created)),
            "level": record.levelname,
            "message": record.getMessage(),
            "logger": record.name,
        }
        for key in ("event", "model", "weights", "report", "attempt"):
            value = getattr(record, key, None)
            if value is not None:
                payload[key] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=True, separators=(",", ":"))


class CriticalConsoleFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        return f"[CRITICAL] {record.getMessage()}"


class CriticalFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return record.levelno >= logging.CRITICAL


def configure_logging(config: Dict[str, Any]) -> logging.Logger:
    settings = config.get("logging", {})
    logger = logging.getLogger("road_ai")
    logger.setLevel(getattr(logging, str(settings.get("level", "INFO")).upper(), logging.INFO))
    logger.handlers.clear()
    file_handler = logging.FileHandler(settings.get("file", "road_ai.log"), encoding="utf-8")
    file_handler.setFormatter(JsonFormatter())
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.CRITICAL)
    console_handler.addFilter(CriticalFilter())
    console_handler.setFormatter(CriticalConsoleFormatter())
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    logger.propagate = False
    return logger


class VideoCaptureWorker:
    """Continuously capture frames, dropping stale frames under load."""

    def __init__(self, source: str, camera: Dict[str, Any], logger: logging.Logger) -> None:
        self.source = source
        self.camera = camera
        self.logger = logger
        self.frames: queue.Queue[np.ndarray] = queue.Queue(maxsize=max(1, int(camera.get("queue_size", 2))))
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, name="video-capture", daemon=True)

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        self.thread.join(timeout=5.0)

    def get(self, timeout: float = 1.0) -> np.ndarray:
        return self.frames.get(timeout=timeout)

    def _run(self) -> None:
        source_value: Union[int, str] = int(self.source) if self.source.isdigit() else self.source
        attempts = 0
        while not self.stop_event.is_set():
            capture = cv2.VideoCapture(source_value)
            try:
                self._configure_capture(capture)
                if not capture.isOpened():
                    raise RuntimeError(f"Could not open video source: {self.source}")
                attempts = 0
                while not self.stop_event.is_set():
                    ok, frame = capture.read()
                    if ok and frame is not None and frame.size:
                        self._put_latest(frame)
                        continue
                    if not bool(self.camera.get("reconnect_on_eof", True)):
                        self.stop_event.set()
                        break
                    raise RuntimeError("video read failed or returned an empty frame")
            except Exception as exc:
                attempts += 1
                maximum = int(self.camera.get("max_reconnect_attempts", 0))
                self.logger.error("capture_failure", extra={"event": "capture_failure", "attempt": attempts}, exc_info=exc)
                if maximum > 0 and attempts >= maximum:
                    self.logger.critical("capture_reconnect_exhausted", extra={"event": "capture_reconnect_exhausted", "attempt": attempts})
                    self.stop_event.set()
                else:
                    self.stop_event.wait(float(self.camera.get("reconnect_delay_seconds", 2.0)))
            finally:
                capture.release()

    def _put_latest(self, frame: np.ndarray) -> None:
        try:
            self.frames.put_nowait(frame)
        except queue.Full:
            try:
                self.frames.get_nowait()
            except queue.Empty:
                return
            try:
                self.frames.put_nowait(frame)
            except queue.Full:
                return

    def _configure_capture(self, capture: cv2.VideoCapture) -> None:
        for prop, key in (
            (cv2.CAP_PROP_FRAME_WIDTH, "width"),
            (cv2.CAP_PROP_FRAME_HEIGHT, "height"),
            (cv2.CAP_PROP_FPS, "fps"),
            (cv2.CAP_PROP_BUFFERSIZE, "buffer_size"),
        ):
            if key in self.camera and self.camera[key] is not None:
                capture.set(prop, float(self.camera[key]))


def load_config(path: str) -> Dict[str, Any]:
    config_path = Path(path).expanduser()
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file does not exist: {config_path}")
    with config_path.open("r", encoding="utf-8") as stream:
        loaded = yaml.safe_load(stream) or {}
    if not isinstance(loaded, dict):
        raise ValueError("Configuration root must be a YAML mapping")
    return loaded


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Threaded RDD2022 road damage detection")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--source", default=None)
    parser.add_argument("--model-type", choices=("yolov8", "yolov10", "both"), default=None)
    parser.add_argument("--yolov8-weights", default=None)
    parser.add_argument("--yolov10-weights", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--no-display", action="store_true")
    return parser.parse_args()


def draw_detections(frame: np.ndarray, detections: Sequence[Detection]) -> None:
    colors = {"D00": (255, 180, 0), "D10": (0, 220, 255), "D20": (0, 100, 255), "D40": (0, 0, 255)}
    for detection in detections:
        x1, y1, x2, y2 = detection.box
        color = colors.get(detection.class_name, (255, 255, 255))
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"{detection.class_name} {detection.confidence:.2f}", (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)


def main() -> int:
    args = parse_args()
    capture_worker: Optional[VideoCaptureWorker] = None
    writer: Optional[cv2.VideoWriter] = None
    try:
        config = load_config(args.config)
        logger = configure_logging(config)
        models = dict(config.get("models", {}))
        camera = dict(config.get("camera", {}))
        runtime = dict(config.get("runtime", {}))
        if args.source is not None:
            camera["source"] = args.source
        for argument, key in (
            (args.model_type, "model_type"),
            (args.yolov8_weights, "yolov8_weights"),
            (args.yolov10_weights, "yolov10_weights"),
            (args.device, "device"),
        ):
            if argument is not None:
                models[key] = argument
        if args.output is not None:
            runtime["output"] = args.output
        detector = RoadDamageDetector(logger=logger, **models)
        reporter = RoadDamageReporter(config.get("classes", {}), logger)
        capture_worker = VideoCaptureWorker(str(camera.get("source", "0")), camera, logger)
        capture_worker.start()
        started = time.perf_counter()
        frames = 0
        while not capture_worker.stop_event.is_set():
            try:
                frame = capture_worker.get(timeout=1.0)
            except queue.Empty:
                continue
            result = detector.detect(frame)
            frames += 1
            fps = frames / max(time.perf_counter() - started, 1e-6)
            reporter.emit(result, fps)
            draw_detections(frame, result.detections)
            cv2.putText(frame, f"FPS: {fps:.1f} Latency: {result.latency_ms:.2f} ms", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            if runtime.get("output") and writer is None:
                height, width = frame.shape[:2]
                writer = cv2.VideoWriter(str(runtime["output"]), cv2.VideoWriter_fourcc(*"mp4v"), float(camera.get("fps", 30)), (width, height))
                if not writer.isOpened():
                    raise RuntimeError(f"Could not open output video: {runtime['output']}")
            if writer is not None:
                writer.write(frame)
            if not args.no_display and runtime.get("display", True):
                cv2.imshow(str(runtime.get("window_name", "RDD2022 Road Damage Detection")), frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        return 0
    except KeyboardInterrupt:
        return 0
    except (FileNotFoundError, RuntimeError, ValueError, TypeError) as exc:
        logging.getLogger("road_ai").critical("pipeline_failed", extra={"event": "pipeline_failed"}, exc_info=exc)
        return 1
    finally:
        if capture_worker is not None:
            capture_worker.stop()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    raise SystemExit(main())