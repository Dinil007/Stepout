"""YOLO detector wrapper — single source of truth for inference parameters."""

import time
from typing import Dict, List, Optional

import torch
from ultralytics import YOLO

from app.core.config import get_config
from app.detection.detection_filter import parse_yolo_results
from app.detection.detection_types import Detection


class YoloDetector:
    """Load YOLO once and run predict/track with config-driven parameters."""

    def __init__(
        self,
        config: Optional[Dict] = None,
        model_path: Optional[str] = None,
        device: Optional[str] = None,
        conf: Optional[float] = None,
        iou: Optional[float] = None,
        imgsz: Optional[int] = None,
        classes: Optional[List[int]] = None,
        tracker_config: Optional[str] = None,
    ) -> None:
        cfg = config or get_config().raw
        models_cfg = cfg.get("models", {})
        video_cfg = cfg.get("video", {})
        tracking_cfg = cfg.get("tracking", {})
        
        self.model_path = model_path or models_cfg.get("yolo_model_path", "yolov8x.pt")
        requested_device = device or cfg.get("device", "cuda:0")
        self.device = "cpu" if not torch.cuda.is_available() else requested_device
        self.conf = conf if conf is not None else float(models_cfg.get("confidence_threshold", 0.25))
        self.iou = iou if iou is not None else float(models_cfg.get("iou_threshold", 0.5))
        self.imgsz = imgsz if imgsz is not None else int(models_cfg.get("image_size", 640))
        self.classes = classes or models_cfg.get("classes", [0, 32])
        self.tracker_config = tracker_config or tracking_cfg.get("tracker_config", "app/tracking/bytetrack_custom.yaml")
        self.model: Optional[YOLO] = None
        self.last_inference_ms: float = 0.0

    def load(self) -> None:
        self.model = YOLO(self.model_path)
        self.model.to(self.device)
        try:
            self.model.fuse()
        except Exception:
            pass
        if torch.cuda.is_available() and "cuda" in self.device:
            self.model.model.half()

    def track(self, frame, persist: bool = True) -> List[Detection]:
        if self.model is None:
            raise RuntimeError("YoloDetector.load() must be called before track()")
        t0 = time.perf_counter()
        results = self.model.track(
            source=frame,
            persist=persist,
            tracker=self.tracker_config,
            classes=self.classes,
            conf=self.conf,
            iou=self.iou,
            imgsz=self.imgsz,
            verbose=False,
            device=self.device,
        )
        self.last_inference_ms = (time.perf_counter() - t0) * 1000.0
        return parse_yolo_results(results)

    def predict(self, frame) -> List[Detection]:
        if self.model is None:
            raise RuntimeError("YoloDetector.load() must be called before predict()")
        t0 = time.perf_counter()
        results = self.model.predict(
            source=frame,
            classes=self.classes,
            conf=self.conf,
            iou=self.iou,
            imgsz=self.imgsz,
            verbose=False,
            device=self.device,
        )
        self.last_inference_ms = (time.perf_counter() - t0) * 1000.0
        return parse_yolo_results(results)
