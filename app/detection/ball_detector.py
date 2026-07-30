"""
Ball Detection Module - Complete Implementation

Dedicated YOLO-based ball detector with:
1. YOLO Detection → Get ball bounding boxes (class 32, high-res inference)
2. Best Detection Selection → Score by confidence + distance to prediction
3. Ball-specific filtering (size, area, pitch position)
4. Integration with BallTracker and BallInterpolator

Flow:
  YOLO Inference → Parse Results → Filter → Score → Best Selection → BallTracker
"""

import logging
import time
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from app.core.config import get_config
from app.detection.detection_types import Detection

logger = logging.getLogger(__name__)


class BallDetector:
    """
    Dedicated ball detector using YOLO with ball-optimized parameters.
    
    Uses higher resolution (960px) and lower confidence threshold (0.10)
    specifically for detecting small footballs in broadcast video.
    """

    def __init__(
        self,
        config: Optional[Dict] = None,
        model_path: Optional[str] = None,
        device: Optional[str] = None,
        conf: Optional[float] = None,
        iou: Optional[float] = None,
        imgsz: Optional[int] = None,
    ) -> None:
        """
        Args:
            config: Optional config dict (overrides config.yaml)
            model_path: Path to YOLO weights file
            device: Target device ('cuda:0', 'cpu', etc.)
            conf: Detection confidence threshold for ball
            iou: NMS IoU threshold
            imgsz: Inference image size (higher = better for small objects)
        """
        cfg = config or get_config().raw
        models_cfg = cfg.get("models", {})
        tracking_cfg = cfg.get("tracking", {})

        # Ball-optimized parameters
        self.model_path = model_path or models_cfg.get("yolo_model_path", "yolov8x.pt")
        requested_device = device or cfg.get("device", "cuda:0")
        self.device = "cpu" if not torch.cuda.is_available() else requested_device
        self.conf = conf if conf is not None else float(
            models_cfg.get("ball_confidence_threshold", 0.10)
        )
        self.iou = iou if iou is not None else float(
            models_cfg.get("iou_threshold", 0.5)
        )
        self.imgsz = imgsz if imgsz is not None else int(
            models_cfg.get("ball_image_size", 960)
        )
        # Ball is class 32 in COCO
        self.classes = [32]

        # Association parameters
        self.max_match_dist = float(
            tracking_cfg.get("ball_max_match_dist", 180.0)
        )
        self.dist_penalty_scale = float(
            tracking_cfg.get("ball_max_match_dist", 180.0)
        )

        # Ball filtering parameters
        self.max_ball_area = int(
            cfg.get("detection_filter", {}).get("max_ball_area", 2600)
        )
        self.ball_center_margin = float(
            cfg.get("detection_filter", {}).get("ball_center_margin_px", 65.0)
        )

        self.model: Optional[YOLO] = None
        self.last_inference_ms: float = 0.0
        self._pitch_roi: Optional[np.ndarray] = None

        # Detection statistics
        self.total_detections: int = 0
        self.filtered_detections: int = 0
        self.accepted_detections: int = 0

    def load(self) -> None:
        """Load YOLO model and move to target device with optimizations."""
        logger.info(
            f"Loading ball detector: model={self.model_path}, "
            f"device={self.device}, imgsz={self.imgsz}, conf={self.conf}"
        )
        self.model = YOLO(self.model_path)
        self.model.to(self.device)
        try:
            self.model.fuse()
        except Exception:
            pass
        if torch.cuda.is_available() and "cuda" in self.device:
            self.model.model.half()
        logger.info("Ball detector loaded successfully")

    def set_pitch_roi(self, roi: np.ndarray) -> None:
        """Set pitch ROI polygon for filtering detections outside the field."""
        self._pitch_roi = roi

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """
        Run YOLO inference for ball detection only.
        
        Args:
            frame: Input video frame (BGR format)
            
        Returns:
            List of Detection objects for ball candidates
        """
        if self.model is None:
            raise RuntimeError("BallDetector.load() must be called before detect()")

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

        # Parse YOLO results into Detection objects
        detections: List[Detection] = []
        if results and results[0].boxes is not None:
            for box in results[0].boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                detections.append(Detection(cls_id, conf, (x1, y1, x2, y2)))

        self.total_detections += len(detections)
        return detections

    def filter_detections(
        self,
        detections: List[Detection],
        frame_shape: Optional[Tuple[int, int]] = None,
    ) -> List[Detection]:
        """
        Filter ball detections by geometry and pitch position.
        
        Filters applied:
        1. Area check: Reject if bbox area > max_ball_area
        2. Pitch ROI check: Reject if center is outside pitch (with margin)
        3. Frame boundary check: Reject if bbox extends beyond frame edges
        
        Args:
            detections: List of ball Detection objects
            frame_shape: (height, width) of the frame for boundary checks
            
        Returns:
            Filtered list of valid ball detections
        """
        valid: List[Detection] = []
        for det in detections:
            cx, cy = det.center

            # 1. Area check
            if det.area > self.max_ball_area:
                det.reject_reason = "ball_too_large"
                self.filtered_detections += 1
                continue

            # 2. Frame boundary check
            if frame_shape is not None:
                h, w = frame_shape[:2]
                margin = 10  # pixels
                if (cx < -margin or cx > w + margin or
                    cy < -margin or cy > h + margin):
                    det.reject_reason = "out_of_frame"
                    self.filtered_detections += 1
                    continue

            # 3. Pitch ROI check
            if self._pitch_roi is not None:
                inside = cv2.pointPolygonTest(
                    self._pitch_roi,
                    (float(cx), float(cy)),
                    False,
                )
                if inside < 0:
                    det.reject_reason = "ball_outside_pitch"
                    self.filtered_detections += 1
                    continue

            valid.append(det)

        self.accepted_detections += len(valid)
        return valid

    def score_detection(
        self,
        det: Detection,
        predicted_center: Optional[Tuple[float, float]] = None,
    ) -> float:
        """
        Score a ball detection combining confidence and distance to prediction.
        
        Scoring formula:
            score = confidence - min(distance / dist_penalty_scale, 2.0)
        
        This penalizes detections far from the predicted position while
        rewarding high-confidence detections near the prediction.
        
        Args:
            det: Ball Detection object
            predicted_center: (x, y) predicted position from Kalman filter
            
        Returns:
            Combined score (higher is better)
        """
        conf = float(det.conf)
        if predicted_center is None:
            return conf

        cx, cy = det.center
        dist = float(np.hypot(cx - predicted_center[0], cy - predicted_center[1]))
        # Penalize distance from prediction (capped at 2.0)
        return conf - min(dist / self.dist_penalty_scale, 2.0)

    def pick_best_detection(
        self,
        detections: List[Detection],
        predicted_center: Optional[Tuple[float, float]] = None,
    ) -> Optional[Detection]:
        """
        Select the best ball detection from multiple candidates.
        
        Uses score_detection() to rank candidates by confidence + proximity
        to the predicted position.
        
        Args:
            detections: List of ball Detection objects
            predicted_center: (x, y) predicted position from Kalman filter
            
        Returns:
            Best Detection or None if no detections
        """
        if not detections:
            return None

        return max(
            detections,
            key=lambda d: self.score_detection(d, predicted_center),
        )

    def detect_and_filter(
        self,
        frame: np.ndarray,
        predicted_center: Optional[Tuple[float, float]] = None,
    ) -> Tuple[Optional[Detection], List[Detection], float]:
        """
        Complete ball detection pipeline: infer → filter → score → select best.
        
        This is the main entry point for ball detection in a frame.
        
        Args:
            frame: Input video frame (BGR format)
            predicted_center: (x, y) predicted position from Kalman filter
                             (used for scoring proximity)
                             
        Returns:
            Tuple of:
            - Best Detection (or None if no valid detection)
            - All filtered detections (for debugging/visualization)
            - Inference time in milliseconds
        """
        # Step 1: YOLO inference
        raw_dets = self.detect(frame)

        # Step 2: Filter by geometry
        frame_shape = frame.shape[:2]  # (H, W)
        filtered_dets = self.filter_detections(raw_dets, frame_shape)

        # Step 3: Score and pick best
        best_det = self.pick_best_detection(filtered_dets, predicted_center)

        return best_det, filtered_dets, self.last_inference_ms

    def get_metrics(self) -> Dict:
        """Get ball detection performance metrics."""
        return {
            "total_raw_detections": self.total_detections,
            "filtered_detections": self.filtered_detections,
            "accepted_detections": self.accepted_detections,
            "last_inference_ms": round(self.last_inference_ms, 2),
            "confidence_threshold": self.conf,
            "image_size": self.imgsz,
            "device": self.device,
        }

    def detection_to_dict(self, det: Detection) -> Dict:
        """Convert a Detection to the dict format expected by BallTracker."""
        cx, cy = det.center
        return {
            "center": (float(cx), float(cy)),
            "bbox": list(det.bbox),
            "confidence": float(det.conf),
        }

    def detections_to_dict_list(self, detections: List[Detection]) -> List[Dict]:
        """Convert a list of Detections to the dict format for BallTracker."""
        return [self.detection_to_dict(d) for d in detections]