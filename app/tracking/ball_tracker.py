"""
Ball Tracker Module

Dedicated single-object tracker for the football (ball).
Uses Kalman Filter prediction, motion-consistency gating, and
distance-weighted detection association.
"""

import logging
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from app.core.config import get_config

logger = logging.getLogger(__name__)

_F = np.array([
    [1, 0, 1, 0],
    [0, 1, 0, 1],
    [0, 0, 1, 0],
    [0, 0, 0, 1],
], dtype=np.float32)

_H = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
], dtype=np.float32)

_Q = np.eye(4, dtype=np.float32) * 0.1
_R = np.eye(2, dtype=np.float32) * 4.0
_P0 = np.eye(4, dtype=np.float32) * 100.0


class BallKalmanFilter:
    def __init__(self, cx: float, cy: float) -> None:
        self.x = np.array([[cx], [cy], [0.0], [0.0]], dtype=np.float32)
        self.P = _P0.copy()

    def predict(self) -> Tuple[float, float]:
        self.x = _F @ self.x
        self.P = _F @ self.P @ _F.T + _Q
        return float(self.x[0, 0]), float(self.x[1, 0])

    def update(self, cx: float, cy: float) -> Tuple[float, float]:
        z = np.array([[cx], [cy]], dtype=np.float32)
        S = _H @ self.P @ _H.T + _R
        K = self.P @ _H.T @ np.linalg.inv(S)
        self.x = self.x + K @ (z - _H @ self.x)
        self.P = (np.eye(4, dtype=np.float32) - K @ _H) @ self.P
        return float(self.x[0, 0]), float(self.x[1, 0])


class BallTrack:
    BALL_TRACK_ID: int = 1

    def __init__(self, frame: int, cx: float, cy: float, bbox: List[int], confidence: float) -> None:
        self.track_id: int = self.BALL_TRACK_ID
        self.kf: BallKalmanFilter = BallKalmanFilter(cx, cy)
        self.predicted_center: Tuple[float, float] = (cx, cy)
        self.confirmed_center: Tuple[float, float] = (cx, cy)
        self.bbox: List[int] = bbox
        self.confidence: float = confidence
        self.missing_frames: int = 0
        self.is_lost: bool = False
        self.image_history: deque = deque(maxlen=45)
        self.image_history.append((cx, cy))
        self.longest_streak: int = 1
        self.current_streak: int = 1

    def predict(self) -> Tuple[float, float]:
        self.predicted_center = self.kf.predict()
        return self.predicted_center

    def update(self, cx: float, cy: float, bbox: List[int], confidence: float, frame: int) -> None:
        self.confirmed_center = self.kf.update(cx, cy)
        self.bbox = bbox
        self.confidence = confidence
        self.missing_frames = 0
        self.is_lost = False
        self.current_streak += 1
        self.longest_streak = max(self.longest_streak, self.current_streak)
        self.image_history.append(self.confirmed_center)


class BallTracker:
    """
    YOLO Ball Detection -> Kalman Prediction -> Motion Consistency -> Association -> Continuation
    """

    def __init__(self, config: Optional[Dict] = None) -> None:
        cfg = config or get_config().raw
        ball_cfg = cfg.get("ball_tracking", {})
        
        models_cfg = cfg.get("models", {})
        self.max_missing_frames = int(ball_cfg.get("max_missing_frames", 45))
        self.max_match_dist = float(ball_cfg.get("association_distance", 180.0))
        self.trajectory_len = int(ball_cfg.get("trajectory_len", 45))
        self.dist_penalty_scale = float(ball_cfg.get("prediction_distance", 180.0))
        self.kalman_enabled = bool(ball_cfg.get("kalman_enabled", True))
        self.ball_imgsz = int(models_cfg.get("ball_image_size", 960))
        self.ball_conf = float(models_cfg.get("ball_confidence_threshold", 0.10))
        
        self._track: Optional[BallTrack] = None
        self.raw_detections: int = 0
        self.accepted_detections: int = 0
        self.predicted_frames: int = 0
        self.missing_frames: int = 0
        self.coverage_frames: int = 0
        self.total_frames: int = 0

    def update(self, detections: List[Dict], frame_number: int) -> Optional[Dict]:
        self.total_frames += 1
        best_det = self._pick_best_detection(detections)

        if self._track is None or self._track.is_lost:
            self.raw_detections += len(detections)
            if best_det is not None:
                cx, cy = best_det["center"]
                self._track = BallTrack(
                    frame=frame_number,
                    cx=float(cx),
                    cy=float(cy),
                    bbox=best_det["bbox"],
                    confidence=best_det["confidence"],
                )
                self.accepted_detections += 1
                self.coverage_frames += 1
            else:
                self.missing_frames += 1
                return None
        else:
            pred_cx, pred_cy = self._track.predict() if self.kalman_enabled else self._track.predicted_center

            if best_det is not None:
                det_cx, det_cy = best_det["center"]
                dist = float(np.hypot(det_cx - pred_cx, det_cy - pred_cy))
                if dist <= self.max_match_dist:
                    self._track.update(
                        float(det_cx),
                        float(det_cy),
                        best_det["bbox"],
                        best_det["confidence"],
                        frame_number,
                    )
                    self.raw_detections += len(detections)
                    self.accepted_detections += 1
                    self.coverage_frames += 1
                else:
                    self.raw_detections += len(detections)
                    self._handle_missing()
            else:
                self._handle_missing()

        if self._track is None or self._track.is_lost:
            return None

        is_predicted = self._track.missing_frames > 0
        if is_predicted:
            self.predicted_frames += 1
        center = self._track.predicted_center if is_predicted else self._track.confirmed_center
        return {
            "frame": frame_number,
            "track_id": self._track.track_id,
            "center": (round(float(center[0]), 1), round(float(center[1]), 1)),
            "bbox": self._track.bbox,
            "confidence": round(self._track.confidence, 4) if not is_predicted else 0.0,
            "is_predicted": is_predicted,
            "image_history": list(self._track.image_history),
            "longest_streak": self._track.longest_streak,
        }

    def get_trajectory(self) -> List[Tuple[float, float]]:
        if self._track is None:
            return []
        return list(self._track.image_history)

    def is_active(self) -> bool:
        return self._track is not None and not self._track.is_lost

    def longest_streak(self) -> int:
        return self._track.longest_streak if self._track is not None else 0

    def get_metrics(self) -> Dict:
        """Return ball tracking coverage metrics."""
        coverage_ratio = self.coverage_frames / max(self.total_frames, 1)
        return {
            "raw_detections": self.raw_detections,
            "accepted_detections": self.accepted_detections,
            "predicted_frames": self.predicted_frames,
            "missing_frames": self.missing_frames,
            "coverage_ratio": round(coverage_ratio, 3),
            "longest_continuous_track": self.longest_streak(),
            "average_confidence": round(self._track.confidence, 3) if self._track else 0.0,
        }

    def _score_detection(self, det: Dict, pred: Optional[Tuple[float, float]]) -> float:
        conf = float(det["confidence"])
        if pred is None:
            return conf
        cx, cy = det["center"]
        dist = float(np.hypot(cx - pred[0], cy - pred[1]))
        return conf - min(dist / self.dist_penalty_scale, 2.0)

    def _pick_best_detection(self, detections: List[Dict]) -> Optional[Dict]:
        if not detections:
            return None
        pred = None
        if self._track is not None and not self._track.is_lost:
            pred = self._track.predicted_center
        return max(detections, key=lambda d: self._score_detection(d, pred))

    def _handle_missing(self) -> None:
        if self._track is None:
            return
        self._track.missing_frames += 1
        if self._track.missing_frames > self.max_missing_frames:
            self._track.is_lost = True
            self._track.current_streak = 0
        else:
            self.missing_frames += 1