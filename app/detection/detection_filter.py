"""Geometric and confidence filtering for player and ball detections."""

from typing import List, Optional, Tuple

import cv2
import numpy as np

from app.detection.detection_types import Detection
from app.homography.homography_utils import transform_point


def parse_yolo_results(results) -> List[Detection]:
    dets: List[Detection] = []
    if not results or results[0].boxes is None:
        return dets
    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        tid = int(box.id[0]) if getattr(box, "id", None) is not None and box.id is not None else -1
        dets.append(Detection(cls_id, conf, (x1, y1, x2, y2), tid))
    return dets


class DetectionFilter:
    """Reject non-playable detections using pitch polygon, geometry, and homography."""

    def __init__(
        self,
        pitch_roi: np.ndarray,
        foot_margin: float = 28.0,
        ball_center_margin: float = 65.0,
        min_player_height: int = 16,
        min_player_conf: float = 0.0,
        min_aspect: float = 0.12,
        max_aspect: float = 0.95,
        max_ball_area: int = 2600,
        homography_matrix: Optional[np.ndarray] = None,
        pitch_length_m: float = 105.0,
        pitch_width_m: float = 68.0,
        homography_margin_m: float = 2.0,
    ) -> None:
        self.pitch_roi = pitch_roi
        self.foot_margin = foot_margin
        self.ball_center_margin = ball_center_margin
        self.min_player_height = min_player_height
        self.min_player_conf = min_player_conf
        self.min_aspect = min_aspect
        self.max_aspect = max_aspect
        self.max_ball_area = max_ball_area
        self.homography_matrix = homography_matrix
        self.pitch_length_m = pitch_length_m
        self.pitch_width_m = pitch_width_m
        self.homography_margin_m = homography_margin_m

    def set_homography(self, matrix: Optional[np.ndarray]) -> None:
        self.homography_matrix = matrix

    def inside_pitch(self, point: Tuple[int, int], margin: Optional[float] = None) -> bool:
        margin = self.foot_margin if margin is None else margin
        return cv2.pointPolygonTest(self.pitch_roi, (float(point[0]), float(point[1])), True) >= -margin

    def _inside_homography_bounds(self, foot: Tuple[int, int]) -> bool:
        if self.homography_matrix is None:
            return True
        try:
            fx, fy = transform_point(foot, self.homography_matrix)
            m = self.homography_margin_m
            return (-m <= fx <= self.pitch_length_m + m) and (-m <= fy <= self.pitch_width_m + m)
        except Exception:
            return True

    def split(self, dets: List[Detection]) -> Tuple[List[Detection], List[Detection], List[Detection]]:
        players: List[Detection] = []
        rejected: List[Detection] = []
        balls: List[Detection] = []

        for d in dets:
            if d.cls_id == 0:
                h = d.height
                w = d.width
                if d.conf < self.min_player_conf:
                    d.reject_reason = "low_confidence"
                    rejected.append(d)
                elif not self.inside_pitch(d.foot):
                    d.reject_reason = "foot_outside_pitch"
                    rejected.append(d)
                elif h < self.min_player_height:
                    d.reject_reason = "bbox_too_small"
                    rejected.append(d)
                elif not (self.min_aspect <= w / max(h, 1) <= self.max_aspect):
                    d.reject_reason = "bad_aspect_ratio"
                    rejected.append(d)
                elif not self._inside_homography_bounds(d.foot):
                    d.reject_reason = "homography_out_of_bounds"
                    rejected.append(d)
                else:
                    players.append(d)
            elif d.cls_id == 32:
                if not self.inside_pitch(d.center, margin=self.ball_center_margin):
                    d.reject_reason = "ball_outside_pitch"
                    rejected.append(d)
                elif d.area > self.max_ball_area:
                    d.reject_reason = "ball_too_large"
                    rejected.append(d)
                else:
                    balls.append(d)
        return players, rejected, balls
