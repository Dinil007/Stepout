"""Homography validation and metrics module."""

import logging
from dataclasses import dataclass, field
from typing import Dict, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class HomographyValidationResult:
    """Validation result for a homography matrix."""
    valid: bool
    message: str
    determinant: float
    condition_number: float
    reprojection_error: float
    out_of_bounds_count: int
    confidence: float


class HomographyMetrics:
    """Validate and score homography quality."""

    def __init__(self, config: Dict) -> None:
        self.config = config.get("homography", {})
        validation_cfg = self.config.get("validation", {})
        self.max_reprojection_error = float(validation_cfg.get("max_reprojection_error", 3.0))
        self.min_determinant = float(validation_cfg.get("min_determinant", 0.01))
        self.max_determinant = float(validation_cfg.get("max_determinant", 10.0))
        self.min_confidence = float(validation_cfg.get("min_confidence", 0.7))
        self.pitch_length = float(self.config.get("pitch_length_m", 105.0))
        self.pitch_width = float(self.config.get("pitch_width_m", 68.0))

    def validate_calibration(self, calibration_result) -> Dict:
        """Validate calibration result and return summary."""
        if calibration_result is None or not calibration_result.success:
            return {"valid": False, "message": "No calibration"}

        det = np.linalg.det(calibration_result.homography_matrix[:2, :2])
        cond = np.linalg.cond(calibration_result.homography_matrix[:2, :2])

        valid = True
        messages = []

        if calibration_result.reprojection_error > self.max_reprojection_error:
            valid = False
            messages.append(f"Reprojection error too high: {calibration_result.reprojection_error:.2f}")

        if not (self.min_determinant < det < self.max_determinant):
            valid = False
            messages.append(f"Determinant out of range: {det:.3f}")

        if calibration_result.confidence < self.min_confidence:
            valid = False
            messages.append(f"Confidence too low: {calibration_result.confidence:.2f}")

        message = "; ".join(messages) if messages else "Validation passed"

        return {
            "valid": valid,
            "message": message,
            "determinant": float(det),
            "condition_number": float(cond),
            "reprojection_error": float(calibration_result.reprojection_error),
            "confidence": float(calibration_result.confidence),
        }

    def compute_world_error(self, pixel_points: np.ndarray, world_points: np.ndarray, homography: np.ndarray) -> float:
        """Compute mean reprojection error in world coordinates."""
        if pixel_points.shape[0] == 0:
            return 0.0

        H_inv = np.linalg.inv(homography)
        predicted = cv2.perspectiveTransform(pixel_points.reshape(-1, 1, 2), H_inv).reshape(-1, 2)
        errors = np.linalg.norm(world_points - predicted, axis=1)
        return float(np.mean(errors))

    def check_out_of_bounds(self, world_points: np.ndarray, margin: float = 0.0) -> int:
        """Count points outside pitch bounds."""
        x = world_points[:, 0]
        y = world_points[:, 1]
        in_bounds = (
            (-margin <= x) &
            (x <= self.pitch_length + margin) &
            (-margin <= y) &
            (y <= self.pitch_width + margin)
        )
        return int((~in_bounds).sum())