"""Manual calibration strategy.

Uses manually selected pitch points from config or interactive selection.
"""

import json
import logging
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from app.homography.pitch_model import PitchModel, PitchDimensions

logger = logging.getLogger(__name__)


@dataclass
class ManualCalibrationResult:
    """Output of manual calibration."""
    homography_matrix: np.ndarray
    confidence: float
    reprojection_error: float
    determinant: float
    message: str
    success: bool
    pixels_per_metre_x: float = 0.0
    pixels_per_metre_y: float = 0.0
    image_points: Optional[np.ndarray] = None
    world_points: Optional[np.ndarray] = None


class ManualCalibration:
    """
    Manual calibration strategy.

    Loads calibration points from config file or accepts them programmatically.
    Computes homography from image points to world points.
    """

    def __init__(self, config: Dict, pitch_model: Optional[PitchModel] = None) -> None:
        """
        Args:
            config: Config dict loaded from config file.
            pitch_model: Optional PitchModel instance.
        """
        self.config = config
        self.pitch_model = pitch_model

    def estimate(self, image_points: Optional[np.ndarray] = None) -> ManualCalibrationResult:
        """
        Estimate homography from manual calibration points.

        Args:
            image_points: Optional override for image points.
                          If None, load from config file.

        Returns:
            ManualCalibrationResult with homography matrix.
        """
        # Load points from config if not provided
        if image_points is None:
            calib_cfg = self.config.get("calibration_points", {})
            source = np.array(calib_cfg.get("source", []), dtype=np.float64)
            destination = np.array(calib_cfg.get("destination", []), dtype=np.float64)
        else:
            source = image_points
            calib_cfg = self.config.get("calibration_points", {})
            destination = np.array(calib_cfg.get("destination", []), dtype=np.float64)

        if source.shape[0] < 4 or destination.shape[0] < 4:
            return ManualCalibrationResult(
                homography_matrix=np.eye(3),
                confidence=0.0,
                reprojection_error=float('inf'),
                determinant=1.0,
                message=f"Insufficient calibration points: {source.shape[0]} source, {destination.shape[0]} dest",
                success=False,
            )

        if source.shape[0] != destination.shape[0]:
            return ManualCalibrationResult(
                homography_matrix=np.eye(3),
                confidence=0.0,
                reprojection_error=float('inf'),
                determinant=1.0,
                message=f"Point count mismatch: {source.shape[0]} vs {destination.shape[0]}",
                success=False,
            )

        homography, mask = cv2.findHomography(
            source, destination, cv2.RANSAC, ransacReprojThreshold=3.0, maxIters=2000, confidence=0.99
        )

        if homography is None:
            return ManualCalibrationResult(
                homography_matrix=np.eye(3),
                confidence=0.0,
                reprojection_error=float('inf'),
                determinant=1.0,
                message="RANSAC failed to compute homography",
                success=False,
                image_points=source,
                world_points=destination,
            )

        # Compute reprojection error
        inliers = mask.ravel() == 1
        if inliers.sum() > 0:
            reprojected = cv2.perspectiveTransform(source[inliers].reshape(-1, 1, 2), homography)
            reprojection_errors = np.linalg.norm(
                destination[inliers] - reprojected.reshape(-1, 2), axis=1
            )
            reprojection_error = float(np.mean(reprojection_errors))
        else:
            reprojection_error = float('inf')

        det = np.linalg.det(homography[:2, :2])
        mppx_x = np.linalg.norm(homography[:2, 0])
        mppx_y = np.linalg.norm(homography[:2, 1])

        inlier_ratio = inliers.sum() / max(len(source), 1)
        error_factor = max(0.0, 1.0 - reprojection_error / 5.0)
        confidence = float(inlier_ratio * 0.6 + error_factor * 0.4)

        result = ManualCalibrationResult(
            homography_matrix=homography,
            confidence=confidence,
            reprojection_error=reprojection_error,
            determinant=float(det),
            message=f"Manual calibration successful: {inliers.sum()}/{len(source)} inliers",
            success=True,
            pixels_per_metre_x=float(mppx_x),
            pixels_per_metre_y=float(mppx_y),
            image_points=source,
            world_points=destination,
        )
        return result