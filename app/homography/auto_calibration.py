"""Automatic calibration strategy — placeholder for future implementation.

DO NOT implement yet. This defines the interface so future calibration
modules can plug into HomographyEstimator without API changes.
"""

import logging
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class AutoCalibrationResult:
    """Output placeholder for automatic calibration."""
    def __init__(self) -> None:
        self.homography_matrix = np.eye(3, dtype=np.float64)
        self.confidence = 0.0
        self.reprojection_error = float('inf')
        self.determinant = 1.0
        self.message = "Not implemented"
        self.success = False
        self.pixels_per_metre_x = 0.0
        self.pixels_per_metre_y = 0.0
        self.method = "auto"


class AutoCalibration(ABC):
    """
    Abstract base for automatic calibration strategies.

    Future implementations:
      - PitchLineDetectorCalibration: detect pitch lines with Hough/CNN
      - DeepLearningCalibration: end-to-end homography regression
      - MultiCameraCalibration: multi-view bundle adjustment
      - PanoramicStitchingCalibration: wide-angle pitch stitching
      - BroadcastCalibration: detect broadcast graphics overlay
    """

    @abstractmethod
    def estimate(self, frame: np.ndarray) -> AutoCalibrationResult:
        """
        Estimate homography from a single frame or frame sequence.

        Args:
            frame: Input frame(s).

        Returns:
            AutoCalibrationResult with homography matrix.
        """
        raise NotImplementedError


class PlaceholderAutoCalibration(AutoCalibration):
    """Placeholder auto calibration — returns identity matrix."""

    def estimate(self, frame: np.ndarray) -> AutoCalibrationResult:
        return AutoCalibrationResult()