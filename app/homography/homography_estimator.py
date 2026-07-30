"""Homography estimator with strategy pattern for calibration modes.

Supports:
  - Manual calibration
  - Automatic calibration (placeholder)

Public API:
  estimate_camera_motion()
  estimate_homography()
  transform_tracks()
  validate()
  visualize()
"""

import logging
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from app.homography.camera_motion import CameraMotionEstimator, CameraMotion
from app.homography.manual_calibration import ManualCalibration, ManualCalibrationResult
from app.homography.auto_calibration import AutoCalibration, AutoCalibrationResult
from app.homography.pitch_model import PitchModel
from app.homography.homography_metrics import HomographyMetrics

logger = logging.getLogger(__name__)


class HomographyEstimator:
    """
    Hybrid homography estimation engine.

    Combines:
      - per-frame camera motion from feature tracking
      - global pitch homography from calibration strategy

    The calibration strategy is pluggable:
      ManualCalibration, AutoCalibration, or future implementations.
    """

    def __init__(self, config: Dict, strategy: Optional[AutoCalibration] = None) -> None:
        self.config = config
        self.pitch_model = PitchModel.from_config(config.get("pitch", {}))
        self.calibration_strategy = strategy or ManualCalibration(config, self.pitch_model)
        self.camera_motion_estimator = CameraMotionEstimator()
        self.metrics = HomographyMetrics(config)
        self.camera_motion_history: List[CameraMotion] = []
        self.calibration_result: Optional[ManualCalibrationResult] = None
        self._camera_motion: Optional[CameraMotion] = None

    def initialize(self, first_frame: Optional[np.ndarray] = None) -> bool:
        """Initialize with first frame and calibration."""
        if first_frame is not None:
            gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
            self.camera_motion_estimator._initialize(gray)

        self.calibration_result = self.calibration_strategy.estimate()
        if not self.calibration_result.success:
            logger.warning("Initial calibration failed: %s", self.calibration_result.message)
            return False
        logger.info("Calibration initialized: %s", self.calibration_result.message)
        return True

    def estimate_camera_motion(self, frame: np.ndarray, frame_number: int) -> Optional[CameraMotion]:
        """Estimate per-frame camera motion."""
        motion = self.camera_motion_estimator.estimate(frame, frame_number)
        if motion is not None:
            self._camera_motion = motion
            self.camera_motion_history.append(motion)
        return motion

    def estimate_homography(self, image_points: Optional[np.ndarray] = None) -> ManualCalibrationResult:
        """Estimate global homography using current calibration strategy."""
        if isinstance(self.calibration_strategy, ManualCalibration):
            result = self.calibration_strategy.estimate(image_points)
        else:
            frame = image_points  # For auto strategies
            result = self.calibration_strategy.estimate(frame)
        self.calibration_result = result
        return result

    def get_world_position(self, pixel_point: np.ndarray) -> Optional[np.ndarray]:
        """Transform pixel to world coordinates using current homography."""
        if self.calibration_result is None or not self.calibration_result.success:
            return None
        H = self.calibration_result.homography_matrix
        pt = np.array([[pixel_point]], dtype=np.float64)
        world = cv2.perspectiveTransform(pt, np.linalg.inv(H))
        return world.reshape(2)

    def get_camera_stabilized_position(self, pixel_point: np.ndarray) -> Optional[np.ndarray]:
        """Apply camera motion compensation to pixel point."""
        if self._camera_motion is None:
            return pixel_point
        H = np.linalg.inv(self._camera_motion.transform)
        pt = np.array([[pixel_point]], dtype=np.float64)
        stabilized = cv2.perspectiveTransform(pt, H)
        return stabilized.reshape(2)

    def validate(self) -> Dict:
        """Validate current homography quality."""
        if self.calibration_result is None:
            return {"valid": False, "message": "No calibration"}
        return self.metrics.validate_calibration(self.calibration_result)

    def transform_tracks(self, tracks: List[Dict]) -> List[Dict]:
        """Transform track positions through all coordinate systems."""
        if self.calibration_result is None:
            return tracks
        transformed = []
        for track in tracks:
            pixel = np.array([track["x"], track["y"]], dtype=np.float64)
            camera = self.get_camera_stabilized_position(pixel)
            world = self.get_world_position(camera if camera is not None else pixel)
            track["camera_x"] = float(camera[0]) if camera is not None else None
            track["camera_y"] = float(camera[1]) if camera is not None else None
            track["world_x"] = float(world[0]) if world is not None else None
            track["world_y"] = float(world[1]) if world is not None else None
            transformed.append(track)
        return transformed

    def visualize(self, frame: np.ndarray) -> np.ndarray:
        """Visualize homography overlay on frame."""
        vis = frame.copy()
        if self.calibration_result is None:
            return vis
        pitch = self.pitch_model.get_all_elements()
        # Draw pitch outline
        corners = pitch["corners"]
        for i in range(4):
            p1 = tuple(corners[i].astype(int))
            p2 = tuple(corners[(i + 1) % 4].astype(int))
            cv2.line(vis, p1, p2, (0, 255, 255), 2)
        return vis