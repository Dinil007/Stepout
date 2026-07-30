"""Coordinate transformation module.

Transforms between coordinate systems:
  pixel -> camera_stabilized -> world -> pitch
"""

import logging
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class WorldPosition:
    """World position in meters."""
    x: float
    y: float


@dataclass
class PitchPosition:
    """Pitch-relative position with bounds checking."""
    world: WorldPosition
    in_bounds: bool


class CoordinateTransform:
    """Transform positions between coordinate systems."""

    def __init__(self, homography_matrix: np.ndarray, camera_motion: Optional[np.ndarray] = None) -> None:
        self.homography = homography_matrix
        self.camera_motion = camera_motion
        self.homography_inv = np.linalg.inv(homography_matrix)

    def pixel_to_camera(self, pixel: np.ndarray) -> np.ndarray:
        """Apply camera motion compensation."""
        if self.camera_motion is None:
            return pixel
        pt = np.array([[pixel]], dtype=np.float64)
        stabilized = cv2.perspectiveTransform(pt, np.linalg.inv(self.camera_motion))
        return stabilized.reshape(2)

    def camera_to_world(self, camera: np.ndarray) -> np.ndarray:
        """Transform camera-stabilized to world coordinates."""
        pt = np.array([[camera]], dtype=np.float64)
        world = cv2.perspectiveTransform(pt, self.homography_inv)
        return world.reshape(2)

    def pixel_to_world(self, pixel: np.ndarray) -> np.ndarray:
        """Transform pixel directly to world coordinates."""
        camera = self.pixel_to_camera(pixel)
        return self.camera_to_world(camera)

    def world_to_pixel(self, world: np.ndarray) -> np.ndarray:
        """Transform world to pixel coordinates."""
        pt = np.array([[world]], dtype=np.float64)
        pixel = cv2.perspectiveTransform(pt, self.homography)
        return pixel.reshape(2)

    def transform_batch(self, pixels: np.ndarray) -> np.ndarray:
        """Batch transform pixel coordinates to world."""
        camera = np.array([self.pixel_to_camera(p) for p in pixels])
        return np.array([self.camera_to_world(c) for c in camera])