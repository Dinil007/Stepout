"""Camera motion estimation module.

Estimates camera motion between consecutive frames using:
- Lucas-Kanade optical flow
- goodFeaturesToTrack
- RANSAC-based model selection: translation / affine / homography

Outputs per-frame transformation matrix, displacement, rotation, scale, confidence.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class CameraMotion:
    """Estimated camera motion for one frame."""
    frame: int
    transform: np.ndarray
    model: str
    confidence: float
    displacement: float
    rotation: float
    scale: float
    inlier_count: int
    tracked_count: int


class CameraMotionEstimator:
    """
    Estimate camera motion using feature tracking and model selection.

    Uses Lucas-Kanade optical flow with goodFeaturesToTrack.
    Automatically selects the best transformation model:
      - translation
      - affine
      - homography

    Ignores specified regions via masks.
    """

    def __init__(
        self,
        max_features: int = 800,
        lk_win_size: Tuple[int, int] = (15, 15),
        lk_max_level: int = 3,
        lk_criteria: Optional[Tuple] = None,
        min_inlier_ratio: float = 0.15,
        mask: Optional[np.ndarray] = None,
    ) -> None:
        self.max_features = max_features
        self.lk_win_size = lk_win_size
        self.lk_max_level = lk_max_level
        self.lk_criteria = lk_criteria or (
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
            30,
            0.01,
        )
        self.min_inlier_ratio = min_inlier_ratio
        self.mask = mask

        self.prev_gray: Optional[np.ndarray] = None
        self.prev_pts: Optional[np.ndarray] = None
        self.history: List[CameraMotion] = []

    def reset(self) -> None:
        """Reset tracker state."""
        self.prev_gray = None
        self.prev_pts = None
        self.history = []

    def estimate(self, frame: np.ndarray, frame_number: int) -> Optional[CameraMotion]:
        """
        Estimate camera motion for one frame.

        Args:
            frame: BGR frame.
            frame_number: Frame index.

        Returns:
            CameraMotion or None if estimation fails.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if self.prev_gray is None or self.prev_pts is None:
            self._initialize(gray)
            return None

        # Forward flow
        curr_pts, status, err = cv2.calcOpticalFlowPyrLK(
            self.prev_gray,
            gray,
            self.prev_pts,
            None,
            winSize=self.lk_win_size,
            maxLevel=self.lk_max_level,
            criteria=self.lk_criteria,
        )

        if curr_pts is None or status is None:
            self._initialize(gray)
            return None

        status = status.ravel().astype(bool)
        n_total = len(self.prev_pts)
        n_tracked = int(status.sum())
        if n_tracked < 8:
            self._initialize(gray)
            return None

        p0 = self.prev_pts[status].reshape(-1, 2)
        p1 = curr_pts[status].reshape(-1, 2)

        if self.mask is not None:
            inside = self.mask(
                p1.astype(int)
            ).ravel()
            status2 = inside > 0
            if status2.sum() < 8:
                self._initialize(gray)
                return None
            p0 = p0[status2]
            p1 = p1[status2]

        # Try models in increasing complexity; pick first with enough inliers
        models = [
            ("translation", self._estimate_translation),
            ("affine", self._estimate_affine),
            ("homography", self._estimate_homography),
        ]

        best = None  # type: Optional[Tuple[str, np.ndarray, float, int]]
        for name, fn in models:
            transform, inliers, confidence = fn(p0, p1)
            inlier_count = int(inliers.sum()) if inliers is not None else 0
            if best is None or inlier_count > best[3]:
                best = (name, transform, confidence, inlier_count)

            if best is not None and best[1] is not None:
                break

        if best is None or best[1] is None:
            self._initialize(gray)
            return None

        model_name, transform, confidence, inlier_count = best
        displacement = float(np.linalg.norm(transform[:2, 2]))
        rotation = float(np.arctan2(transform[1, 0], transform[0, 0]))
        scale = float(np.sqrt(np.linalg.det(transform[:2, :2])))

        motion = CameraMotion(
            frame=frame_number,
            transform=transform,
            model=model_name,
            confidence=confidence,
            displacement=displacement,
            rotation=rotation,
            scale=scale,
            inlier_count=inlier_count,
            tracked_count=len(p0),
        )
        self.history.append(motion)
        self.prev_gray = gray
        self.prev_pts = p1.reshape(-1, 1, 2).astype(np.float32)
        return motion

    def _initialize(self, gray: np.ndarray) -> None:
        """Initialize feature points."""
        pts = cv2.goodFeaturesToTrack(
            gray,
            self.max_features,
            qualityLevel=0.01,
            minDistance=7,
            blockSize=7,
            useHarrisDetector=False,
        )
        self.prev_gray = gray
        self.prev_pts = pts if pts is not None else np.empty((0, 1, 2), np.float32)

    def _estimate_translation(self, p0: np.ndarray, p1: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray], float]:
        dx = np.median(p1[:, 0] - p0[:, 0])
        dy = np.median(p1[:, 1] - p0[:, 1])
        transform = np.array([[1, 0, dx], [0, 1, dy], [0, 0, 1]], dtype=np.float32)
        residuals = np.linalg.norm(p1 - (p0 + np.array([dx, dy])), axis=1)
        inliers = residuals < 3.0
        confidence = float(inliers.mean()) if len(inliers) else 0.0
        return transform, inliers, confidence

    def _estimate_affine(self, p0: np.ndarray, p1: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray], float]:
        transform, inliers = cv2.estimateAffinePartial2D(
            p0, p1, method=cv2.RANSAC, ransacReprojThreshold=3.0, maxIters=2000, confidence=0.99
        )
        if transform is None:
            return np.eye(3), None, 0.0
        H = np.eye(3, dtype=np.float32)
        H[:2, :] = transform
        confidence = float(inliers.mean()) if inliers is not None else 0.0
        return H, inliers, confidence

    def _estimate_homography(self, p0: np.ndarray, p1: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray], float]:
        transform, inliers = cv2.findHomography(
            p0, p1, cv2.RANSAC, ransacReprojThreshold=3.0, maxIters=2000, confidence=0.99
        )
        if transform is None or inliers is None:
            return np.eye(3), None, 0.0
        confidence = float(inliers.mean())
        return transform, inliers, confidence