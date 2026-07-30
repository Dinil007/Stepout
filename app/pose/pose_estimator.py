"""
Pose Estimator Module

Wraps MediaPipe Pose (Tasks API 0.10+) to extract 33 3D body landmarks from player crops.
Supports single-frame and batch processing with normalized and pixel coordinates.
"""

import os
import sys
import urllib.request
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import cv2
import numpy as np

try:
    import mediapipe as mp
except ImportError:
    raise ImportError("MediaPipe is required: pip install mediapipe")

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)

# Default model asset path
DEFAULT_MODEL_PATH = "models/pose_landmarker_full.task"
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task"


# ==========================================
# MediaPipe Landmark Index Reference
# ==========================================
class LandmarkIndex:
    """MediaPipe Pose 33-landmark index constants."""
    NOSE                = 0
    LEFT_EYE_INNER     = 1
    LEFT_EYE           = 2
    LEFT_EYE_OUTER     = 3
    RIGHT_EYE_INNER    = 4
    RIGHT_EYE          = 5
    RIGHT_EYE_OUTER    = 6
    LEFT_EAR           = 7
    RIGHT_EAR          = 8
    MOUTH_LEFT         = 9
    MOUTH_RIGHT        = 10
    LEFT_SHOULDER      = 11
    RIGHT_SHOULDER     = 12
    LEFT_ELBOW         = 13
    RIGHT_ELBOW        = 14
    LEFT_WRIST         = 15
    RIGHT_WRIST        = 16
    LEFT_PINKY         = 17
    RIGHT_PINKY        = 18
    LEFT_INDEX         = 19
    RIGHT_INDEX        = 20
    LEFT_THUMB         = 21
    RIGHT_THUMB        = 22
    LEFT_HIP           = 23
    RIGHT_HIP          = 24
    LEFT_KNEE          = 25
    RIGHT_KNEE         = 26
    LEFT_ANKLE         = 27
    RIGHT_ANKLE        = 28
    LEFT_HEEL          = 29
    RIGHT_HEEL         = 30
    LEFT_FOOT_INDEX    = 31
    RIGHT_FOOT_INDEX   = 32


# Standard 33-Landmark Pose Skeleton Connections
POSE_CONNECTIONS: List[Tuple[int, int]] = [
    # Torso
    (11, 12), (11, 23), (12, 24), (23, 24),
    # Left Arm
    (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
    # Right Arm
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    # Left Leg
    (23, 25), (25, 27), (27, 29), (27, 31), (29, 31),
    # Right Leg
    (24, 26), (26, 28), (28, 30), (28, 32), (30, 32),
]


@dataclass
class Landmark:
    """Represents a single MediaPipe pose landmark."""
    index: int
    x_norm: float          # Normalized [0, 1] horizontal
    y_norm: float          # Normalized [0, 1] vertical
    z_norm: float          # Depth (relative to hip)
    visibility: float      # Confidence score [0, 1]
    x_px: float = 0.0     # Pixel coordinate
    y_px: float = 0.0


@dataclass
class PoseResult:
    """Full pose estimation result for a single player crop."""
    track_id: int
    success: bool
    confidence: float
    landmarks: List[Landmark] = field(default_factory=list)

    def get_landmark(self, index: int) -> Optional[Landmark]:
        """Returns a specific landmark by MediaPipe index."""
        if index < len(self.landmarks):
            return self.landmarks[index]
        return None

    def get_point_px(self, index: int) -> Optional[Tuple[float, float]]:
        """Returns (x_px, y_px) for a landmark if visible."""
        lm = self.get_landmark(index)
        if lm and lm.visibility > 0.4:
            return lm.x_px, lm.y_px
        return None

    def get_point_norm(self, index: int) -> Optional[Tuple[float, float, float]]:
        """Returns (x_norm, y_norm, z_norm) for a landmark."""
        lm = self.get_landmark(index)
        if lm:
            return lm.x_norm, lm.y_norm, lm.z_norm
        return None


def ensure_model_exists(model_path: str = DEFAULT_MODEL_PATH) -> str:
    """Downloads the MediaPipe PoseLandmarker task model if not present."""
    if not os.path.exists(model_path):
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        logger.info(f"Downloading MediaPipe PoseLandmarker model from {MODEL_URL}...")
        urllib.request.urlretrieve(MODEL_URL, model_path)
        logger.info(f"Model downloaded successfully to {model_path}.")
    return model_path


class PoseEstimator:
    """
    MediaPipe PoseLandmarker wrapper for player crop landmark extraction.
    Supports MediaPipe 0.10+ Tasks API and legacy solutions fallback.
    """

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL_PATH,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        model_complexity: int = 1
    ):
        self.model_path = ensure_model_exists(model_path)
        self._mode = "tasks"

        try:
            # Modern MediaPipe 0.10+ Tasks API
            base_options = mp.tasks.BaseOptions(model_asset_path=self.model_path)
            options = mp.tasks.vision.PoseLandmarkerOptions(
                base_options=base_options,
                running_mode=mp.tasks.vision.RunningMode.IMAGE,
                min_pose_detection_confidence=min_detection_confidence,
                min_pose_presence_confidence=min_tracking_confidence
            )
            self._landmarker = mp.tasks.vision.PoseLandmarker.create_from_options(options)
            logger.info("PoseEstimator initialized using MediaPipe Tasks API.")

        except Exception as e:
            logger.warning(f"Tasks API initialization fallback: {e}")
            if hasattr(mp, "solutions") and hasattr(mp.solutions, "pose"):
                self._mode = "legacy"
                self._pose = mp.solutions.pose.Pose(
                    static_image_mode=True,
                    model_complexity=model_complexity,
                    min_detection_confidence=min_detection_confidence
                )
                logger.info("PoseEstimator initialized using legacy MediaPipe solutions.")
            else:
                raise RuntimeError(f"Failed to initialize MediaPipe Pose: {e}")

    def estimate(self, image: np.ndarray, track_id: int = -1) -> PoseResult:
        """
        Runs pose estimation on a single player crop.

        Args:
            image: BGR player crop (OpenCV).
            track_id: Player track ID.

        Returns:
            PoseResult object.
        """
        if image is None or image.size == 0:
            return PoseResult(track_id=track_id, success=False, confidence=0.0)

        h, w = image.shape[:2]

        try:
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            if self._mode == "tasks":
                mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                detection_result = self._landmarker.detect(mp_img)

                if not detection_result.pose_landmarks or len(detection_result.pose_landmarks) == 0:
                    return PoseResult(track_id=track_id, success=False, confidence=0.0)

                raw_landmarks = detection_result.pose_landmarks[0]
                landmarks = []
                total_vis = 0.0

                for idx, lm in enumerate(raw_landmarks):
                    vis = getattr(lm, "visibility", 1.0)
                    if vis is None:
                        vis = getattr(lm, "presence", 1.0) or 1.0

                    landmark = Landmark(
                        index=idx,
                        x_norm=float(lm.x),
                        y_norm=float(lm.y),
                        z_norm=float(lm.z),
                        visibility=float(vis),
                        x_px=float(lm.x) * w,
                        y_px=float(lm.y) * h
                    )
                    landmarks.append(landmark)
                    total_vis += vis

                confidence = total_vis / max(len(landmarks), 1)
                return PoseResult(
                    track_id=track_id,
                    success=True,
                    confidence=round(confidence, 3),
                    landmarks=landmarks
                )

            else:
                legacy_res = self._pose.process(rgb)
                if legacy_res.pose_landmarks is None:
                    return PoseResult(track_id=track_id, success=False, confidence=0.0)

                landmarks = []
                total_vis = 0.0
                for idx, lm in enumerate(legacy_res.pose_landmarks.landmark):
                    landmark = Landmark(
                        index=idx,
                        x_norm=float(lm.x),
                        y_norm=float(lm.y),
                        z_norm=float(lm.z),
                        visibility=float(lm.visibility),
                        x_px=float(lm.x) * w,
                        y_px=float(lm.y) * h
                    )
                    landmarks.append(landmark)
                    total_vis += lm.visibility

                confidence = total_vis / max(len(landmarks), 1)
                return PoseResult(
                    track_id=track_id,
                    success=True,
                    confidence=round(confidence, 3),
                    landmarks=landmarks
                )

        except Exception as e:
            logger.warning(f"Pose estimation error for track_id={track_id}: {e}")
            return PoseResult(track_id=track_id, success=False, confidence=0.0)

    def estimate_batch(self, player_crops: List[Tuple[int, np.ndarray]]) -> List[PoseResult]:
        """Processes multiple player crops sequentially."""
        return [self.estimate(crop, track_id=tid) for tid, crop in player_crops]

    def draw_landmarks(self, image: np.ndarray, pose_result: PoseResult) -> np.ndarray:
        """
        Draws pose skeleton keypoints and connection lines on player crop using OpenCV.

        Args:
            image: BGR player crop.
            pose_result: Estimated PoseResult.

        Returns:
            Annotated BGR image.
        """
        if not pose_result.success or image is None:
            return image

        annotated = image.copy()

        # 1. Draw connection lines
        for idx1, idx2 in POSE_CONNECTIONS:
            pt1 = pose_result.get_point_px(idx1)
            pt2 = pose_result.get_point_px(idx2)

            if pt1 and pt2:
                p1 = (int(round(pt1[0])), int(round(pt1[1])))
                p2 = (int(round(pt2[0])), int(round(pt2[1])))
                cv2.line(annotated, p1, p2, (0, 255, 255), 2, cv2.LINE_AA)

        # 2. Draw keypoint circles
        for lm in pose_result.landmarks:
            if lm.visibility > 0.4:
                cx, cy = int(round(lm.x_px)), int(round(lm.y_px))
                cv2.circle(annotated, (cx, cy), 4, (0, 255, 0), -1, cv2.LINE_AA)
                cv2.circle(annotated, (cx, cy), 5, (0, 0, 0), 1, cv2.LINE_AA)

        return annotated

    def release(self):
        """Releases underlying landmarker resources."""
        if hasattr(self, "_landmarker"):
            self._landmarker.close()
        elif hasattr(self, "_pose"):
            self._pose.close()
        logger.info("PoseEstimator resources released.")
