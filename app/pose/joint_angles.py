"""
Joint Angles Module

Computes anatomical joint angles from MediaPipe 33 pose landmarks using
vector algebra. Returns angles in degrees for key joints: knee, hip, ankle,
elbow, shoulder, and trunk.
"""

import logging
import math
from typing import Optional, Tuple
import numpy as np

from app.pose.pose_estimator import PoseResult, LandmarkIndex

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)

# Minimum landmark visibility to consider it reliable
MIN_VISIBILITY: float = 0.4


def _get_point(pose: PoseResult, index: int) -> Optional[np.ndarray]:
    """
    Retrieves the 3D normalized (x, y, z) coordinate of a landmark.
    Returns None if the landmark is below minimum visibility threshold.
    """
    lm = pose.get_landmark(index)
    if lm is None or lm.visibility < MIN_VISIBILITY:
        return None
    return np.array([lm.x_norm, lm.y_norm, lm.z_norm], dtype=np.float64)


def _angle_between_three_points(
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray
) -> float:
    """
    Computes the angle at vertex B formed by rays BA and BC.

    Args:
        a: First endpoint point.
        b: Vertex (joint center).
        c: Second endpoint point.

    Returns:
        Angle in degrees [0, 180].
    """
    ba = a - b
    bc = c - b

    cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-9)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    return float(math.degrees(math.acos(cos_angle)))


def _angle_or_none(a: Optional[np.ndarray], b: Optional[np.ndarray], c: Optional[np.ndarray]) -> Optional[float]:
    """Returns computed angle or None if any point is invisible."""
    if a is None or b is None or c is None:
        return None
    return round(_angle_between_three_points(a, b, c), 2)


# ==========================================
# Knee Angles
# ==========================================
def left_knee_angle(pose: PoseResult) -> Optional[float]:
    """
    Angle at the left knee (Hip → Knee → Ankle).
    Full extension = ~180°, Flexion = decreases toward 0°.
    """
    return _angle_or_none(
        _get_point(pose, LandmarkIndex.LEFT_HIP),
        _get_point(pose, LandmarkIndex.LEFT_KNEE),
        _get_point(pose, LandmarkIndex.LEFT_ANKLE)
    )


def right_knee_angle(pose: PoseResult) -> Optional[float]:
    """Angle at the right knee (Hip → Knee → Ankle)."""
    return _angle_or_none(
        _get_point(pose, LandmarkIndex.RIGHT_HIP),
        _get_point(pose, LandmarkIndex.RIGHT_KNEE),
        _get_point(pose, LandmarkIndex.RIGHT_ANKLE)
    )


# ==========================================
# Hip Angles
# ==========================================
def left_hip_angle(pose: PoseResult) -> Optional[float]:
    """
    Angle at left hip (Shoulder → Hip → Knee).
    Represents hip flexion/extension in the sagittal plane.
    """
    return _angle_or_none(
        _get_point(pose, LandmarkIndex.LEFT_SHOULDER),
        _get_point(pose, LandmarkIndex.LEFT_HIP),
        _get_point(pose, LandmarkIndex.LEFT_KNEE)
    )


def right_hip_angle(pose: PoseResult) -> Optional[float]:
    """Angle at right hip (Shoulder → Hip → Knee)."""
    return _angle_or_none(
        _get_point(pose, LandmarkIndex.RIGHT_SHOULDER),
        _get_point(pose, LandmarkIndex.RIGHT_HIP),
        _get_point(pose, LandmarkIndex.RIGHT_KNEE)
    )


# ==========================================
# Ankle Angles
# ==========================================
def left_ankle_angle(pose: PoseResult) -> Optional[float]:
    """
    Angle at left ankle (Knee → Ankle → Foot Index).
    Represents plantar/dorsiflexion.
    """
    return _angle_or_none(
        _get_point(pose, LandmarkIndex.LEFT_KNEE),
        _get_point(pose, LandmarkIndex.LEFT_ANKLE),
        _get_point(pose, LandmarkIndex.LEFT_FOOT_INDEX)
    )


def right_ankle_angle(pose: PoseResult) -> Optional[float]:
    """Angle at right ankle (Knee → Ankle → Foot Index)."""
    return _angle_or_none(
        _get_point(pose, LandmarkIndex.RIGHT_KNEE),
        _get_point(pose, LandmarkIndex.RIGHT_ANKLE),
        _get_point(pose, LandmarkIndex.RIGHT_FOOT_INDEX)
    )


# ==========================================
# Elbow Angles
# ==========================================
def left_elbow_angle(pose: PoseResult) -> Optional[float]:
    """
    Angle at left elbow (Shoulder → Elbow → Wrist).
    Full extension = ~180°.
    """
    return _angle_or_none(
        _get_point(pose, LandmarkIndex.LEFT_SHOULDER),
        _get_point(pose, LandmarkIndex.LEFT_ELBOW),
        _get_point(pose, LandmarkIndex.LEFT_WRIST)
    )


def right_elbow_angle(pose: PoseResult) -> Optional[float]:
    """Angle at right elbow (Shoulder → Elbow → Wrist)."""
    return _angle_or_none(
        _get_point(pose, LandmarkIndex.RIGHT_SHOULDER),
        _get_point(pose, LandmarkIndex.RIGHT_ELBOW),
        _get_point(pose, LandmarkIndex.RIGHT_WRIST)
    )


# ==========================================
# Shoulder Angles
# ==========================================
def left_shoulder_angle(pose: PoseResult) -> Optional[float]:
    """
    Angle at left shoulder (Elbow → Shoulder → Hip).
    Represents arm elevation in the sagittal plane.
    """
    return _angle_or_none(
        _get_point(pose, LandmarkIndex.LEFT_ELBOW),
        _get_point(pose, LandmarkIndex.LEFT_SHOULDER),
        _get_point(pose, LandmarkIndex.LEFT_HIP)
    )


def right_shoulder_angle(pose: PoseResult) -> Optional[float]:
    """Angle at right shoulder (Elbow → Shoulder → Hip)."""
    return _angle_or_none(
        _get_point(pose, LandmarkIndex.RIGHT_ELBOW),
        _get_point(pose, LandmarkIndex.RIGHT_SHOULDER),
        _get_point(pose, LandmarkIndex.RIGHT_HIP)
    )


# ==========================================
# Trunk Lean Angle
# ==========================================
def trunk_lean_angle(pose: PoseResult) -> Optional[float]:
    """
    Measures the forward lean of the trunk from vertical.
    Computed from the vector between midpoint of shoulders and midpoint of hips
    relative to a vertical reference axis.

    Returns:
        Trunk lean in degrees. 0° = fully upright, positive = forward lean.
    """
    left_shoulder = _get_point(pose, LandmarkIndex.LEFT_SHOULDER)
    right_shoulder = _get_point(pose, LandmarkIndex.RIGHT_SHOULDER)
    left_hip = _get_point(pose, LandmarkIndex.LEFT_HIP)
    right_hip = _get_point(pose, LandmarkIndex.RIGHT_HIP)

    if any(p is None for p in [left_shoulder, right_shoulder, left_hip, right_hip]):
        return None

    shoulder_mid = (left_shoulder + right_shoulder) / 2.0
    hip_mid = (left_hip + right_hip) / 2.0

    trunk_vector = shoulder_mid - hip_mid  # Points upward (from hip to shoulder)
    vertical = np.array([0.0, -1.0, 0.0])  # Negative Y = upward in MediaPipe

    cos_angle = np.dot(trunk_vector, vertical) / (np.linalg.norm(trunk_vector) * np.linalg.norm(vertical) + 1e-9)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    return round(float(math.degrees(math.acos(cos_angle))), 2)


# ==========================================
# Convenience: All Joint Angles at Once
# ==========================================
def compute_all_joint_angles(pose: PoseResult) -> dict:
    """
    Computes all joint angles from a PoseResult and returns them as a dict.

    Returns:
        Dict with angle values in degrees. None if landmark was not visible.
    """
    return {
        "left_knee_deg": left_knee_angle(pose),
        "right_knee_deg": right_knee_angle(pose),
        "left_hip_deg": left_hip_angle(pose),
        "right_hip_deg": right_hip_angle(pose),
        "left_ankle_deg": left_ankle_angle(pose),
        "right_ankle_deg": right_ankle_angle(pose),
        "left_elbow_deg": left_elbow_angle(pose),
        "right_elbow_deg": right_elbow_angle(pose),
        "left_shoulder_deg": left_shoulder_angle(pose),
        "right_shoulder_deg": right_shoulder_angle(pose),
        "trunk_lean_deg": trunk_lean_angle(pose),
    }
