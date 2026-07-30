"""
Biomechanics Analysis Module

Computes advanced running biomechanics metrics from MediaPipe pose landmarks,
including cadence, stride length, knee drive, hip extension, trunk lean,
symmetry index, and estimated running efficiency.
"""

import logging
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple
import numpy as np

from app.pose.pose_estimator import PoseResult, LandmarkIndex
from app.pose.joint_angles import (
    left_knee_angle, right_knee_angle,
    left_hip_angle, right_hip_angle,
    trunk_lean_angle
)

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)

# Minimum landmark visibility threshold
MIN_VISIBILITY: float = 0.4

# Reference for good running efficiency (degree benchmarks)
OPTIMAL_TRUNK_LEAN_DEG: float = 10.0
OPTIMAL_KNEE_DRIVE_DEG: float = 70.0


@dataclass
class BiomechanicsResult:
    """Biomechanical metrics computed from a single pose frame."""
    track_id: int
    frame_number: int

    # Cadence (estimated steps per minute based on foot alternation)
    cadence_spm: Optional[float] = None

    # Stride & Step
    stride_length_norm: Optional[float] = None   # Normalized to body height proportion
    step_frequency_hz: Optional[float] = None

    # Lower limb drive
    left_knee_drive_deg: Optional[float] = None
    right_knee_drive_deg: Optional[float] = None

    # Hip extension
    left_hip_extension_deg: Optional[float] = None
    right_hip_extension_deg: Optional[float] = None

    # Trunk
    trunk_lean_deg: Optional[float] = None

    # Symmetry Index (0.0 = perfect symmetry, higher = more asymmetry)
    symmetry_index: Optional[float] = None

    # Vertical oscillation (relative displacement of hip midpoint in Y)
    vertical_oscillation_norm: Optional[float] = None

    # Ground contact estimation (is heel near ground plane)
    left_ground_contact: bool = False
    right_ground_contact: bool = False

    # Running efficiency score (0-100)
    efficiency_score: Optional[float] = None

    def to_dict(self) -> Dict:
        return asdict(self)


class BiomechanicsAnalyzer:
    """
    Analyzes running biomechanics from sequential PoseResult frames.
    Maintains per-track history for cadence and oscillation calculation.
    """

    def __init__(self, fps: float, ground_contact_y_threshold: float = 0.85):
        """
        Initializes the BiomechanicsAnalyzer.

        Args:
            fps: Video frame rate.
            ground_contact_y_threshold: Normalized Y threshold below which a foot is grounded.
                                        (In MediaPipe, Y=1.0 is the bottom of image.)
        """
        if fps <= 0:
            raise ValueError(f"FPS must be positive. Got: {fps}")

        self.fps = fps
        self.dt = 1.0 / fps
        self.ground_contact_y_threshold = ground_contact_y_threshold

        # State per track: last hip Y position for vertical oscillation
        self._last_hip_y: Dict[int, float] = {}

        # Foot state tracking for cadence
        self._foot_states: Dict[int, Dict[str, bool]] = {}
        self._step_timestamps: Dict[int, List[float]] = {}
        self._frame_counter: Dict[int, int] = {}

    def _get_point(self, pose: PoseResult, index: int) -> Optional[np.ndarray]:
        lm = pose.get_landmark(index)
        if lm is None or lm.visibility < MIN_VISIBILITY:
            return None
        return np.array([lm.x_norm, lm.y_norm, lm.z_norm], dtype=np.float64)

    def _compute_symmetry_index(
        self, left_val: Optional[float], right_val: Optional[float]
    ) -> Optional[float]:
        """
        Computes the Symmetry Index (SI) between left and right joint angles.
        SI = 2 * |L - R| / (L + R) * 100
        SI of 0% = perfect symmetry, >10% = notable asymmetry.
        """
        if left_val is None or right_val is None:
            return None
        denominator = abs(left_val) + abs(right_val)
        if denominator < 1e-6:
            return 0.0
        si = 2.0 * abs(left_val - right_val) / denominator * 100.0
        return round(si, 2)

    def _estimate_stride_length(self, pose: PoseResult) -> Optional[float]:
        """
        Estimates stride length as the Euclidean distance between left and right
        ankle normalized coordinates.
        """
        left_ankle = self._get_point(pose, LandmarkIndex.LEFT_ANKLE)
        right_ankle = self._get_point(pose, LandmarkIndex.RIGHT_ANKLE)
        if left_ankle is None or right_ankle is None:
            return None
        dist = np.linalg.norm(left_ankle[:2] - right_ankle[:2])
        return round(float(dist), 4)

    def _estimate_vertical_oscillation(
        self, pose: PoseResult, track_id: int
    ) -> Optional[float]:
        """
        Estimates hip vertical displacement from previous frame as a proxy
        for vertical oscillation (excess vertical movement while running).
        """
        left_hip = self._get_point(pose, LandmarkIndex.LEFT_HIP)
        right_hip = self._get_point(pose, LandmarkIndex.RIGHT_HIP)
        if left_hip is None or right_hip is None:
            return None
        hip_y = float((left_hip[1] + right_hip[1]) / 2.0)
        osc = None
        if track_id in self._last_hip_y:
            osc = round(abs(hip_y - self._last_hip_y[track_id]), 4)
        self._last_hip_y[track_id] = hip_y
        return osc

    def _detect_ground_contact(
        self, pose: PoseResult
    ) -> Tuple[bool, bool]:
        """
        Estimates whether each foot is in ground contact based on
        normalized Y coordinate threshold.
        """
        left_heel = pose.get_landmark(LandmarkIndex.LEFT_HEEL)
        right_heel = pose.get_landmark(LandmarkIndex.RIGHT_HEEL)

        left_contact = (
            left_heel is not None
            and left_heel.visibility > MIN_VISIBILITY
            and left_heel.y_norm >= self.ground_contact_y_threshold
        )
        right_contact = (
            right_heel is not None
            and right_heel.visibility > MIN_VISIBILITY
            and right_heel.y_norm >= self.ground_contact_y_threshold
        )
        return left_contact, right_contact

    def _estimate_cadence(self, track_id: int, frame_number: int) -> Optional[float]:
        """
        Estimates cadence (steps/min) from step event timestamp history.
        Uses sliding window of last 10 step intervals.
        """
        timestamps = self._step_timestamps.get(track_id, [])
        if len(timestamps) < 2:
            return None
        intervals = [timestamps[i + 1] - timestamps[i] for i in range(len(timestamps) - 1)]
        avg_interval = np.mean(intervals[-10:])  # Sliding window last 10 steps
        if avg_interval < 1e-6:
            return None
        cadence = 60.0 / avg_interval  # steps per minute
        return round(float(cadence), 1)

    def _estimate_efficiency(
        self,
        trunk_lean: Optional[float],
        knee_drive: Optional[float],
        symmetry: Optional[float],
        vertical_osc: Optional[float]
    ) -> Optional[float]:
        """
        Estimates a running efficiency score (0-100) from key biomechanical indicators.
        Higher is better.
        """
        score = 100.0
        deductions = 0.0

        # Penalize excessive trunk lean (optimal ~10°)
        if trunk_lean is not None:
            lean_deviation = abs(trunk_lean - OPTIMAL_TRUNK_LEAN_DEG)
            deductions += min(lean_deviation * 0.5, 20.0)

        # Penalize poor knee drive
        if knee_drive is not None:
            knee_deviation = max(0.0, OPTIMAL_KNEE_DRIVE_DEG - knee_drive)
            deductions += min(knee_deviation * 0.3, 15.0)

        # Penalize high asymmetry
        if symmetry is not None:
            deductions += min(symmetry * 0.5, 20.0)

        # Penalize excessive vertical oscillation (>0.03 normalized = inefficient)
        if vertical_osc is not None and vertical_osc > 0.03:
            deductions += min((vertical_osc - 0.03) * 200.0, 15.0)

        return round(max(0.0, score - deductions), 1)

    def analyze(self, pose: PoseResult, frame_number: int) -> BiomechanicsResult:
        """
        Performs complete biomechanics analysis for a single pose frame.

        Args:
            pose: PoseResult from PoseEstimator.
            frame_number: Current video frame number.

        Returns:
            BiomechanicsResult with all computed metrics.
        """
        result = BiomechanicsResult(track_id=pose.track_id, frame_number=frame_number)

        if not pose.success:
            return result

        track_id = pose.track_id

        # Knee angles (knee drive = smaller knee angle = more flexion)
        lk = left_knee_angle(pose)
        rk = right_knee_angle(pose)
        result.left_knee_drive_deg = lk
        result.right_knee_drive_deg = rk

        # Hip extension
        lh = left_hip_angle(pose)
        rh = right_hip_angle(pose)
        result.left_hip_extension_deg = lh
        result.right_hip_extension_deg = rh

        # Trunk lean
        result.trunk_lean_deg = trunk_lean_angle(pose)

        # Symmetry Index (based on knee angles)
        result.symmetry_index = self._compute_symmetry_index(lk, rk)

        # Stride length
        result.stride_length_norm = self._estimate_stride_length(pose)

        # Step frequency
        if result.stride_length_norm is not None and result.stride_length_norm > 0:
            result.step_frequency_hz = round(self.fps / max(1.0, result.stride_length_norm * 100.0), 2)

        # Vertical oscillation
        result.vertical_oscillation_norm = self._estimate_vertical_oscillation(pose, track_id)

        # Ground contact
        result.left_ground_contact, result.right_ground_contact = self._detect_ground_contact(pose)

        # Record step event for cadence
        if track_id not in self._step_timestamps:
            self._step_timestamps[track_id] = []
        if result.left_ground_contact or result.right_ground_contact:
            prev_states = self._foot_states.get(track_id, {"left": False, "right": False})
            current_states = {"left": result.left_ground_contact, "right": result.right_ground_contact}
            if current_states != prev_states:
                self._step_timestamps[track_id].append(frame_number / self.fps)
            self._foot_states[track_id] = current_states

        result.cadence_spm = self._estimate_cadence(track_id, frame_number)

        # Running efficiency score
        result.efficiency_score = self._estimate_efficiency(
            trunk_lean=result.trunk_lean_deg,
            knee_drive=lk,
            symmetry=result.symmetry_index,
            vertical_osc=result.vertical_oscillation_norm
        )

        return result
