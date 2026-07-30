"""
Injury Risk Assessment Module

Rule-based biomechanical injury risk evaluator. Assesses biomechanical risk factors:
knee valgus, excessive trunk lean, hip drop, asymmetry, reduced knee flexion (stiff-legged landing),
and landing instability.
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
from app.pose.biomechanics import BiomechanicsResult

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)

# Risk Thresholds
EXCESSIVE_TRUNK_LEAN_DEG: float = 20.0
REDUCED_KNEE_FLEXION_DEG: float = 145.0  # Stiffness indicator: knee angle near 180° upon landing
ASYMMETRY_THRESHOLD_PCT: float = 15.0
KNEE_VALGUS_X_THRESHOLD: float = 0.04    # Knee collapsing inward relative to hip-ankle line


@dataclass
class RiskAssessment:
    """Dataclass holding injury risk evaluation results."""
    track_id: int
    risk_level: str                         # "LOW", "MEDIUM", "HIGH"
    risk_score: float                       # 0 to 100 (higher = greater risk)
    flags: List[str] = field(default_factory=list)
    explanations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return asdict(self)


class InjuryRiskEvaluator:
    """
    Evaluates biomechanical risk factors per player frame/sequence.
    """

    def __init__(
        self,
        trunk_lean_threshold: float = EXCESSIVE_TRUNK_LEAN_DEG,
        reduced_flexion_threshold: float = REDUCED_KNEE_FLEXION_DEG,
        asymmetry_threshold: float = ASYMMETRY_THRESHOLD_PCT
    ):
        self.trunk_lean_threshold = trunk_lean_threshold
        self.reduced_flexion_threshold = reduced_flexion_threshold
        self.asymmetry_threshold = asymmetry_threshold

    def _check_knee_valgus(self, pose: PoseResult) -> List[Tuple[str, str, float]]:
        """
        Detects knee valgus (knee collapsing inward in 2D frontal plane view).
        Compares knee X coordinate against hip and ankle X coordinates.
        """
        issues = []
        # Left leg valgus: Knee X is significantly to the right of Hip-Ankle line (inward)
        left_hip = pose.get_landmark(LandmarkIndex.LEFT_HIP)
        left_knee = pose.get_landmark(LandmarkIndex.LEFT_KNEE)
        left_ankle = pose.get_landmark(LandmarkIndex.LEFT_ANKLE)

        if left_hip and left_knee and left_ankle:
            hip_ankle_mid_x = (left_hip.x_norm + left_ankle.x_norm) / 2.0
            # If knee collapses inward
            valgus_displacement = left_knee.x_norm - hip_ankle_mid_x
            if valgus_displacement > KNEE_VALGUS_X_THRESHOLD:
                issues.append((
                    "Left Knee Valgus",
                    f"Left knee collapses inward by {valgus_displacement:.3f} norm units, increasing ACL strain.",
                    25.0
                ))

        right_hip = pose.get_landmark(LandmarkIndex.RIGHT_HIP)
        right_knee = pose.get_landmark(LandmarkIndex.RIGHT_KNEE)
        right_ankle = pose.get_landmark(LandmarkIndex.RIGHT_ANKLE)

        if right_hip and right_knee and right_ankle:
            hip_ankle_mid_x = (right_hip.x_norm + right_ankle.x_norm) / 2.0
            valgus_displacement = hip_ankle_mid_x - right_knee.x_norm
            if valgus_displacement > KNEE_VALGUS_X_THRESHOLD:
                issues.append((
                    "Right Knee Valgus",
                    f"Right knee collapses inward by {valgus_displacement:.3f} norm units, increasing ACL strain.",
                    25.0
                ))

        return issues

    def _check_hip_drop(self, pose: PoseResult) -> Optional[Tuple[str, str, float]]:
        """
        Detects Trendelenburg sign (hip drop) by checking pelvis slope (Y difference between hips).
        """
        left_hip = pose.get_landmark(LandmarkIndex.LEFT_HIP)
        right_hip = pose.get_landmark(LandmarkIndex.RIGHT_HIP)

        if left_hip and right_hip:
            hip_diff_y = abs(left_hip.y_norm - right_hip.y_norm)
            if hip_diff_y > 0.05:  # Pelvic tilt threshold
                side = "Left" if left_hip.y_norm > right_hip.y_norm else "Right"
                return (
                    f"{side} Pelvic Hip Drop",
                    f"Significant pelvic drop ({hip_diff_y:.3f} norm units) indicating gluteus medius weakness.",
                    20.0
                )
        return None

    def evaluate(
        self,
        pose: PoseResult,
        biomechanics: Optional[BiomechanicsResult] = None
    ) -> RiskAssessment:
        """
        Evaluates injury risk for a given player pose frame.

        Args:
            pose: PoseResult from PoseEstimator.
            biomechanics: Optional BiomechanicsResult.

        Returns:
            RiskAssessment object.
        """
        track_id = pose.track_id
        flags: List[str] = []
        explanations: List[str] = []
        risk_score: float = 0.0

        if not pose.success:
            return RiskAssessment(
                track_id=track_id,
                risk_level="LOW",
                risk_score=0.0,
                flags=["No Pose Data"],
                explanations=["Pose estimation was unsuccessful for this frame."]
            )

        # 1. Knee Valgus Check
        valgus_issues = self._check_knee_valgus(pose)
        for flag, exp, weight in valgus_issues:
            flags.append(flag)
            explanations.append(exp)
            risk_score += weight

        # 2. Hip Drop Check
        hip_issue = self._check_hip_drop(pose)
        if hip_issue:
            flag, exp, weight = hip_issue
            flags.append(flag)
            explanations.append(exp)
            risk_score += weight

        # 3. Excessive Trunk Lean Check
        t_lean = trunk_lean_angle(pose)
        if t_lean is not None and t_lean > self.trunk_lean_threshold:
            flags.append("Excessive Trunk Lean")
            explanations.append(
                f"Trunk lean angle of {t_lean:.1f}° exceeds safe threshold of {self.trunk_lean_threshold}°, risking hamstring/back strain."
            )
            risk_score += 15.0

        # 4. Reduced Knee Flexion Check (Stiff landing / impact)
        lk = left_knee_angle(pose)
        rk = right_knee_angle(pose)
        if (lk is not None and lk > self.reduced_flexion_threshold) or (rk is not None and rk > self.reduced_flexion_threshold):
            flags.append("Reduced Knee Flexion (Stiff Stance)")
            explanations.append("Extended knee angle during ground contact increases vertical ground reaction forces.")
            risk_score += 15.0

        # 5. Asymmetry Check from Biomechanics
        if biomechanics and biomechanics.symmetry_index is not None:
            if biomechanics.symmetry_index > self.asymmetry_threshold:
                flags.append("High Bilateral Asymmetry")
                explanations.append(
                    f"Bilateral movement asymmetry of {biomechanics.symmetry_index:.1f}% exceeds safe threshold ({self.asymmetry_threshold}%)."
                )
                risk_score += 20.0

        # 6. Landing Instability Check (high vertical oscillation + ground contact)
        if biomechanics and biomechanics.vertical_oscillation_norm is not None:
            if biomechanics.vertical_oscillation_norm > 0.05:
                flags.append("Landing Instability")
                explanations.append("High vertical oscillation indicates unstable landing energy dissipation.")
                risk_score += 15.0

        # Cap risk score at 100.0
        risk_score = round(min(100.0, risk_score), 1)

        # Determine Risk Level
        if risk_score < 25.0:
            risk_level = "LOW"
        elif risk_score < 55.0:
            risk_level = "MEDIUM"
        else:
            risk_level = "HIGH"

        if not flags:
            explanations.append("No biomechanical risk factors detected. Movement pattern is within safe limits.")

        return RiskAssessment(
            track_id=track_id,
            risk_level=risk_level,
            risk_score=risk_score,
            flags=flags,
            explanations=explanations
        )
