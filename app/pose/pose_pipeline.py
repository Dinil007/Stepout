"""
Pose & Biomechanics Integrated Pipeline

Integrates player crop pose estimation, 3D landmark extraction, joint angle computation,
running biomechanics analysis, temporal gait evaluation, and injury risk assessment.
"""

import logging
from typing import Dict, List, Tuple, Optional, Any
import cv2
import numpy as np

from app.pose.pose_estimator import PoseEstimator, PoseResult
from app.pose.joint_angles import compute_all_joint_angles
from app.pose.biomechanics import BiomechanicsAnalyzer, BiomechanicsResult
from app.pose.gait_analysis import GaitAnalyzer, GaitReport
from app.pose.injury_risk import InjuryRiskEvaluator, RiskAssessment

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)


class PosePipeline:
    """
    End-to-End Pose & Biomechanics Processing Pipeline.
    """

    def __init__(
        self,
        fps: float = 30.0,
        model_complexity: int = 1,
        min_detection_confidence: float = 0.5
    ):
        self.fps = fps
        self.pose_estimator = PoseEstimator(
            model_complexity=model_complexity,
            min_detection_confidence=min_detection_confidence
        )
        self.biomechanics_analyzer = BiomechanicsAnalyzer(fps=fps)
        self.gait_analyzer = GaitAnalyzer(fps=fps)
        self.injury_evaluator = InjuryRiskEvaluator()

    @staticmethod
    def crop_player(frame: np.ndarray, bbox: Tuple[int, int, int, int], margin_pct: float = 0.1) -> np.ndarray:
        """
        Crops player region from frame with an optional margin padding.

        Args:
            frame: Full BGR video frame.
            bbox: (x1, y1, x2, y2) bounding box.
            margin_pct: Additional padding percentage around box.

        Returns:
            Cropped BGR image array.
        """
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = bbox

        bw = x2 - x1
        bh = y2 - y1

        mx = int(bw * margin_pct)
        my = int(bh * margin_pct)

        cx1 = max(0, x1 - mx)
        cy1 = max(0, y1 - my)
        cx2 = min(w, x2 + mx)
        cy2 = min(h, y2 + my)

        return frame[cy1:cy2, cx1:cx2]

    def process_crop(
        self,
        crop: np.ndarray,
        track_id: int = -1,
        frame_number: int = 0
    ) -> Dict[str, Any]:
        """
        Processes a single cropped player image through the full pose pipeline.

        Args:
            crop: BGR player crop.
            track_id: Player track ID.
            frame_number: Current frame index.

        Returns:
            Dict containing pose_result, joint_angles, biomechanics, gait_report,
            injury_risk, and annotated_crop.
        """
        # 1. MediaPipe Pose Estimation
        pose_result: PoseResult = self.pose_estimator.estimate(crop, track_id=track_id)

        # 2. Joint Angles
        joint_angles = compute_all_joint_angles(pose_result) if pose_result.success else {}

        # 3. Biomechanics
        biomechanics: BiomechanicsResult = self.biomechanics_analyzer.analyze(pose_result, frame_number)

        # 4. Gait Analysis
        if pose_result.success:
            self.gait_analyzer.update(biomechanics)
        gait_report: GaitReport = self.gait_analyzer.generate_report(track_id)

        # 5. Injury Risk Evaluation
        injury_risk: RiskAssessment = self.injury_evaluator.evaluate(pose_result, biomechanics)

        # 6. Draw Pose Skeleton Annotation on Crop
        annotated_crop = self.pose_estimator.draw_landmarks(crop, pose_result)

        # Overlay Risk Banner if high/medium
        if injury_risk.risk_level in ["MEDIUM", "HIGH"]:
            color = (0, 0, 255) if injury_risk.risk_level == "HIGH" else (0, 165, 255)
            cv2.putText(
                annotated_crop,
                f"RISK: {injury_risk.risk_level}",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )

        return {
            "track_id": track_id,
            "frame_number": frame_number,
            "pose_result": pose_result,
            "joint_angles": joint_angles,
            "biomechanics": biomechanics,
            "gait_report": gait_report,
            "injury_risk": injury_risk,
            "annotated_crop": annotated_crop
        }

    def process_frame_players(
        self,
        frame: np.ndarray,
        player_detections: List[Dict],
        frame_number: int = 0
    ) -> List[Dict[str, Any]]:
        """
        Extracts crops and processes pose for all player detections in a video frame.

        Args:
            frame: Full BGR frame.
            player_detections: List of dicts containing 'track_id' and 'bbox'.
            frame_number: Current frame index.

        Returns:
            List of result dicts per player.
        """
        results = []
        for det in player_detections:
            tid = det.get("track_id", -1)
            bbox = det.get("bbox")
            if bbox is None:
                continue

            crop = self.crop_player(frame, bbox)
            if crop.size == 0:
                continue

            res = self.process_crop(crop, track_id=tid, frame_number=frame_number)
            results.append(res)

        return results

    def release(self):
        """Releases underlying pose estimator resources."""
        self.pose_estimator.release()
