"""
Motion + Appearance Association Module.

Combines ByteTrack's motion-based matching with OSNet appearance similarity.
This is a POST-processing layer that runs AFTER ByteTrack's native association.

Architecture:
    ByteTrack Output → Motion Score → Appearance Score → Final Score → Refined Tracks

The motion score comes from ByteTrack's internal IoU/Kalman matching.
Appearance score is computed via cosine similarity of OSNet embeddings.
Final score is a weighted combination.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

from app.tracking.tracker_config import ReIDConfig

logger = logging.getLogger(__name__)


class MotionAppearanceAssociator:
    """
    Combines motion and appearance scores for track refinement.

    Does NOT replace ByteTrack association.
    Instead, it adds an appearance verification layer on top of ByteTrack's
    motion-based matching to improve identity consistency.
    """

    def __init__(self, config: ReIDConfig) -> None:
        """
        Args:
            config: ReID configuration with weights and thresholds
        """
        self.appearance_weight = config.appearance_weight
        self.motion_weight = config.motion_weight
        self.similarity_threshold = config.similarity_threshold

    def compute_final_score(
        self,
        motion_score: float,
        appearance_score: float,
    ) -> float:
        """
        Compute the final combined score.

        Formula:
            FinalScore = motion_weight × MotionScore + appearance_weight × AppearanceScore

        Args:
            motion_score: ByteTrack's motion/IoU match score (0-1)
            appearance_score: Cosine similarity of embeddings (-1 to 1, clipped to 0-1)

        Returns:
            Weighted final score
        """
        # Clamp appearance score to [0, 1] for meaningful weighting
        appearance_score = max(0.0, min(1.0, appearance_score))
        # Clamp motion score to [0, 1]
        motion_score = max(0.0, min(1.0, motion_score))

        return (
            self.motion_weight * motion_score
            + self.appearance_weight * appearance_score
        )

    def verify_match(
        self,
        track_embedding: Optional[np.ndarray],
        detection_embedding: Optional[np.ndarray],
        motion_score: float,
    ) -> Tuple[float, float, float]:
        """
        Verify a ByteTrack match with appearance similarity.

        Args:
            track_embedding: Running average embedding of the track
            detection_embedding: Embedding of the new detection
            motion_score: ByteTrack's motion/IoU match score

        Returns:
            Tuple of (appearance_score, motion_score, final_score)
        """
        appearance_score = 0.0

        # Compute appearance similarity if both embeddings exist
        if track_embedding is not None and detection_embedding is not None:
            appearance_score = float(np.dot(track_embedding, detection_embedding))

        # Combine scores
        final_score = self.compute_final_score(motion_score, appearance_score)

        return appearance_score, motion_score, final_score

    def compute_cross_similarity_matrix(
        self,
        track_embeddings: List[Optional[np.ndarray]],
        detection_embeddings: List[Optional[np.ndarray]],
    ) -> np.ndarray:
        """
        Compute cosine similarity matrix between tracks and detections.

        Args:
            track_embeddings: List of track average embeddings (or None)
            detection_embeddings: List of detection embeddings (or None)

        Returns:
            (N_tracks × N_detections) similarity matrix
        """
        n_tracks = len(track_embeddings)
        n_dets = len(detection_embeddings)
        sim_matrix = np.zeros((n_tracks, n_dets), dtype=np.float32)

        for t_idx, t_emb in enumerate(track_embeddings):
            for d_idx, d_emb in enumerate(detection_embeddings):
                if t_emb is not None and d_emb is not None:
                    sim_matrix[t_idx, d_idx] = float(np.dot(t_emb, d_emb))
                else:
                    sim_matrix[t_idx, d_idx] = 0.0

        return sim_matrix

    def detect_potential_id_switch(
        self,
        old_track_id: int,
        new_track_id: int,
        appearance_similarity: float,
        motion_distance: float,
        association_score: float,
    ) -> Dict:
        """
        Detect and log potential ID switches.

        Conditions suggesting an ID switch:
        1. Old track disappears
        2. New track appears nearby
        3. High appearance similarity between old and new
        4. Consistent motion pattern

        Args:
            old_track_id: Previously lost track ID
            new_track_id: Newly created track ID
            appearance_similarity: Cosine similarity between embeddings
            motion_distance: Spatial distance in pixels
            association_score: Combined motion + appearance score

        Returns:
            Dict with switch info, or None if no switch detected
        """
        switch_info = {
            "frame": 0,
            "old_track_id": old_track_id,
            "new_track_id": new_track_id,
            "cosine_similarity": round(appearance_similarity, 4),
            "motion_distance": round(motion_distance, 2),
            "association_score": round(association_score, 4),
            "reason": "",
        }

        # Determine reason
        reasons = []
        if appearance_similarity > self.similarity_threshold:
            reasons.append("high_appearance_similarity")
        if motion_distance < 50:  # Close spatial proximity
            reasons.append("close_spatial_proximity")
        if association_score > 0.6:
            reasons.append("high_combined_score")

        switch_info["reason"] = "+".join(reasons) if reasons else "unknown"
        return switch_info

    def get_config(self) -> Dict:
        """Get current association configuration."""
        return {
            "appearance_weight": self.appearance_weight,
            "motion_weight": self.motion_weight,
            "motion_plus_appearance": self.motion_weight + self.appearance_weight,
            "similarity_threshold": self.similarity_threshold,
        }