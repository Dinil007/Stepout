"""
Player Statistics Module

Aggregates all per-player analytics (speed, distance, acceleration, possession, passes)
into a unified PlayerStats dataclass and provides export utilities.
"""

import logging
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any
import pandas as pd

from app.analytics.speed_estimator import SpeedEstimator
from app.analytics.distance_tracker import DistanceTracker
from app.analytics.acceleration_estimator import AccelerationEstimator

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)


@dataclass
class PlayerStats:
    """
    Aggregated statistics for a single tracked player (Phase 1 + Phase 2 Pose).
    """
    track_id: int
    team_id: Any = None
    total_distance_m: float = 0.0
    sprint_distance_m: float = 0.0
    running_distance_m: float = 0.0
    max_speed_kmh: float = 0.0
    avg_speed_kmh: float = 0.0
    peak_acceleration_ms2: float = 0.0
    peak_deceleration_ms2: float = 0.0
    sprint_count: int = 0
    possession_frames: int = 0
    frames_tracked: int = 0

    # Phase 2 Pose & Biomechanics Metrics
    cadence_spm: Optional[float] = None
    stride_length_norm: Optional[float] = None
    knee_drive_deg: Optional[float] = None
    hip_extension_deg: Optional[float] = None
    vertical_oscillation_norm: Optional[float] = None
    ground_contact_pct: Optional[float] = None
    running_efficiency: Optional[float] = None
    left_knee_angle_deg: Optional[float] = None
    right_knee_angle_deg: Optional[float] = None
    left_hip_angle_deg: Optional[float] = None
    right_hip_angle_deg: Optional[float] = None
    trunk_lean_deg: Optional[float] = None
    gait_pattern: str = "Unknown"
    injury_risk_level: str = "LOW"
    injury_risk_score: float = 0.0

    def to_dict(self) -> Dict:
        return asdict(self)


class PlayerStatisticsAggregator:
    """
    Combines outputs from SpeedEstimator, DistanceTracker, and AccelerationEstimator
    into unified per-player statistics profiles.
    """

    def __init__(
        self,
        speed_estimator: SpeedEstimator,
        distance_tracker: DistanceTracker,
        acceleration_estimator: AccelerationEstimator,
        team_assignments: Optional[Dict[int, Any]] = None,
        possession_frames: Optional[Dict[int, int]] = None
    ):
        """
        Initializes the PlayerStatisticsAggregator.

        Args:
            speed_estimator: Populated SpeedEstimator instance.
            distance_tracker: Populated DistanceTracker instance.
            acceleration_estimator: Populated AccelerationEstimator instance.
            team_assignments: Optional dict of track_id -> team_id.
            possession_frames: Optional dict of track_id -> possession frame count.
        """
        self._speed = speed_estimator
        self._distance = distance_tracker
        self._acceleration = acceleration_estimator
        self._team_assignments = team_assignments or {}
        self._possession_frames = possession_frames or {}

    def build_player_stats(self, track_id: int) -> PlayerStats:
        """
        Builds a complete PlayerStats object for a given player.

        Args:
            track_id: Player track ID.

        Returns:
            Populated PlayerStats instance.
        """
        return PlayerStats(
            track_id=track_id,
            team_id=self._team_assignments.get(track_id),
            total_distance_m=self._distance.get_total_distance(track_id),
            sprint_distance_m=self._distance.get_sprint_distance(track_id),
            running_distance_m=self._distance.get_running_distance(track_id),
            max_speed_kmh=self._speed.get_max_speed(track_id),
            avg_speed_kmh=self._speed.get_avg_speed(track_id),
            peak_acceleration_ms2=self._acceleration.get_peak_acceleration(track_id),
            peak_deceleration_ms2=self._acceleration.get_peak_deceleration(track_id),
            sprint_count=self._speed.get_sprint_count(track_id),
            possession_frames=self._possession_frames.get(track_id, 0),
            frames_tracked=len(self._distance.get_frame_distances(track_id))
        )

    def build_all_stats(self) -> List[PlayerStats]:
        """
        Aggregates PlayerStats for every tracked player.

        Returns:
            List of PlayerStats objects.
        """
        all_track_ids = set(
            list(self._distance.get_all_totals().keys()) +
            list(self._team_assignments.keys())
        )

        stats_list = [self.build_player_stats(tid) for tid in sorted(all_track_ids)]
        logger.info(f"Built statistics for {len(stats_list)} players.")
        return stats_list

    def to_dataframe(self) -> pd.DataFrame:
        """Returns all player statistics as a pandas DataFrame."""
        stats_list = self.build_all_stats()
        records = [s.to_dict() for s in stats_list]
        return pd.DataFrame(records)

    def save_csv(self, output_path: str) -> str:
        """Saves player statistics DataFrame as CSV."""
        df = self.to_dataframe()
        df.to_csv(output_path, index=False)
        logger.info(f"Player statistics saved to: {output_path}")
        return output_path
