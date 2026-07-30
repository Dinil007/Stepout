"""
Player Metrics Module

Generates per-player statistics combining all kinematics measurements.
"""

import logging
from typing import Dict, List
import numpy as np

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)


class PlayerMetricsGenerator:
    """
    Generates comprehensive per-player kinematics summaries.
    """

    def generate_summary(
        self,
        track_id: int,
        points: List[Dict],
        sprints: List[Dict]
    ) -> Dict:
        """
        Generates a full player summary.

        Args:
            track_id: Player track ID.
            points: Trajectory points with all computed metrics.
            sprints: Detected sprint events.

        Returns:
            Summary dict with all player kinematics metrics.
        """
        if not points:
            return {'track_id': track_id, 'valid': False}

        # Distance
        total_distance = 0.0
        for i in range(1, len(points)):
            prev = np.array(points[i-1]['smoothed_world_position'])
            curr = np.array(points[i]['smoothed_world_position'])
            total_distance += np.linalg.norm(curr - prev)

        # Speed
        speeds = [p.get('speed_ms', 0) for p in points if p.get('speed_ms', 0) > 0]
        avg_speed_ms = float(np.mean(speeds)) if speeds else 0.0
        max_speed_ms = max(speeds) if speeds else 0.0

        # Acceleration
        accels = [p.get('acceleration_ms2', 0) for p in points if p.get('acceleration_ms2', 0) != 0]
        avg_accel = float(np.mean(accels)) if accels else 0.0
        max_accel = max(accels) if accels else 0.0
        max_decel = min(accels) if accels else 0.0

        # Sprints
        sprint_count = len(sprints)
        sprint_distance = sum(s['sprint_distance_m'] for s in sprints)
        peak_speed_sprint = max((s['peak_speed_kmh'] for s in sprints), default=0.0)

        # Time
        if len(points) > 1:
            time_tracked = points[-1]['timestamp'] - points[0]['timestamp']
        else:
            time_tracked = 0.0

        # Heading change
        heading_changes = [abs(p.get('heading_change', 0)) for p in points if p.get('heading_change', 0) != 0]
        avg_heading_change = float(np.mean(heading_changes)) if heading_changes else 0.0

        return {
            'track_id': track_id,
            'valid': True,
            'total_distance_m': round(total_distance, 2),
            'avg_speed_kmh': round(self._ms_to_kmh(avg_speed_ms), 2),
            'max_speed_kmh': round(self._ms_to_kmh(max_speed_ms), 2),
            'avg_acceleration_ms2': round(avg_accel, 3),
            'max_acceleration_ms2': round(max_accel, 3),
            'max_deceleration_ms2': round(max_decel, 3),
            'sprint_count': sprint_count,
            'sprint_distance_m': round(sprint_distance, 2),
            'peak_sprint_speed_kmh': round(peak_speed_sprint, 2),
            'high_intensity_runs': 0,  # Placeholder, computed separately
            'avg_heading_change_deg': round(avg_heading_change, 2),
            'time_tracked_s': round(time_tracked, 2),
            'valid_frames': len(points),
            'rejected_frames': 0  # Populated by validation
        }

    def generate_batch_summary(
        self,
        all_tracks: Dict[int, List[Dict]],
        all_sprints: Dict[int, List[Dict]]
    ) -> Dict[int, Dict]:
        """
        Generates summaries for all players.

        Args:
            all_tracks: Dict mapping track_id to trajectory with all metrics.
            all_sprints: Dict mapping track_id to sprint list.

        Returns:
            Dict mapping track_id to summary metrics.
        """
        summaries = {}
        for track_id, points in all_tracks.items():
            sprints = all_sprints.get(track_id, [])
            summaries[track_id] = self.generate_summary(track_id, points, sprints)
        return summaries

    @staticmethod
    def _ms_to_kmh(speed_ms: float) -> float:
        """Converts m/s to km/h."""
        return speed_ms * 3.6