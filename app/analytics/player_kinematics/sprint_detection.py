"""
Sprint Detection Module

Detects sprints using configurable rules based on speed thresholds,
minimum duration, and minimum distance.
"""

import logging
from typing import Dict, List, Optional
import numpy as np

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)


class SprintDetector:
    """
    Detects sprint events in player trajectories.
    """

    def __init__(
        self,
        speed_threshold_kmh: float = 25.0,
        min_duration_s: float = 2.0,
        min_distance_m: float = 10.0
    ):
        """
        Initializes SprintDetector.

        Args:
            speed_threshold_kmh: Minimum speed to qualify as sprint.
            min_duration_s: Minimum sprint duration in seconds.
            min_distance_m: Minimum sprint distance in meters.
        """
        self.speed_threshold_kmh = speed_threshold_kmh
        self.min_duration_s = min_duration_s
        self.min_distance_m = min_distance_m

    def detect_sprints(
        self,
        points: List[Dict]
    ) -> List[Dict]:
        """
        Detects sprints in a single player trajectory.

        Args:
            points: List of dicts with 'speed_kmh', 'timestamp', 'smoothed_world_position'.

        Returns:
            List of sprint dicts with sprint metrics.
        """
        if not points:
            return []

        sprints = []
        in_sprint = False
        sprint_indices = []

        for i, p in enumerate(points):
            if p.get('speed_kmh', 0) >= self.speed_threshold_kmh:
                if not in_sprint:
                    in_sprint = True
                    sprint_indices = [i]
                else:
                    sprint_indices.append(i)
            else:
                if in_sprint:
                    sprint = self._build_sprint(points, sprint_indices)
                    if sprint:
                        sprints.append(sprint)
                    in_sprint = False
                    sprint_indices = []

        # Handle trailing sprint
        if in_sprint and sprint_indices:
            sprint = self._build_sprint(points, sprint_indices)
            if sprint:
                sprints.append(sprint)

        return sprints

    def _build_sprint(
        self,
        points: List[Dict],
        indices: List[int]
    ) -> Optional[Dict]:
        """Builds sprint metrics from contiguous fast frames."""
        start_idx = indices[0]
        end_idx = indices[-1]

        duration = points[end_idx]['timestamp'] - points[start_idx]['timestamp']
        if duration < self.min_duration_s:
            return None

        distance = 0.0
        for i in range(start_idx + 1, end_idx + 1):
            prev = np.array(points[i-1]['smoothed_world_position'])
            curr = np.array(points[i]['smoothed_world_position'])
            distance += np.linalg.norm(curr - prev)

        if distance < self.min_distance_m:
            return None

        speeds = [points[i]['speed_kmh'] for i in indices]
        peak_speed = max(speeds)

        return {
            'track_id': points[0]['track_id'],
            'sprint_start_frame': points[start_idx]['frame_number'],
            'sprint_end_frame': points[end_idx]['frame_number'],
            'sprint_start_timestamp': points[start_idx]['timestamp'],
            'sprint_end_timestamp': points[end_idx]['timestamp'],
            'sprint_duration_s': round(duration, 2),
            'sprint_distance_m': round(distance, 2),
            'peak_speed_kmh': round(peak_speed, 2)
        }

    def detect_batch(
        self,
        tracked_players: Dict[int, List[Dict]]
    ) -> Dict[int, List[Dict]]:
        """
        Detects sprints for all players.

        Args:
            tracked_players: Dict mapping track_id to trajectory points.

        Returns:
            Dict mapping track_id to list of sprint dicts.
        """
        results = {}
        for track_id, points in tracked_players.items():
            sprints = self.detect_sprints(points)
            if sprints:
                results[track_id] = sprints
        return results

    def get_summary(
        self,
        all_sprints: Dict[int, List[Dict]]
    ) -> Dict[int, Dict]:
        """
        Returns sprint summary per player.

        Args:
            all_sprints: Dict mapping track_id to sprint list.

        Returns:
            Dict mapping track_id to sprint summary.
        """
        summary = {}
        for track_id, sprints in all_sprints.items():
            total_sprint_dist = sum(s['sprint_distance_m'] for s in sprints)
            summary[track_id] = {
                'sprint_count': len(sprints),
                'total_sprint_distance_m': round(total_sprint_dist, 2),
                'peak_speed_kmh': max((s['peak_speed_kmh'] for s in sprints), default=0.0)
            }
        return summary