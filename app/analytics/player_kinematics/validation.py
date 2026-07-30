"""
Player Kinematics Validation Module

Detects and logs data quality issues in player kinematics outputs:
missing positions, invalid coordinates, negative distances,
impossible speeds, impossible accelerations, trajectory jumps.
"""

import logging
from typing import Dict, List
import numpy as np

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)


class KinematicsValidator:
    """
    Validates kinematics data and logs anomalies.
    """

    def __init__(
        self,
        max_speed_kmh: float = 40.0,
        max_acceleration_ms2: float = 10.0,
        max_jump_m: float = 5.0
    ):
        """
        Initializes KinematicsValidator.

        Args:
            max_speed_kmh: Maximum plausible player speed.
            max_acceleration_ms2: Maximum plausible acceleration.
            max_jump_m: Maximum plausible displacement per frame.
        """
        self.max_speed_ms = max_speed_kmh / 3.6
        self.max_acceleration_ms2 = max_acceleration_ms2
        self.max_jump_m = max_jump_m

    def validate_track(
        self,
        points: List[Dict]
    ) -> Dict:
        """
        Validates a single trajectory.

        Args:
            points: Trajectory points with all computed metrics.

        Returns:
            Validation report dict.
        """
        issues = {
            'missing_positions': 0,
            'invalid_coordinates': 0,
            'negative_distances': 0,
            'impossible_speeds': 0,
            'impossible_accelerations': 0,
            'trajectory_jumps': 0,
            'total_issues': 0
        }

        if not points:
            return issues

        for i, p in enumerate(points):
            pos = p.get('smoothed_world_position') or p.get('clean_world_position')
            if pos is None or pos[0] is None or pos[1] is None:
                issues['missing_positions'] += 1
                continue

            if np.isnan(pos[0]) or np.isnan(pos[1]) or np.isinf(pos[0]) or np.isinf(pos[1]):
                issues['invalid_coordinates'] += 1
                continue

            if i > 0:
                prev = np.array(points[i-1].get('smoothed_world_position') or points[i-1].get('clean_world_position'))
                curr = np.array(pos)
                dist = np.linalg.norm(curr - prev)
                if dist < 0:
                    issues['negative_distances'] += 1
                elif dist > self.max_jump_m:
                    issues['trajectory_jumps'] += 1

            speed_ms = p.get('speed_ms', 0)
            if speed_ms > self.max_speed_ms:
                issues['impossible_speeds'] += 1

            accel = p.get('acceleration_ms2', 0)
            if abs(accel) > self.max_acceleration_ms2:
                issues['impossible_accelerations'] += 1

        issues['total_issues'] = sum(issues.values()) - issues['total_issues']
        return issues

    def validate_batch(
        self,
        all_tracks: Dict[int, List[Dict]]
    ) -> Dict[int, Dict]:
        """
        Validates all trajectories.

        Args:
            all_tracks: Dict mapping track_id to trajectory points.

        Returns:
            Dict mapping track_id to validation report.
        """
        reports = {}
        for track_id, points in all_tracks.items():
            reports[track_id] = self.validate_track(points)
        return reports

    def get_global_report(
        self,
        all_reports: Dict[int, Dict]
    ) -> Dict:
        """
        Aggregates validation reports.

        Args:
            all_reports: Dict mapping track_id to validation report.

        Returns:
            Global validation report.
        """
        totals = {
            'players_processed': len(all_reports),
            'missing_positions': 0,
            'invalid_coordinates': 0,
            'negative_distances': 0,
            'impossible_speeds': 0,
            'impossible_accelerations': 0,
            'trajectory_jumps': 0,
            'total_issues': 0
        }

        for report in all_reports.values():
            for k in totals:
                if k != 'players_processed':
                    totals[k] += report.get(k, 0)

        totals['total_issues'] = sum(v for k, v in totals.items() if k != 'players_processed')
        return totals