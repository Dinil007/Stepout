"""
Ball Metrics Module

Generates per-ball and team-level summary statistics.
"""

import logging
from typing import Dict, List
import numpy as np

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)


class BallMetricsGenerator:
    """
    Generates ball analytics summaries.
    """

    def generate_ball_summary(self, points: List[Dict], touches: List[Dict], passes: List[Dict]) -> Dict:
        """
        Generates summary for a single ball track.

        Args:
            points: Trajectory points with metrics.
            touches: Detected touch events.
            passes: Detected pass events.

        Returns:
            Ball summary dict.
        """
        if not points:
            return {'valid': False}

        speeds = [p.get('ball_speed_ms', 0) for p in points if p.get('ball_speed_ms', 0) > 0]
        avg_speed_ms = float(np.mean(speeds)) if speeds else 0.0
        max_speed_ms = max(speeds) if speeds else 0.0

        total_distance = 0.0
        for i in range(1, len(points)):
            prev = np.array(points[i-1].get('smoothed_world_position', points[i-1].get('clean_world_position', [0,0])))
            curr = np.array(points[i].get('smoothed_world_position', points[i].get('clean_world_position', [0,0])))
            total_distance += np.linalg.norm(curr - prev)

        return {
            'valid': True,
            'total_distance_m': round(total_distance, 2),
            'avg_speed_kmh': round(avg_speed_ms * 3.6, 2),
            'max_speed_kmh': round(max_speed_ms * 3.6, 2),
            'touches': len(touches),
            'passes': len(passes)
        }

    def generate_team_possession(self, possession_events: List[Dict], team_ids: List[str]) -> Dict[str, Dict]:
        """
        Generates possession stats per team.

        Args:
            possession_events: List of possession event dicts.
            team_ids: List of team identifiers.

        Returns:
            Dict mapping team_id to possession stats.
        """
        team_stats = {tid: {'possession_frames': 0, 'possession_events': 0} for tid in team_ids}

        for event in possession_events:
            tid = event.get('team_id')
            if tid in team_stats:
                team_stats[tid]['possession_frames'] += event['frame_end'] - event['frame_start'] + 1
                team_stats[tid]['possession_events'] += 1

        total_frames = sum(s['possession_frames'] for s in team_stats.values()) or 1

        result = {}
        for tid, stats in team_stats.items():
            result[tid] = {
                'possession_pct': round(stats['possession_frames'] / total_frames * 100, 2),
                'possession_time_s': round(stats['possession_frames'] / 25.0, 2),  # assuming 25 fps
                'possession_events': stats['possession_events']
            }
        return result

    def generate_team_possession_from_detector(self, detector) -> Dict[str, Dict]:
        """
        Generates possession stats from PossessionDetector instance.

        Args:
            detector: PossessionDetector instance with accumulated stats.

        Returns:
            Dict mapping team_id to possession stats including percentages.
        """
        summary = detector.get_team_possession_summary()
        percentages = detector.get_possession_percentage()

        result = {}
        for pct_key, pct in percentages.items():
            # Convert percentage key to team name
            team_name = pct_key.replace('_pct', '') if '_pct' in pct_key else pct_key
            if team_name == 'Free_Ball':
                team_name = 'Free Ball'

            # Look up possession time from summary using team name
            time_lookup = team_name if team_name != 'Free Ball' else 'Free Ball'
            possession_time = summary['total_possession_time_seconds'].get(time_lookup, 0)

            result[team_name] = {
                'possession_pct': pct,
                'possession_time_s': possession_time
            }

        return result

    def get_summary(self, passes: List[Dict]) -> Dict:
        """Returns aggregate pass summary."""
        if not passes:
            return {'total_passes': 0, 'successful_passes': 0}

        successful = [p for p in passes if p.get('successful', False)]
        return {
            'total_passes': len(passes),
            'successful_passes': len(successful),
            'total_distance_m': round(sum(p['distance_m'] for p in passes), 2)
        }