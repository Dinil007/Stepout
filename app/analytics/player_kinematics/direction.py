"""
Player Direction Module

Computes velocity vectors, movement angles, and heading changes for player trajectories.
"""

import logging
from typing import Dict, List
import numpy as np

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)


class DirectionAnalyzer:
    """
    Computes movement direction, heading, and velocity vectors for player trajectories.
    """

    def __init__(self, heading_smooth_window: int = 3):
        """
        Initializes DirectionAnalyzer.

        Args:
            heading_smooth_window: Window size for smoothing heading changes.
        """
        self.heading_smooth_window = heading_smooth_window

    def process_track(
        self,
        points: List[Dict]
    ) -> List[Dict]:
        """
        Computes direction metrics for a full trajectory.

        Args:
            points: List of dicts with 'smoothed_world_position' and 'timestamp'.

        Returns:
            List of dicts with added direction metrics.
        """
        if len(points) < 2:
            for p in points:
                p['vx'] = 0.0
                p['vy'] = 0.0
                p['heading'] = 0.0
                p['heading_change'] = 0.0
            return points

        vx_list = []
        vy_list = []
        heading_list = []
        heading_changes = []

        for i in range(len(points)):
            if i == 0:
                vx_list.append(0.0)
                vy_list.append(0.0)
                heading_list.append(0.0)
                heading_changes.append(0.0)
                continue

            dt = points[i]['timestamp'] - points[i-1]['timestamp']
            if dt <= 0:
                vx_list.append(0.0)
                vy_list.append(0.0)
                heading_list.append(heading_list[-1] if heading_list else 0.0)
                heading_changes.append(0.0)
                continue

            dx = points[i]['smoothed_world_position'][0] - points[i-1]['smoothed_world_position'][0]
            dy = points[i]['smoothed_world_position'][1] - points[i-1]['smoothed_world_position'][1]

            vx = dx / dt
            vy = dy / dt

            heading = np.degrees(np.arctan2(vy, vx)) if (vx != 0 or vy != 0) else heading_list[-1]

            # Heading change
            prev_heading = heading_list[-1]
            change = heading - prev_heading
            # Normalize to [-180, 180]
            while change > 180:
                change -= 360
            while change < -180:
                change += 360

            vx_list.append(vx)
            vy_list.append(vy)
            heading_list.append(heading)
            heading_changes.append(change)

        # Smooth heading changes
        w = self.heading_smooth_window
        smoothed_changes = []
        for i in range(len(heading_changes)):
            start = max(0, i - w + 1)
            window = heading_changes[start:i+1]
            smoothed_changes.append(float(np.mean(window)))

        track_id = points[0]['track_id']
        for i, p in enumerate(points):
            p['track_id'] = track_id
            p['vx'] = round(vx_list[i], 3)
            p['vy'] = round(vy_list[i], 3)
            p['heading'] = round(heading_list[i], 2)
            p['heading_change'] = round(smoothed_changes[i], 2)

        return points

    def process_batch(
        self,
        tracks_speed: Dict[int, List[Dict]]
    ) -> Dict[int, List[Dict]]:
        """
        Computes direction for all players.

        Args:
            tracks_speed: Dict mapping track_id to trajectory with speed metrics.

        Returns:
            Dict mapping track_id to trajectory with direction metrics.
        """
        results = {}
        for track_id, points in tracks_speed.items():
            results[track_id] = self.process_track(points)
            logger.info("Track %d: direction computed for %d frames", track_id, len(points))
        return results

    def get_summary(
        self,
        tracks_direction: Dict[int, List[Dict]]
    ) -> Dict[int, Dict[str, float]]:
        """
        Returns aggregate direction summaries for all players.

        Args:
            tracks_direction: Dict mapping track_id to trajectory with direction metrics.

        Returns:
            Dict mapping track_id to summary metrics.
        """
        summary = {}
        for track_id, points in tracks_direction.items():
            changes = [abs(p['heading_change']) for p in points if p.get('heading_change', 0) != 0]
            if changes:
                summary[track_id] = {
                    'avg_heading_change_deg': round(float(np.mean(changes)), 2),
                    'valid_frames': len(changes)
                }
            else:
                summary[track_id] = {
                    'avg_heading_change_deg': 0.0,
                    'valid_frames': 0
                }
        return summary