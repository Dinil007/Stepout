"""
Player Speed Module

Computes instantaneous speed, rolling average speed, maximum speed, and average speed
using timestamps rather than fixed frame intervals. Rejects unrealistic speed values.
"""

import logging
from typing import Dict, List, Optional
import numpy as np

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)


class SpeedCalculator:
    """
    Computes per-frame and aggregate speed metrics for player trajectories.
    """

    def __init__(
        self,
        max_speed_kmh: float = 40.0,
        rolling_window_frames: int = 5
    ):
        """
        Initializes SpeedCalculator.

        Args:
            max_speed_kmh: Maximum physically plausible player speed in km/h.
            rolling_window_frames: Window size for rolling average speed.
        """
        self.max_speed_ms = max_speed_kmh / 3.6
        self.rolling_window_frames = rolling_window_frames

    @staticmethod
    def ms_to_kmh(speed_ms: float) -> float:
        """Converts meters per second to kilometers per hour."""
        return speed_ms * 3.6

    def compute_frame_speed(
        self,
        prev_pos: np.ndarray,
        curr_pos: np.ndarray,
        dt: float
    ) -> float:
        """
        Computes instantaneous speed between two positions.

        Args:
            prev_pos: Previous position (x, y).
            curr_pos: Current position (x, y).
            dt: Time difference in seconds.

        Returns:
            Speed in m/s.
        """
        displacement = np.linalg.norm(curr_pos - prev_pos)
        if dt <= 0:
            return 0.0
        speed = displacement / dt
        return float(np.clip(speed, 0.0, self.max_speed_ms))

    def process_track(
        self,
        points: List[Dict]
    ) -> List[Dict]:
        """
        Computes speed metrics for a full trajectory.

        Args:
            points: List of dicts with 'smoothed_world_position' and 'timestamp'.

        Returns:
            List of dicts with added speed metrics.
        """
        if len(points) < 2:
            return points

        speeds = []
        for i in range(len(points)):
            if i == 0:
                speeds.append(0.0)
                continue

            prev_pos = np.array(points[i-1]['smoothed_world_position'])
            curr_pos = np.array(points[i]['smoothed_world_position'])
            dt = points[i]['timestamp'] - points[i-1]['timestamp']

            if dt <= 0:
                speeds.append(0.0)
            else:
                speed = self.compute_frame_speed(prev_pos, curr_pos, dt)
                speeds.append(speed)

        # Compute rolling average
        w = self.rolling_window_frames
        rolling = []
        for i in range(len(speeds)):
            start = max(0, i - w + 1)
            window = speeds[start:i+1]
            valid = [s for s in window if s > 0]
            rolling.append(float(np.mean(valid)) if valid else 0.0)

        track_id = points[0]['track_id']
        for i, p in enumerate(points):
            p['track_id'] = track_id
            p['speed_ms'] = round(speeds[i], 3)
            p['speed_kmh'] = round(self.ms_to_kmh(speeds[i]), 2)
            p['rolling_speed_kmh'] = round(self.ms_to_kmh(rolling[i]), 2)

        return points

    def process_batch(
        self,
        smoothed_tracks: Dict[int, List[Dict]]
    ) -> Dict[int, List[Dict]]:
        """
        Computes speed for all players.

        Args:
            smoothed_tracks: Dict mapping track_id to smoothed trajectory points.

        Returns:
            Dict mapping track_id to trajectory with speed metrics.
        """
        results = {}
        for track_id, points in smoothed_tracks.items():
            results[track_id] = self.process_track(points)
            logger.info("Track %d: speed computed for %d frames", track_id, len(points))
        return results

    def get_summary(
        self,
        tracks_speed: Dict[int, List[Dict]]
    ) -> Dict[int, Dict[str, float]]:
        """
        Returns aggregate speed summaries for all players.

        Args:
            tracks_speed: Dict mapping track_id to trajectory with speed metrics.

        Returns:
            Dict mapping track_id to summary metrics.
        """
        summary = {}
        for track_id, points in tracks_speed.items():
            speeds = [p['speed_ms'] for p in points if p.get('speed_ms', 0) > 0]
            if not speeds:
                summary[track_id] = {
                    'avg_speed_kmh': 0.0,
                    'max_speed_kmh': 0.0,
                    'valid_frames': 0
                }
                continue

            max_s = max(speeds)
            avg_s = float(np.mean(speeds))
            summary[track_id] = {
                'avg_speed_kmh': round(self.ms_to_kmh(avg_s), 2),
                'max_speed_kmh': round(self.ms_to_kmh(max_s), 2),
                'valid_frames': len(speeds)
            }
        return summary