"""
Ball Acceleration Module

Computes instantaneous acceleration, average acceleration, and maximum acceleration
for the football using smoothed speed values.
"""

import logging
from typing import Dict, List
import numpy as np

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)


class BallAccelerationCalculator:
    """
    Computes ball acceleration metrics from speed data.
    """

    def __init__(self, rolling_window_frames: int = 5):
        self.rolling_window_frames = rolling_window_frames

    def process_track(self, points: List[Dict]) -> List[Dict]:
        if len(points) < 2:
            for p in points:
                p['ball_acceleration_ms2'] = 0.0
                p['rolling_acceleration_ms2'] = 0.0
            return points

        accelerations = []
        for i in range(len(points)):
            if i == 0:
                accelerations.append(0.0)
                continue
            dt = points[i]['timestamp'] - points[i-1]['timestamp']
            if dt <= 0:
                accelerations.append(0.0)
                continue
            dv = points[i]['ball_speed_ms'] - points[i-1]['ball_speed_ms']
            accel = dv / dt
            accelerations.append(float(np.clip(accel, -50.0, 50.0)))

        w = self.rolling_window_frames
        rolling = []
        for i in range(len(accelerations)):
            start = max(0, i - w + 1)
            window = accelerations[start:i+1]
            valid = [a for a in window if a != 0]
            rolling.append(float(np.mean(valid)) if valid else 0.0)

        track_id = points[0]['track_id']
        for i, p in enumerate(points):
            p['track_id'] = track_id
            p['ball_acceleration_ms2'] = round(accelerations[i], 3)
            p['rolling_acceleration_ms2'] = round(rolling[i], 3)

        return points

    def process_batch(self, tracks_speed: Dict[int, List[Dict]]) -> Dict[int, List[Dict]]:
        results = {}
        for track_id, points in tracks_speed.items():
            results[track_id] = self.process_track(points)
            logger.info("Ball track %d: acceleration computed for %d frames", track_id, len(points))
        return results

    def get_summary(self, tracks_accel: Dict[int, List[Dict]]) -> Dict[int, Dict[str, float]]:
        summary = {}
        for track_id, points in tracks_accel.items():
            accels = [p['ball_acceleration_ms2'] for p in points if p.get('ball_acceleration_ms2', 0) != 0]
            if not accels:
                summary[track_id] = {'avg_acceleration_ms2': 0.0, 'max_acceleration_ms2': 0.0, 'valid_frames': 0}
                continue
            max_a = max(accels)
            avg_a = float(np.mean(accels))
            summary[track_id] = {
                'avg_acceleration_ms2': round(avg_a, 3),
                'max_acceleration_ms2': round(max_a, 3),
                'valid_frames': len(accels)
            }
        return summary