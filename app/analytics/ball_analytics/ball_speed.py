"""
Ball Speed Module

Computes instantaneous speed, rolling speed, average speed, and maximum speed
for the football using timestamps. Rejects impossible values.
"""

import logging
from typing import Dict, List
import numpy as np

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)


class BallSpeedCalculator:
    """
    Computes ball speed metrics from world coordinates.
    """

    def __init__(
        self,
        max_speed_kmh: float = 120.0,
        rolling_window_frames: int = 5
    ):
        self.max_speed_ms = max_speed_kmh / 3.6
        self.rolling_window_frames = rolling_window_frames

    @staticmethod
    def ms_to_kmh(speed_ms: float) -> float:
        return speed_ms * 3.6

    def compute_frame_speed(self, prev_pos: np.ndarray, curr_pos: np.ndarray, dt: float) -> float:
        displacement = np.linalg.norm(curr_pos - prev_pos)
        if dt <= 0:
            return 0.0
        speed = displacement / dt
        return float(np.clip(speed, 0.0, self.max_speed_ms))

    def process_track(self, points: List[Dict]) -> List[Dict]:
        if len(points) < 2:
            for p in points:
                p['ball_speed_ms'] = 0.0
                p['ball_speed_kmh'] = 0.0
                p['rolling_speed_kmh'] = 0.0
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
            p['ball_speed_ms'] = round(speeds[i], 3)
            p['ball_speed_kmh'] = round(self.ms_to_kmh(speeds[i]), 2)
            p['rolling_speed_kmh'] = round(self.ms_to_kmh(rolling[i]), 2)

        return points

    def process_batch(self, smoothed_tracks: Dict[int, List[Dict]]) -> Dict[int, List[Dict]]:
        results = {}
        for track_id, points in smoothed_tracks.items():
            results[track_id] = self.process_track(points)
            logger.info("Ball track %d: speed computed for %d frames", track_id, len(points))
        return results

    def get_summary(self, tracks_speed: Dict[int, List[Dict]]) -> Dict[int, Dict[str, float]]:
        summary = {}
        for track_id, points in tracks_speed.items():
            speeds = [p['ball_speed_ms'] for p in points if p.get('ball_speed_ms', 0) > 0]
            if not speeds:
                summary[track_id] = {'avg_speed_kmh': 0.0, 'max_speed_kmh': 0.0, 'valid_frames': 0}
                continue
            max_s = max(speeds)
            avg_s = float(np.mean(speeds))
            summary[track_id] = {
                'avg_speed_kmh': round(self.ms_to_kmh(avg_s), 2),
                'max_speed_kmh': round(self.ms_to_kmh(max_s), 2),
                'valid_frames': len(speeds)
            }
        return summary