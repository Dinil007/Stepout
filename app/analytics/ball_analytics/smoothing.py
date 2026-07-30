"""
Ball Trajectory Smoothing Module

Applies Moving Average, Savitzky-Golay, or Kalman smoothing to ball trajectories.
Configuration driven, defaults to Savitzky-Golay.
"""

import logging
from typing import Dict, List
import numpy as np

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)


class BallTrajectorySmoother:
    """
    Smooths ball trajectories using configurable methods.
    """

    def __init__(
        self,
        method: str = 'savgol',
        window_size: int = 5,
        polyorder: int = 2,
        kalman_process_noise: float = 0.01,
        kalman_measurement_noise: float = 0.1
    ):
        self.method = method
        self.window_size = window_size
        self.polyorder = polyorder
        self.kalman_process_noise = kalman_process_noise
        self.kalman_measurement_noise = kalman_measurement_noise

    def _moving_average(self, positions: np.ndarray) -> np.ndarray:
        smoothed = np.zeros_like(positions)
        w = self.window_size
        for i in range(len(positions)):
            start = max(0, i - w // 2)
            end = min(len(positions), i + w // 2 + 1)
            smoothed[i] = np.mean(positions[start:end], axis=0)
        return smoothed

    def _savgol(self, positions: np.ndarray) -> np.ndarray:
        try:
            from scipy.signal import savgol_filter
            w = min(self.window_size, len(positions) - 1)
            if w % 2 == 0:
                w += 1
            p = min(self.polyorder, w - 1)
            x = savgol_filter(positions[:, 0], w, p, mode='interp')
            y = savgol_filter(positions[:, 1], w, p, mode='interp')
            return np.column_stack([x, y])
        except ImportError:
            logger.warning("scipy not available, falling back to moving average")
            return self._moving_average(positions)

    def _kalman(self, positions: np.ndarray) -> np.ndarray:
        try:
            from pykalman import KalmanFilter
            kf = KalmanFilter(
                transition_matrices=np.eye(2),
                observation_matrices=np.eye(2),
                transition_covariance=self.kalman_process_noise * np.eye(2),
                observation_covariance=self.kalman_measurement_noise * np.eye(2)
            )
            smoothed, _ = kf.smooth(positions)
            return smoothed
        except ImportError:
            logger.warning("pykalman not available, falling back to savgol")
            return self._savgol(positions)

    def smooth_trajectory(self, cleaned_points: List[Dict]) -> List[Dict]:
        if len(cleaned_points) < 3:
            for p in cleaned_points:
                p['smoothed_world_position'] = p['clean_world_position']
            return cleaned_points

        positions = np.array([p['clean_world_position'] for p in cleaned_points])

        if self.method == 'moving_avg':
            smoothed = self._moving_average(positions)
        elif self.method == 'kalman':
            smoothed = self._kalman(positions)
        else:
            smoothed = self._savgol(positions)

        for i, p in enumerate(cleaned_points):
            p['smoothed_world_position'] = smoothed[i].tolist()

        return cleaned_points

    def smooth_batch(self, cleaned_tracks: Dict[int, List[Dict]]) -> Dict[int, List[Dict]]:
        smoothed_tracks = {}
        for track_id, points in cleaned_tracks.items():
            smoothed_tracks[track_id] = self.smooth_trajectory(points)
            logger.info("Ball track %d smoothed with %d points", track_id, len(points))
        return smoothed_tracks