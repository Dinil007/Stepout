"""
Acceleration Estimator Module

Computes instantaneous player acceleration from consecutive speed measurements.
Applies EMA smoothing and classifies acceleration events (positive, negative, neutral).
"""

import logging
from typing import Dict, List, Optional
import numpy as np

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)

# EMA smoothing factor for acceleration
DEFAULT_EMA_ALPHA: float = 0.3

# Threshold (m/s²) to classify significant acceleration or deceleration
ACCELERATION_THRESHOLD_MS2: float = 1.5


class AccelerationEstimator:
    """
    Estimates player acceleration by computing first-order differences in smoothed speed.
    """

    def __init__(self, fps: float, ema_alpha: float = DEFAULT_EMA_ALPHA):
        """
        Initializes the AccelerationEstimator.

        Args:
            fps: Frames per second.
            ema_alpha: EMA smoothing coefficient.
        """
        if fps <= 0:
            raise ValueError(f"FPS must be positive. Got: {fps}")

        self.fps = fps
        self.dt = 1.0 / fps
        self.ema_alpha = ema_alpha

        # State: track_id -> previous smoothed speed (m/s)
        self._last_speed_ms: Dict[int, float] = {}

        # State: track_id -> smoothed acceleration
        self._smoothed_accel: Dict[int, float] = {}

        # Historical acceleration log: track_id -> [accel_ms2, ...]
        self._accel_history: Dict[int, List[float]] = {}

    def _apply_ema(self, track_id: int, raw_accel: float) -> float:
        """Applies Exponential Moving Average to raw acceleration."""
        if track_id not in self._smoothed_accel:
            self._smoothed_accel[track_id] = raw_accel
        else:
            prev = self._smoothed_accel[track_id]
            self._smoothed_accel[track_id] = (
                self.ema_alpha * raw_accel + (1.0 - self.ema_alpha) * prev
            )
        return self._smoothed_accel[track_id]

    @staticmethod
    def classify_acceleration(accel_ms2: float) -> str:
        """
        Classifies the acceleration event type.

        Returns:
            One of: 'Accelerating', 'Decelerating', 'Neutral'.
        """
        if accel_ms2 > ACCELERATION_THRESHOLD_MS2:
            return "Accelerating"
        elif accel_ms2 < -ACCELERATION_THRESHOLD_MS2:
            return "Decelerating"
        else:
            return "Neutral"

    def update(self, track_id: int, speed_ms: float) -> Optional[Dict]:
        """
        Updates acceleration estimate for a single player.

        Args:
            track_id: Player track ID.
            speed_ms: Current smoothed speed in m/s.

        Returns:
            Dict with acceleration metrics or None if no previous speed.
        """
        if track_id not in self._last_speed_ms:
            self._last_speed_ms[track_id] = speed_ms
            return None

        raw_accel = (speed_ms - self._last_speed_ms[track_id]) / self.dt
        smoothed = self._apply_ema(track_id, raw_accel)
        self._last_speed_ms[track_id] = speed_ms

        self._accel_history.setdefault(track_id, []).append(smoothed)

        return {
            "track_id": track_id,
            "acceleration_ms2": round(smoothed, 3),
            "type": self.classify_acceleration(smoothed)
        }

    def update_batch(self, player_speeds_ms: Dict[int, float]) -> Dict[int, Dict]:
        """Batch acceleration update for multiple players in one frame."""
        results = {}
        for track_id, speed in player_speeds_ms.items():
            result = self.update(track_id, speed)
            if result:
                results[track_id] = result
        return results

    def get_peak_acceleration(self, track_id: int) -> float:
        """Returns the peak positive acceleration (m/s²) recorded."""
        history = self._accel_history.get(track_id, [])
        return float(max(history)) if history else 0.0

    def get_peak_deceleration(self, track_id: int) -> float:
        """Returns the peak deceleration (most negative m/s²) recorded."""
        history = self._accel_history.get(track_id, [])
        return float(min(history)) if history else 0.0

    def get_summary(self, track_id: int) -> Dict:
        """Returns full acceleration analytics summary for a player."""
        return {
            "track_id": track_id,
            "peak_acceleration_ms2": self.get_peak_acceleration(track_id),
            "peak_deceleration_ms2": self.get_peak_deceleration(track_id),
            "frames_tracked": len(self._accel_history.get(track_id, []))
        }

    def clear(self, track_id: Optional[int] = None) -> None:
        """Clears state for one or all players."""
        if track_id is not None:
            self._last_speed_ms.pop(track_id, None)
            self._smoothed_accel.pop(track_id, None)
            self._accel_history.pop(track_id, None)
        else:
            self._last_speed_ms.clear()
            self._smoothed_accel.clear()
            self._accel_history.clear()
