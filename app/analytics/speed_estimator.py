"""
Speed Estimator Module

Computes real-time player speed by comparing consecutive pitch coordinate positions,
applies Exponential Moving Average (EMA) smoothing to suppress tracking jitter,
and outputs speed metrics in both m/s and km/h.
"""

import logging
from typing import Dict, List, Tuple, Optional
import numpy as np

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)

# Speed classification thresholds (km/h)
WALK_THRESHOLD_KMH: float = 7.0
JOG_THRESHOLD_KMH: float = 14.0
RUN_THRESHOLD_KMH: float = 20.0
SPRINT_THRESHOLD_KMH: float = 25.0

# Smoothing factor for Exponential Moving Average (0 < alpha <= 1)
DEFAULT_EMA_ALPHA: float = 0.3


class SpeedEstimator:
    """
    Estimates instantaneous and smoothed player speed from consecutive
    2D pitch coordinate positions using EMA filtering.
    """

    def __init__(
        self,
        fps: float,
        ema_alpha: float = DEFAULT_EMA_ALPHA,
        max_displacement_m: float = 2.0,
        min_movement_m: float = 0.0
    ):
        """
        Initializes the SpeedEstimator.

        Args:
            fps: Video frame rate (frames per second).
            ema_alpha: EMA smoothing factor (higher = less smoothing).
            max_displacement_m: Maximum allowed displacement per frame (m). Larger values are filtered as tracking artifacts.
            min_movement_m: Minimum displacement to consider as movement. Below this, speed=0.
        """
        if fps <= 0:
            raise ValueError(f"FPS must be positive. Got: {fps}")
        if not (0 < ema_alpha <= 1.0):
            raise ValueError(f"EMA alpha must be in (0, 1]. Got: {ema_alpha}")
        if max_displacement_m <= 0:
            raise ValueError(f"max_displacement_m must be positive. Got: {max_displacement_m}")
        if min_movement_m < 0:
            raise ValueError(f"min_movement_m must be non-negative. Got: {min_movement_m}")

        self.fps = fps
        self.dt = 1.0 / fps
        self.ema_alpha = ema_alpha
        self.max_displacement_m = max_displacement_m
        self.min_movement_m = min_movement_m

        # State: track_id -> last known pitch position
        self._last_positions: Dict[int, Tuple[float, float]] = {}

        # State: track_id -> smoothed speed (m/s)
        self._smoothed_speeds_ms: Dict[int, float] = {}

        # Historical speed log: track_id -> [speed_kmh, ...]
        self._speed_history: Dict[int, List[float]] = {}

    def _compute_displacement(
        self,
        prev: Tuple[float, float],
        curr: Tuple[float, float]
    ) -> float:
        """Computes Euclidean displacement in meters between two pitch positions."""
        dx = curr[0] - prev[0]
        dy = curr[1] - prev[1]
        return float(np.sqrt(dx * dx + dy * dy))

    def _apply_ema(self, track_id: int, raw_speed_ms: float) -> float:
        """Applies EMA smoothing to the raw instantaneous speed."""
        if track_id not in self._smoothed_speeds_ms:
            self._smoothed_speeds_ms[track_id] = raw_speed_ms
        else:
            prev = self._smoothed_speeds_ms[track_id]
            self._smoothed_speeds_ms[track_id] = (
                self.ema_alpha * raw_speed_ms + (1.0 - self.ema_alpha) * prev
            )
        return self._smoothed_speeds_ms[track_id]

    @staticmethod
    def ms_to_kmh(speed_ms: float) -> float:
        """Converts speed from meters per second to kilometers per hour."""
        return speed_ms * 3.6

    @staticmethod
    def classify_speed(speed_kmh: float) -> str:
        """
        Classifies player movement intensity based on speed threshold.

        Returns:
            One of: 'Standing', 'Walking', 'Jogging', 'Running', 'Sprinting'.
        """
        if speed_kmh < 1.0:
            return "Standing"
        elif speed_kmh < WALK_THRESHOLD_KMH:
            return "Walking"
        elif speed_kmh < JOG_THRESHOLD_KMH:
            return "Jogging"
        elif speed_kmh < RUN_THRESHOLD_KMH:
            return "Running"
        else:
            return "Sprinting"

    def update(
        self,
        track_id: int,
        position_m: Tuple[float, float]
    ) -> Optional[Dict[str, float | str]]:
        """
        Updates speed estimation for a single player with their latest position.

        Args:
            track_id: Player track ID.
            position_m: Current (x, y) position in real-world meters.

        Returns:
            Dict with speed metrics, or None if no previous position exists.
            Returns zeroed metrics when tracking artifacts are detected.
        """
        if track_id in self._last_positions:
            prev_pos = self._last_positions[track_id]
            displacement_m = self._compute_displacement(prev_pos, position_m)

            # Detect tracking artifacts: position jumps > max_displacement_m
            # indicate track ID switches or detection failures, not real player movement.
            if displacement_m > self.max_displacement_m:
                logger.debug(
                    "Track %d: filtered position jump %.2fm (likely tracking artifact)",
                    track_id, displacement_m
                )
                # Reuse previous valid speed to avoid corrupt metrics
                smoothed_ms = self._smoothed_speeds_ms.get(track_id, 0.0)
                speed_kmh = self.ms_to_kmh(smoothed_ms)
                intensity = self.classify_speed(speed_kmh)
            elif displacement_m < self.min_movement_m:
                # Suppress jitter: treat as standing
                smoothed_ms = 0.0
                speed_kmh = 0.0
                intensity = self.classify_speed(speed_kmh)
            else:
                raw_speed_ms = displacement_m / self.dt
                smoothed_ms = self._apply_ema(track_id, raw_speed_ms)
                speed_kmh = self.ms_to_kmh(smoothed_ms)
                intensity = self.classify_speed(speed_kmh)

            # Record history
            self._speed_history.setdefault(track_id, []).append(speed_kmh)

            self._last_positions[track_id] = position_m

            return {
                "track_id": track_id,
                "speed_ms": round(smoothed_ms, 3),
                "speed_kmh": round(speed_kmh, 2),
                "intensity": intensity
            }

        self._last_positions[track_id] = position_m
        return None

    def update_batch(
        self,
        player_positions: Dict[int, Tuple[float, float]]
    ) -> Dict[int, Dict]:
        """
        Batch updates speed for multiple players in a single frame.

        Args:
            player_positions: Dict mapping track_id to (x, y) real-world meter position.

        Returns:
            Dict mapping track_id to speed metrics.
        """
        results = {}
        for track_id, position in player_positions.items():
            result = self.update(track_id, position)
            if result is not None:
                results[track_id] = result
        return results

    def get_max_speed(self, track_id: int) -> float:
        """Returns the recorded maximum speed (km/h) for a track ID."""
        history = self._speed_history.get(track_id, [])
        return float(max(history)) if history else 0.0

    def get_avg_speed(self, track_id: int) -> float:
        """Returns the time-averaged speed (km/h) for a track ID."""
        history = self._speed_history.get(track_id, [])
        return float(np.mean(history)) if history else 0.0

    def get_sprint_count(self, track_id: int) -> int:
        """Counts the number of frames a player's speed exceeded the sprint threshold."""
        history = self._speed_history.get(track_id, [])
        return int(np.sum(np.array(history) >= SPRINT_THRESHOLD_KMH))

    def get_speed_history(self, track_id: int) -> List[float]:
        """Returns full speed (km/h) history for a track ID."""
        return self._speed_history.get(track_id, [])

    def get_summary(self, track_id: int) -> Dict:
        """Returns a complete speed analytics summary for a player."""
        return {
            "track_id": track_id,
            "max_speed_kmh": self.get_max_speed(track_id),
            "avg_speed_kmh": self.get_avg_speed(track_id),
            "sprint_count": self.get_sprint_count(track_id),
            "frames_tracked": len(self._speed_history.get(track_id, []))
        }

    def clear(self, track_id: Optional[int] = None) -> None:
        """Resets state for one or all players."""
        if track_id is not None:
            self._last_positions.pop(track_id, None)
            self._smoothed_speeds_ms.pop(track_id, None)
            self._speed_history.pop(track_id, None)
        else:
            self._last_positions.clear()
            self._smoothed_speeds_ms.clear()
            self._speed_history.clear()
