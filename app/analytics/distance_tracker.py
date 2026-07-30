"""
Distance Tracker Module

Accumulates cumulative distance covered per player across frames using
real-world pitch coordinate positions (meters). Supports per-player distance
retrieval, sprint distance separation, and zone-based distance breakdown.
"""

import logging
from typing import Dict, List, Tuple, Optional
import numpy as np

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)

# Sprint speed threshold to separate sprint distance from base running distance
SPRINT_SPEED_THRESHOLD_KMH: float = 20.0


class DistanceTracker:
    """
    Tracks cumulative real-world distance covered by each player across video frames.
    """

    def __init__(self):
        """Initializes the DistanceTracker with empty player state stores."""
        # track_id -> cumulative total distance in meters
        self._total_distance: Dict[int, float] = {}

        # track_id -> total sprint distance in meters (speed > threshold)
        self._sprint_distance: Dict[int, float] = {}

        # track_id -> last known position
        self._last_positions: Dict[int, Tuple[float, float]] = {}

        # track_id -> list of per-frame distances
        self._frame_distances: Dict[int, List[float]] = {}

    @staticmethod
    def _euclidean_distance(
        pos_a: Tuple[float, float],
        pos_b: Tuple[float, float]
    ) -> float:
        """Computes Euclidean distance (meters) between two 2D coordinates."""
        dx = pos_b[0] - pos_a[0]
        dy = pos_b[1] - pos_a[1]
        return float(np.sqrt(dx * dx + dy * dy))

    def update(
        self,
        track_id: int,
        position_m: Tuple[float, float],
        speed_kmh: float = 0.0
    ) -> float:
        """
        Updates cumulative distance for a single player.

        Args:
            track_id: Player track ID.
            position_m: Current real-world position (x, y) in meters.
            speed_kmh: Player speed at this frame (km/h), used for sprint tracking.

        Returns:
            Frame displacement distance in meters.
        """
        if track_id not in self._last_positions:
            self._last_positions[track_id] = position_m
            self._total_distance[track_id] = 0.0
            self._sprint_distance[track_id] = 0.0
            self._frame_distances[track_id] = []
            return 0.0

        dist_m = self._euclidean_distance(self._last_positions[track_id], position_m)

        # Filter tracking artifacts: position jumps >5m are not real movement.
        # Maximum plausible player displacement between consecutive frames at 30fps is ~2.8m (40 km/h).
        # Larger jumps indicate ByteTrack ID switches or detection failures.
        if dist_m > 5.0:
            dist_m = 0.0
            logger.debug(
                "DistanceTracker: filtered %.2fm artifact for track %d",
                dist_m, track_id
            )

        self._total_distance[track_id] += dist_m
        self._frame_distances[track_id].append(dist_m)

        if speed_kmh >= SPRINT_SPEED_THRESHOLD_KMH:
            self._sprint_distance[track_id] += dist_m

        self._last_positions[track_id] = position_m
        return dist_m

    def update_batch(
        self,
        player_positions: Dict[int, Tuple[float, float]],
        player_speeds: Optional[Dict[int, float]] = None
    ) -> Dict[int, float]:
        """
        Batch distance update for multiple players in a frame.

        Args:
            player_positions: Dict mapping track_id to (x, y) position in meters.
            player_speeds: Optional dict of track_id to speed (km/h) for sprint classification.

        Returns:
            Dict of track_id -> frame displacement (meters).
        """
        results = {}
        for track_id, position in player_positions.items():
            speed = player_speeds.get(track_id, 0.0) if player_speeds else 0.0
            results[track_id] = self.update(track_id, position, speed_kmh=speed)
        return results

    def get_total_distance(self, track_id: int) -> float:
        """Returns cumulative distance covered (meters)."""
        return round(self._total_distance.get(track_id, 0.0), 2)

    def get_sprint_distance(self, track_id: int) -> float:
        """Returns total sprint distance (meters) above threshold."""
        return round(self._sprint_distance.get(track_id, 0.0), 2)

    def get_running_distance(self, track_id: int) -> float:
        """Returns non-sprint total distance (meters)."""
        total = self.get_total_distance(track_id)
        sprint = self.get_sprint_distance(track_id)
        return round(max(0.0, total - sprint), 2)

    def get_frame_distances(self, track_id: int) -> List[float]:
        """Returns per-frame distance list for a player."""
        return self._frame_distances.get(track_id, [])

    def get_all_totals(self) -> Dict[int, float]:
        """Returns cumulative distance for all tracked players."""
        return {tid: round(dist, 2) for tid, dist in self._total_distance.items()}

    def get_summary(self, track_id: int) -> Dict:
        """Returns full distance analytics summary for a player."""
        return {
            "track_id": track_id,
            "total_distance_m": self.get_total_distance(track_id),
            "sprint_distance_m": self.get_sprint_distance(track_id),
            "running_distance_m": self.get_running_distance(track_id),
            "frames_tracked": len(self._frame_distances.get(track_id, []))
        }

    def clear(self, track_id: Optional[int] = None) -> None:
        """Clears state for one or all players."""
        if track_id is not None:
            self._total_distance.pop(track_id, None)
            self._sprint_distance.pop(track_id, None)
            self._last_positions.pop(track_id, None)
            self._frame_distances.pop(track_id, None)
        else:
            self._total_distance.clear()
            self._sprint_distance.clear()
            self._last_positions.clear()
            self._frame_distances.clear()
