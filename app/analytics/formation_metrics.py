from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from app.analytics.formation_config import FormationConfig
from app.analytics.formation_types import FormationDetection, FormationMetrics, PlayerPosition

logger = logging.getLogger(__name__)


class FormationMetricsEngine:
    """Calculates tactical shape metrics from tracked player positions.

    This engine is responsible only for metric computation and does not
    perform formation detection or visualization.

    Attributes:
        config: Optional configuration controlling thresholds and behavior.
    """

    def __init__(self, config: FormationConfig | None = None) -> None:
        self.config = config if config is not None else FormationConfig()

    def validate_players(self, players: Sequence[PlayerPosition]) -> None:
        """Validate input player positions.

        Args:
            players: Sequence of PlayerPosition instances.

        Raises:
            ValueError: If validation fails.
        """
        if not players:
            raise ValueError("Player list is empty.")
        valid_count = 0
        for player in players:
            if not player.is_valid() or not player.within_pitch_bounds():
                logger.warning(
                    "Skipping invalid player player_id=%s during metrics calculation.",
                    player.player_id,
                )
            else:
                valid_count += 1
        if valid_count < 2:
            raise ValueError("Too few valid players for metrics calculation.")

    def _to_arrays(
        self, players: Sequence[PlayerPosition]
    ) -> tuple[np.ndarray, np.ndarray]:
        """Extract normalized x and y coordinates as NumPy arrays.

        Args:
            players: Sequence of PlayerPosition instances.

        Returns:
            Tuple of (x_array, y_array) as float64 NumPy arrays.
        """
        xs = np.array([p.x for p in players], dtype=np.float64)
        ys = np.array([p.y for p in players], dtype=np.float64)
        return xs, ys

    def calculate_width(self, xs: np.ndarray) -> float:
        """Calculate team width as the lateral spread of players.

        Args:
            xs: Array of normalized x coordinates.

        Returns:
            Width in normalized units [0.0, 1.0].
        """
        return float(np.ptp(xs))

    def calculate_length(self, ys: np.ndarray) -> float:
        """Calculate team length as the longitudinal spread of players.

        Args:
            ys: Array of normalized y coordinates.

        Returns:
            Length in normalized units [0.0, 1.0].
        """
        return float(np.ptp(ys))

    def calculate_centroid(self, xs: np.ndarray, ys: np.ndarray) -> tuple[float, float]:
        """Calculate team centroid as the mean position.

        Args:
            xs: Array of normalized x coordinates.
            ys: Array of normalized y coordinates.

        Returns:
            Tuple of (centroid_x, centroid_y).
        """
        return (float(np.mean(xs)), float(np.mean(ys)))

    def calculate_compactness(
        self, xs: np.ndarray, ys: np.ndarray, width: float, length: float
    ) -> float:
        """Calculate compactness as the ratio of convex hull area to bounding box area.

        A value close to 1.0 indicates a very compact shape; values near 0.0
        indicate a dispersed shape.

        Args:
            xs: Array of normalized x coordinates.
            ys: Array of normalized y coordinates.
            width: Team width.
            length: Team length.

        Returns:
            Compactness score in [0.0, 1.0].
        """
        area = self.calculate_convex_hull(xs, ys)
        box_area = width * length
        if box_area <= 0:
            return 0.0
        return float(np.clip(area / box_area, 0.0, 1.0))

    def calculate_convex_hull(self, xs: np.ndarray, ys: np.ndarray) -> float:
        """Calculate the area of the convex hull enclosing the players.

        Uses a monotone chain algorithm. Returns 0.0 if fewer than 3 players.

        Args:
            xs: Array of normalized x coordinates.
            ys: Array of normalized y coordinates.

        Returns:
            Convex hull area in normalized units squared.
        """
        n = len(xs)
        if n < 3:
            return 0.0
        points = np.column_stack((xs, ys))
        # Sort by x then y
        points = points[np.lexsort((points[:, 1], points[:, 0]))]
        # Build lower hull
        lower = []
        for p in points:
            while len(lower) >= 2 and self._cross(lower[-2], lower[-1], p) <= 0:
                lower.pop()
            lower.append(p)
        # Build upper hull
        upper = []
        for p in points[::-1]:
            while len(upper) >= 2 and self._cross(upper[-2], upper[-1], p) <= 0:
                upper.pop()
            upper.append(p)
        # Concatenate, removing duplicate endpoints
        hull = np.array(lower[:-1] + upper[:-1])
        if len(hull) < 3:
            return 0.0
        # Shoelace formula
        x = hull[:, 0]
        y = hull[:, 1]
        return float(0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))))

    def _cross(self, o: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
        """Compute 2D cross product of OA and OB vectors.

        Args:
            o: Origin point.
            a: First point.
            b: Second point.

        Returns:
            Cross product value.
        """
        return float((a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]))

    def calculate_line_heights(
        self, ys: np.ndarray
    ) -> tuple[float, float, float]:
        """Calculate defensive, midfield, and forward line heights.

        Players are grouped by y-coordinate tertiles.

        Args:
            ys: Array of normalized y coordinates.

        Returns:
            Tuple of (defensive_line, midfield_line, forward_line).
        """
        n = len(ys)
        if n == 0:
            return (0.0, 0.0, 0.0)
        sorted_ys = np.sort(ys)
        q1 = float(sorted_ys[int(n * 0.33)])
        q2 = float(sorted_ys[int(n * 0.66)])
        defensive = float(np.mean(ys[ys <= q1])) if np.any(ys <= q1) else q1
        midfield = (
            float(np.mean(ys[(ys > q1) & (ys <= q2)]))
            if np.any((ys > q1) & (ys <= q2))
            else q2
        )
        forward = float(np.mean(ys[ys > q2])) if np.any(ys > q2) else q2
        return defensive, midfield, forward

    def calculate_interplayer_distance(
        self, xs: np.ndarray, ys: np.ndarray
    ) -> float:
        """Calculate mean Euclidean distance between all player pairs.

        Args:
            xs: Array of normalized x coordinates.
            ys: Array of normalized y coordinates.

        Returns:
            Average inter-player distance.
        """
        n = len(xs)
        if n < 2:
            return 0.0
        points = np.column_stack((xs, ys))
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        # Exclude diagonal (zero distances)
        i, j = np.triu_indices(n, k=1)
        return float(np.mean(dists[i, j]))

    def calculate_density(
        self, player_count: int, width: float, length: float
    ) -> float:
        """Calculate team density as players per normalized pitch area.

        Args:
            player_count: Number of players.
            width: Team width.
            length: Team length.

        Returns:
            Density value.
        """
        area = width * length
        if area <= 0:
            return 0.0
        return float(player_count / area)

    def compute_metrics(
        self,
        players: Sequence[PlayerPosition],
        detection: FormationDetection | None = None,
    ) -> FormationMetrics:
        """Compute all tactical metrics for the given player positions.

        Args:
            players: Sequence of PlayerPosition instances.
            detection: Optional FormationDetection for context.

        Returns:
            Populated FormationMetrics dataclass.

        Raises:
            ValueError: If metrics cannot be computed.
        """
        self.validate_players(players)
        xs, ys = self._to_arrays(players)

        width = self.calculate_width(xs)
        length = self.calculate_length(ys)
        centroid_x, centroid_y = self.calculate_centroid(xs, ys)
        compactness = self.calculate_compactness(xs, ys, width, length)
        convex_hull_area = self.calculate_convex_hull(xs, ys)
        (
            defensive_line,
            midfield_line,
            forward_line,
        ) = self.calculate_line_heights(ys)
        vertical_stretch = length
        horizontal_stretch = width
        interplayer_distance = self.calculate_interplayer_distance(xs, ys)
        density = self.calculate_density(len(players), width, length)

        logger.info(
            "Computed metrics: width=%.3f length=%.3f compactness=%.3f hull=%.3f",
            width,
            length,
            compactness,
            convex_hull_area,
        )

        return FormationMetrics(
            team_width=width,
            team_length=length,
            compactness=compactness,
            centroid_x=centroid_x,
            centroid_y=centroid_y,
            convex_hull_area=convex_hull_area,
            defensive_line=defensive_line,
            midfield_line=midfield_line,
            forward_line=forward_line,
            vertical_stretch=vertical_stretch,
            horizontal_stretch=horizontal_stretch,
        )