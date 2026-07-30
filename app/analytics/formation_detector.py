from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any

from app.analytics.formation_config import FormationConfig
from app.analytics.formation_templates import FormationTemplateRegistry
from app.analytics.formation_types import (
    FormationDetection,
    PlayerPosition,
)

logger = logging.getLogger(__name__)


@dataclass
class FormationDetector:
    """Core tactical formation detection engine.

    Compares tracked player positions against registered formation templates
    and returns the best matching formation with a confidence score.

    Attributes:
        config: Configuration parameters controlling detection behavior.
        registry: Registry of available formation templates.
    """

    config: FormationConfig | None = None
    registry: FormationTemplateRegistry | None = None

    def __post_init__(self) -> None:
        """Initialize defaults if not provided."""
        if self.config is None:
            from app.analytics.formation_config import FormationConfig

            self.config = FormationConfig()
        if self.registry is None:
            from app.analytics.formation_templates import default_registry

            self.registry = default_registry

    def validate_players(self, players: list[PlayerPosition]) -> None:
        """Validate the input player list.

        Args:
            players: List of PlayerPosition instances.

        Raises:
            ValueError: If validation fails.
        """
        if not players:
            raise ValueError("Player list is empty.")
        ids = [p.player_id for p in players]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate player IDs detected.")
        for player in players:
            if not player.is_valid():
                raise ValueError(
                    f"Invalid player position data for player_id={player.player_id}."
                )
            if not player.within_pitch_bounds():
                raise ValueError(
                    f"Player player_id={player.player_id} is outside pitch bounds."
                )

    def group_by_team(
        self, players: list[PlayerPosition]
    ) -> dict[int, list[PlayerPosition]]:
        """Group players by their team identifier.

        Args:
            players: List of PlayerPosition instances.

        Returns:
            Mapping from team_id to list of players.
        """
        teams: dict[int, list[PlayerPosition]] = {}
        for player in players:
            teams.setdefault(player.team_id, []).append(player)
        return teams

    def remove_goalkeeper(
        self, players: list[PlayerPosition]
    ) -> list[PlayerPosition]:
        """Remove goalkeeper(s) if configured to ignore them.

        Args:
            players: List of PlayerPosition instances.

        Returns:
            Filtered list of players.
        """
        if self.config.ignore_goalkeeper:
            return [p for p in players if not p.is_goalkeeper]
        return list(players)

    def normalize_positions(
        self, players: list[PlayerPosition]
    ) -> list[tuple[float, float]]:
        """Return normalized pitch coordinates for the given players.

        Assumes the input coordinates are already normalized to [0, 1].

        Args:
            players: List of PlayerPosition instances.

        Returns:
            List of (x, y) tuples.
        """
        return [(p.x, p.y) for p in players]

    def sort_positions(
        self, positions: list[tuple[float, float]]
    ) -> list[tuple[float, float]]:
        """Sort positions deterministically for stable matching.

        Args:
            positions: List of (x, y) tuples.

        Returns:
            Sorted list of positions.
        """
        return sorted(positions, key=lambda coord: (coord[1], coord[0]))

    def _best_subset(
        self,
        players: list[tuple[float, float]],
        template_positions: list[tuple[float, float]],
    ) -> list[tuple[float, float]]:
        """Select the subset of players closest to the template centroid.

        Args:
            players: Normalized player coordinates.
            template_positions: Template coordinates to match against.

        Returns:
            Subset of players with size equal to template_positions.
        """
        k = len(template_positions)
        cx = sum(x for x, y in template_positions) / k
        cy = sum(y for x, y in template_positions) / k
        sorted_players = sorted(
            players,
            key=lambda p: (p[0] - cx) ** 2 + (p[1] - cy) ** 2,
        )
        return sorted_players[:k]

    def compute_similarity(
        self,
        players: list[tuple[float, float]],
        template_positions: list[tuple[float, float]],
    ) -> float:
        """Compute a deterministic similarity score between players and template.

        Uses mean squared positional error mapped to a [0, 1] confidence score
        via exponential decay. Handles player counts greater than template size
        by selecting the subset closest to the template centroid.

        Args:
            players: Normalized player coordinates.
            template_positions: Template coordinates to match against.

        Returns:
            Similarity score in [0.0, 1.0].
        """
        if len(players) < len(template_positions):
            return 0.0
        if len(players) > len(template_positions):
            players = self._best_subset(players, template_positions)
            players = self.sort_positions(players)
        if not players:
            return 0.0
        total_error = 0.0
        for (px, py), (tx, ty) in zip(players, template_positions):
            dx = px - tx
            dy = py - ty
            total_error += dx * dx + dy * dy
        mse = total_error / len(template_positions)
        score = math.exp(-mse * 20.0)
        return float(score)

    def calculate_confidence(self, score: float) -> float:
        """Calculate detection confidence from the raw similarity score.

        Args:
            score: Raw similarity score in [0.0, 1.0].

        Returns:
            Clamped confidence score in [0.0, 1.0].
        """
        return max(0.0, min(1.0, score))

    def match_templates(
        self, sorted_positions: list[tuple[float, float]]
    ) -> tuple[str, float, float]:
        """Match sorted positions against registered templates.

        Args:
            sorted_positions: Sorted normalized player coordinates.

        Returns:
            Tuple of (best_template_name, best_score, confidence).
        """
        best_name = ""
        best_score = 0.0
        for name in self.registry.list_templates():
            template = self.registry.get_template(name)
            template_positions = self.sort_positions(
                list(template.normalized_positions)
            )
            score = self.compute_similarity(sorted_positions, template_positions)
            logger.debug("Template %s scored %.4f", name, score)
            if score > best_score:
                best_score = score
                best_name = name
        confidence = self.calculate_confidence(best_score)
        return best_name, best_score, confidence

    def detect_team(
        self,
        players: list[PlayerPosition],
        timestamp: Any = None,
        frame_number: int | None = None,
    ) -> FormationDetection:
        """Detect the formation for a single team.

        Args:
            players: List of PlayerPosition instances for one team.
            timestamp: Optional timestamp of the detection.
            frame_number: Optional frame number of the detection.

        Returns:
            FormationDetection describing the best matching formation.

        Raises:
            ValueError: If too few players remain after filtering.
        """
        players = self.remove_goalkeeper(players)
        if len(players) < self.config.minimum_tracked_players:
            raise ValueError(
                f"Not enough players ({len(players)}) for formation detection."
            )
        positions = self.normalize_positions(players)
        sorted_positions = self.sort_positions(positions)
        matched_template, score, confidence = self.match_templates(sorted_positions)
        if not matched_template:
            raise ValueError("No matching formation template found.")
        from datetime import datetime, timezone

        detection = FormationDetection(
            detected_formation=matched_template,
            confidence=confidence,
            frame_number=frame_number if frame_number is not None else 0,
            timestamp=timestamp if timestamp is not None else datetime.now(timezone.utc),
            matched_template=matched_template,
            score=score,
        )
        logger.info(
            "Detected formation %s with confidence %.2f",
            matched_template,
            confidence,
        )
        return detection

    def detect_all_teams(
        self,
        players: list[PlayerPosition],
        timestamp: Any = None,
        frame_number: int | None = None,
    ) -> dict[int, FormationDetection]:
        """Detect formations for all teams in the player list.

        Args:
            players: List of PlayerPosition instances from all teams.
            timestamp: Optional timestamp of the detection.
            frame_number: Optional frame number of the detection.

        Returns:
            Mapping from team_id to FormationDetection.
        """
        self.validate_players(players)
        teams = self.group_by_team(players)
        detections: dict[int, FormationDetection] = {}
        for team_id, team_players in teams.items():
            logger.info("Detecting formation for team_id=%s", team_id)
            try:
                detection = self.detect_team(team_players, timestamp, frame_number)
                detections[team_id] = detection
            except ValueError as exc:
                logger.warning("Skipping team_id=%s: %s", team_id, exc)
        return detections

    def detect(
        self,
        players: list[PlayerPosition],
        timestamp: Any = None,
        frame_number: int | None = None,
    ) -> FormationDetection:
        """Run formation detection across all teams and return the primary result.

        Selects the team with the highest detection confidence.

        Args:
            players: List of PlayerPosition instances from all teams.
            timestamp: Optional timestamp of the detection.
            frame_number: Optional frame number of the detection.

        Returns:
            FormationDetection for the team with highest confidence.

        Raises:
            ValueError: If no team produces a valid detection.
        """
        detections = self.detect_all_teams(players, timestamp, frame_number)
        if not detections:
            raise ValueError("No valid formation detections were produced.")
        best_team = max(detections, key=lambda tid: detections[tid].confidence)
        return detections[best_team]