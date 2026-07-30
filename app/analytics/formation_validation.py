from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Sequence

from app.analytics.formation_templates import FormationTemplateRegistry
from app.analytics.formation_types import (
    FormationDetection,
    FormationMetrics,
    PlayerPosition,
)
from app.analytics.formation_engine import FormationAnalysisResult

logger = logging.getLogger(__name__)


@dataclass
class ValidationReport:
    """Lightweight validation report.

    Attributes:
        overall_valid: True if no errors were found.
        errors: List of error messages.
        warnings: List of warning messages.
        checked_items: Total number of checks performed.
        passed_items: Number of checks that passed.
    """

    overall_valid: bool = True
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    checked_items: int = 0
    passed_items: int = 0

    def add_error(self, message: str) -> None:
        """Record an error.

        Args:
            message: Description of the validation failure.
        """
        self.overall_valid = False
        self.errors.append(message)
        self.checked_items += 1
        logger.error("Validation error: %s", message)

    def add_warning(self, message: str) -> None:
        """Record a warning.

        Args:
            message: Description of the validation concern.
        """
        self.warnings.append(message)
        self.checked_items += 1
        logger.warning("Validation warning: %s", message)

    def add_pass(self) -> None:
        """Record a passed check."""
        self.checked_items += 1
        self.passed_items += 1


class FormationValidator:
    """Validates outputs produced by the formation analysis pipeline.

    This module performs no detection or metrics computation; it only
    verifies correctness and consistency.

    Attributes:
        registry: Registry of available formation templates.
    """

    def __init__(self, registry: FormationTemplateRegistry | None = None) -> None:
        self.registry = registry if registry is not None else FormationTemplateRegistry()

    def validate_player(self, player: PlayerPosition, index: int | None = None) -> list[str]:
        """Validate a single player position.

        Args:
            player: Player position to validate.
            index: Optional index for diagnostic messages.

        Returns:
            List of error messages. Empty if valid.
        """
        errors: list[str] = []
        label = f"player[{index}]" if index is not None else "player"
        if not player.is_valid():
            errors.append(f"{label} has invalid position data.")
        if not player.within_pitch_bounds():
            errors.append(f"{label} is outside pitch bounds.")
        return errors

    def validate_players(self, players: Sequence[PlayerPosition]) -> ValidationReport:
        """Validate a collection of player positions.

        Args:
            players: Sequence of PlayerPosition instances.

        Returns:
            ValidationReport.
        """
        report = ValidationReport()
        if not players:
            report.add_error("Player list is empty.")
            return report
        seen_ids: dict[int, int] = {}
        team_ids: set[int] = set()
        for idx, player in enumerate(players):
            for err in self.validate_player(player, idx):
                report.add_error(err)
            if player.player_id in seen_ids:
                report.add_error(f"Duplicate player ID {player.player_id}.")
            else:
                seen_ids[player.player_id] = idx
            team_ids.add(player.team_id)
        if not team_ids:
            report.add_error("No valid team IDs found.")
        report.add_pass()  # player count presence
        return report

    def validate_detection(self, detection: FormationDetection) -> ValidationReport:
        """Validate a FormationDetection.

        Args:
            detection: Detection result to validate.

        Returns:
            ValidationReport.
        """
        report = ValidationReport()
        if not detection.detected_formation:
            report.add_error("Detected formation name is empty.")
        if not (0.0 <= detection.confidence <= 1.0):
            report.add_error(f"Detection confidence {detection.confidence} is out of range.")
        if not detection.matched_template:
            report.add_error("Matched template name is empty.")
        else:
            if not self.registry.template_exists(detection.matched_template):
                report.add_error(
                    f"Matched template '{detection.matched_template}' not found in registry."
                )
        if detection.frame_number < 0:
            report.add_error(f"Invalid frame number {detection.frame_number}.")
        if detection.timestamp is None:
            report.add_warning("Detection timestamp is missing.")
        else:
            if not isinstance(detection.timestamp, datetime):
                report.add_error("Detection timestamp has invalid type.")
        report.add_pass()
        return report

    def validate_metrics(self, metrics: FormationMetrics) -> ValidationReport:
        """Validate FormationMetrics.

        Args:
            metrics: Metrics to validate.

        Returns:
            ValidationReport.
        """
        report = ValidationReport()
        if metrics.team_width < 0:
            report.add_error(f"team_width is negative: {metrics.team_width}.")
        if metrics.team_length < 0:
            report.add_error(f"team_length is negative: {metrics.team_length}.")
        if metrics.compactness < 0:
            report.add_error(f"compactness is negative: {metrics.compactness}.")
        if metrics.convex_hull_area < 0:
            report.add_error(f"convex_hull_area is negative: {metrics.convex_hull_area}.")
        if metrics.vertical_stretch < 0 or metrics.horizontal_stretch < 0:
            report.add_error("Stretch metrics contain negative values.")
        if not (0.0 <= metrics.centroid_x <= 1.0 and 0.0 <= metrics.centroid_y <= 1.0):
            report.add_error(
                f"Centroid ({metrics.centroid_x}, {metrics.centroid_y}) is out of bounds."
            )
        if metrics.defensive_line < 0 or metrics.midfield_line < 0 or metrics.forward_line < 0:
            report.add_error("One or more line heights are negative.")
        report.add_pass()
        return report

    def validate_analysis(self, result: FormationAnalysisResult) -> ValidationReport:
        """Validate a complete analysis result from FormationEngine.

        Args:
            result: FormationAnalysisResult to validate.

        Returns:
            ValidationReport.
        """
        report = ValidationReport()
        if not result.detected_formation:
            report.add_error("Result missing detected_formation.")
        if not (0.0 <= result.confidence <= 1.0):
            report.add_error(f"Result confidence {result.confidence} is out of range.")
        if result.frame_number < 0:
            report.add_error(f"Result frame_number {result.frame_number} is invalid.")
        if result.timestamp is None:
            report.add_warning("Result timestamp is missing.")
        if result.analysis_duration_seconds < 0:
            report.add_error("analysis_duration_seconds is negative.")
        metrics_report = self.validate_metrics(result.metrics)
        report.checked_items += metrics_report.checked_items
        report.passed_items += metrics_report.passed_items
        report.errors.extend(metrics_report.errors)
        report.warnings.extend(metrics_report.warnings)
        if report.errors:
            report.overall_valid = False
        report.add_pass()
        return report