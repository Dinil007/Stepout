from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from app.analytics.pressing_config import PressingConfig
from app.analytics.pressing_types import (
    PPDAWindow,
    PressingMetrics,
    PressingSequence,
    PressureEvent,
    PressingZone,
)

logger = logging.getLogger(__name__)


@dataclass
class PressingMetricsEngine:
    """Computes team pressing KPIs from detected pressure events and sequences.

    This module only computes aggregate metrics.
    It does not detect pressures, generate visualizations, or expose APIs.

    Attributes:
        config: Configuration parameters for pressing analysis.
    """

    config: PressingConfig

    def calculate_metrics(
        self,
        events: Sequence[PressureEvent],
        sequences: Sequence[PressingSequence],
        ppda_window: PPDAWindow | None = None,
    ) -> PressingMetrics:
        """Compute full pressing metrics for a team.

        Args:
            events: Detected pressure events.
            sequences: Detected pressing sequences.
            ppda_window: Optional PPDA window data.

        Returns:
            PressingMetrics instance.
        """
        total = len(events)
        successful = sum(1 for e in events if e.successful)
        success_rate = self.calculate_pressure_success_rate(events)
        avg_time = self.calculate_average_pressure_time(sequences)
        avg_speed = self.calculate_average_closing_speed(events)
        ppda = self.calculate_ppda(ppda_window) if ppda_window else 0.0
        zone_counts = self.calculate_zone_counts(events)
        return PressingMetrics(
            total_pressures=total,
            successful_pressures=successful,
            pressure_success_rate=success_rate,
            average_pressure_time=avg_time,
            average_closing_speed=avg_speed,
            ppda=ppda,
            high_press_count=zone_counts.get(PressingZone.HIGH_PRESS, 0),
            mid_block_count=zone_counts.get(PressingZone.MID_BLOCK, 0),
            low_block_count=zone_counts.get(PressingZone.LOW_BLOCK, 0),
        )

    def calculate_ppda(self, window: PPDAWindow) -> float:
        """Calculate PPDA ratio.

        Args:
            window: PPDA window data.

        Returns:
            Passes per defensive action ratio.
        """
        if window.defensive_actions <= 0:
            return 0.0
        return window.passes_allowed / window.defensive_actions

    def calculate_pressure_success_rate(self, events: Sequence[PressureEvent]) -> float:
        """Calculate pressure success rate.

        Args:
            events: Sequence of pressure events.

        Returns:
            Success rate between 0.0 and 1.0.
        """
        if not events:
            return 0.0
        successful = sum(1 for e in events if e.successful)
        return successful / len(events)

    def calculate_average_closing_speed(self, events: Sequence[PressureEvent]) -> float:
        """Calculate average closing speed.

        Args:
            events: Sequence of pressure events.

        Returns:
            Average closing speed.
        """
        if not events:
            return 0.0
        speeds = np.array([e.closing_speed for e in events], dtype=float)
        return float(np.mean(speeds))

    def calculate_average_pressure_time(self, sequences: Sequence[PressingSequence]) -> float:
        """Calculate average pressure sequence duration.

        Args:
            sequences: Sequence of pressing sequences.

        Returns:
            Average duration in seconds.
        """
        if not sequences:
            return 0.0
        durations = np.array([s.duration_seconds for s in sequences], dtype=float)
        return float(np.mean(durations))

    def calculate_zone_counts(self, events: Sequence[PressureEvent]) -> dict[PressingZone, int]:
        """Count events by pressing zone.

        Args:
            events: Sequence of pressure events.

        Returns:
            Dictionary mapping PressingZone to event counts.
        """
        counts: dict[PressingZone, int] = {
            PressingZone.HIGH_PRESS: 0,
            PressingZone.MID_BLOCK: 0,
            PressingZone.LOW_BLOCK: 0,
        }
        for event in events:
            zone = self.classify_pressing_zone(event.distance)
            counts[zone] += 1
        return counts

    def classify_pressing_zone(self, distance: float) -> PressingZone:
        """Classify pressing zone based on pressure distance.

        Args:
            distance: Distance of the pressure event.

        Returns:
            PressingZone enum value.
        """
        if distance <= 2.0:
            return PressingZone.HIGH_PRESS
        if distance <= 4.0:
            return PressingZone.MID_BLOCK
        return PressingZone.LOW_BLOCK

    def calculate_team_metrics(
        self,
        events: Sequence[PressureEvent],
        sequences: Sequence[PressingSequence],
        ppda_window: PPDAWindow | None = None,
    ) -> PressingMetrics:
        """Compute team-level pressing metrics.

        Args:
            events: Sequence of pressure events.
            sequences: Sequence of pressing sequences.
            ppda_window: Optional PPDA window.

        Returns:
            PressingMetrics instance.
        """
        return self.calculate_metrics(events, sequences, ppda_window)

    def reset(self) -> None:
        """Reset internal engine state."""
        logger.info("PressingMetricsEngine reset.")