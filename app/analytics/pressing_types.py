from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from enum import Enum


class PressingZone(str, Enum):
    """Enumeration of pressing defensive zones.

    Attributes:
        HIGH_PRESS: High press in the opponent's defensive third.
        MID_BLOCK: Mid-block press in the middle third.
        LOW_BLOCK: Low block deep in defensive half.
    """

    HIGH_PRESS = "high_press"
    MID_BLOCK = "mid_block"
    LOW_BLOCK = "low_block"


@dataclass
class PressureEvent:
    """Represents one pressure action against an opponent.

    Attributes:
        attacker_id: ID of the pressing attacker.
        defender_id: ID of the defender being pressed.
        team_id: ID of the pressing team.
        frame_number: Frame number in the video sequence.
        timestamp: Timestamp of the pressure event.
        distance: Distance between attacker and defender at pressure moment.
        closing_speed: Speed at which attacker closed the distance.
        pressure_angle: Angle of approach relative to defender orientation (radians).
        successful: Whether the pressure resulted in a turnover or forced error.
    """

    attacker_id: int
    defender_id: int
    team_id: int
    frame_number: int
    timestamp: datetime
    distance: float
    closing_speed: float
    pressure_angle: float
    successful: bool

    def is_valid(self) -> bool:
        """Validate basic value ranges.

        Returns:
            True if fields are within reasonable bounds, False otherwise.
        """
        if self.frame_number < 0 or self.attacker_id < 0 or self.defender_id < 0 or self.team_id < 0:
            return False
        if self.distance < 0.0 or self.closing_speed < 0.0:
            return False
        return True

    def within_bounds(self, max_distance: float = 100.0, max_speed: float = 10.0) -> bool:
        """Check if event values are within expected bounds.

        Args:
            max_distance: Maximum expected pressure distance.
            max_speed: Maximum expected closing speed.

        Returns:
            True if within bounds, False otherwise.
        """
        return 0.0 <= self.distance <= max_distance and 0.0 <= self.closing_speed <= max_speed


@dataclass
class PressingSequence:
    """Represents a continuous sequence of pressure actions.

    Attributes:
        sequence_id: Unique identifier for the sequence.
        team_id: ID of the pressing team.
        start_frame: Starting frame number of the sequence.
        end_frame: Ending frame number of the sequence.
        start_time: Timestamp when the sequence started.
        end_time: Timestamp when the sequence ended.
        pressure_events: List of PressureEvent instances in the sequence.
        duration_seconds: Duration of the sequence in seconds.
    """

    sequence_id: int
    team_id: int
    start_frame: int
    end_frame: int
    start_time: datetime
    end_time: datetime
    pressure_events: list[PressureEvent] = field(default_factory=list)
    duration_seconds: float = 0.0

    def is_valid(self) -> bool:
        """Validate sequence bounds and event consistency.

        Returns:
            True if the sequence is valid, False otherwise.
        """
        if self.start_frame < 0 or self.end_frame < 0:
            return False
        if self.start_frame > self.end_frame:
            return False
        if self.duration_seconds < 0:
            return False
        if self.start_time > self.end_time:
            return False
        return True

    def event_count(self) -> int:
        """Return number of pressure events in the sequence.

        Returns:
            Count of pressure events.
        """
        return len(self.pressure_events)


@dataclass
class PPDAWindow:
    """Passes Per Defensive Action (PPDA) window.

    Attributes:
        team_id: ID of the team being measured.
        start_time: Start timestamp of the window.
        end_time: End timestamp of the window.
        passes_allowed: Number of opponent passes allowed.
        defensive_actions: Number of defensive actions taken.
        ppda: Passes per defensive action ratio.
    """

    team_id: int
    start_time: datetime
    end_time: datetime
    passes_allowed: int
    defensive_actions: int
    ppda: float

    def is_valid(self) -> bool:
        """Validate PPDA values.

        Returns:
            True if values are valid, False otherwise.
        """
        if self.passes_allowed < 0 or self.defensive_actions < 0:
            return False
        if self.ppda < 0.0:
            return False
        if self.start_time > self.end_time:
            return False
        return True


@dataclass
class PressingMetrics:
    """Aggregate pressing metrics for a team.

    Attributes:
        total_pressures: Total number of pressure events.
        successful_pressures: Number of successful pressure events.
        pressure_success_rate: Success rate as a fraction (0.0-1.0).
        average_pressure_time: Average duration of pressure sequences in seconds.
        average_closing_speed: Average closing speed of pressure attempts.
        ppda: Latest PPDA ratio.
        high_press_count: Number of high press events.
        mid_block_count: Number of mid-block events.
        low_block_count: Number of low block events.
    """

    total_pressures: int
    successful_pressures: int
    pressure_success_rate: float
    average_pressure_time: float
    average_closing_speed: float
    ppda: float
    high_press_count: int
    mid_block_count: int
    low_block_count: int

    def is_valid(self) -> bool:
        """Validate metric ranges and internal consistency.

        Returns:
            True if metrics are valid, False otherwise.
        """
        if self.total_pressures < 0 or self.successful_pressures < 0:
            return False
        if self.successful_pressures > self.total_pressures:
            return False
        if not (0.0 <= self.pressure_success_rate <= 1.0):
            return False
        if self.average_pressure_time < 0.0 or self.average_closing_speed < 0.0 or self.ppda < 0.0:
            return False
        if self.high_press_count < 0 or self.mid_block_count < 0 or self.low_block_count < 0:
            return False
        return True

    def pressure_distribution(self) -> dict[PressingZone, int]:
        """Return counts by pressing zone.

        Returns:
            Dictionary mapping PressingZone to event counts.
        """
        return {
            PressingZone.HIGH_PRESS: self.high_press_count,
            PressingZone.MID_BLOCK: self.mid_block_count,
            PressingZone.LOW_BLOCK: self.low_block_count,
        }


@dataclass
class PressingDetection:
    """Represents detected pressing style and confidence.

    Attributes:
        pressing_style: Predominant pressing zone.
        confidence: Confidence in the detection (0.0-1.0).
        frame_number: Frame number where the detection was made.
        timestamp: Timestamp of the detection.
    """

    pressing_style: PressingZone
    confidence: float
    frame_number: int
    timestamp: datetime

    def is_valid(self) -> bool:
        """Validate detection values.

        Returns:
            True if valid, False otherwise.
        """
        if self.frame_number < 0:
            return False
        if not (0.0 <= self.confidence <= 1.0):
            return False
        if not isinstance(self.pressing_style, PressingZone):
            return False
        return True