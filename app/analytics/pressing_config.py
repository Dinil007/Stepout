from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.analytics.pressing_types import PressingZone


@dataclass
class PressingConfig:
    """Configuration parameters for pressing analysis.

    Attributes:
        pressure_distance_threshold: Maximum distance to consider a pressure active.
        high_press_line_y: Y coordinate threshold for high press zone.
        mid_block_line_y: Y coordinate threshold for mid-block zone.
        low_block_line_y: Y coordinate threshold for low block zone.
        minimum_pressure_duration: Minimum duration for a valid pressure sequence.
        minimum_closing_speed: Minimum closing speed to count as a pressure.
        ppda_window_seconds: Time window for PPDA calculation.
        confidence_threshold: Minimum confidence for pressing style detection.
        smoothing_window: Window size for smoothing pressure counts.
        enable_validation: Whether to enable input validation.
        enable_logging: Whether to enable logging.
    """

    pressure_distance_threshold: float = 5.0
    high_press_line_y: float = 0.35
    mid_block_line_y: float = 0.65
    low_block_line_y: float = 0.85
    minimum_pressure_duration: float = 1.5
    minimum_closing_speed: float = 0.5
    ppda_window_seconds: float = 5.0
    confidence_threshold: float = 0.7
    smoothing_window: int = 3
    enable_validation: bool = True
    enable_logging: bool = True

    def validate(self) -> list[str]:
        """Validate configuration values.

        Returns:
            List of validation error messages.
        """
        errors = []
        if self.pressure_distance_threshold <= 0:
            errors.append("pressure_distance_threshold must be positive.")
        if not (0.0 <= self.high_press_line_y <= 1.0):
            errors.append("high_press_line_y must be in [0, 1].")
        if not (0.0 <= self.mid_block_line_y <= 1.0):
            errors.append("mid_block_line_y must be in [0, 1].")
        if not (0.0 <= self.low_block_line_y <= 1.0):
            errors.append("low_block_line_y must be in [0, 1].")
        if self.high_press_line_y >= self.mid_block_line_y:
            errors.append("high_press_line_y must be less than mid_block_line_y.")
        if self.mid_block_line_y >= self.low_block_line_y:
            errors.append("mid_block_line_y must be less than low_block_line_y.")
        if self.minimum_pressure_duration <= 0:
            errors.append("minimum_pressure_duration must be positive.")
        if self.minimum_closing_speed < 0:
            errors.append("minimum_closing_speed must be non-negative.")
        if self.ppda_window_seconds <= 0:
            errors.append("ppda_window_seconds must be positive.")
        if not (0.0 <= self.confidence_threshold <= 1.0):
            errors.append("confidence_threshold must be in [0, 1].")
        if self.smoothing_window <= 0:
            errors.append("smoothing_window must be positive.")
        return errors

    def is_valid(self) -> bool:
        """Check if configuration is valid.

        Returns:
            True if valid, False otherwise.
        """
        return len(self.validate()) == 0

    def to_dict(self) -> dict[str, Any]:
        """Serialize configuration to dictionary.

        Returns:
            Dictionary representation of configuration.
        """
        return {
            "pressure_distance_threshold": self.pressure_distance_threshold,
            "high_press_line_y": self.high_press_line_y,
            "mid_block_line_y": self.mid_block_line_y,
            "low_block_line_y": self.low_block_line_y,
            "minimum_pressure_duration": self.minimum_pressure_duration,
            "minimum_closing_speed": self.minimum_closing_speed,
            "ppda_window_seconds": self.ppda_window_seconds,
            "confidence_threshold": self.confidence_threshold,
            "smoothing_window": self.smoothing_window,
            "enable_validation": self.enable_validation,
            "enable_logging": self.enable_logging,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PressingConfig:
        """Create configuration from dictionary.

        Args:
            data: Dictionary with configuration values.

        Returns:
            PressingConfig instance.
        """
        return cls(**data)

    def copy(self) -> PressingConfig:
        """Create a copy of the configuration.

        Returns:
            New PressingConfig with same values.
        """
        return PressingConfig(**self.to_dict())

    def update(self, **kwargs: Any) -> PressingConfig:
        """Update configuration with new values.

        Args:
            **kwargs: Configuration fields to update.

        Returns:
            Updated PressingConfig instance.
        """
        data = self.to_dict()
        data.update(kwargs)
        return PressingConfig.from_dict(data)