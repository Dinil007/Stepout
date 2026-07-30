from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Sequence

from app.analytics.pressing_engine import PressingAnalysisResult
from app.analytics.pressing_types import (
    PPDAWindow,
    PressingDetection,
    PressingMetrics,
    PressingSequence,
    PressureEvent,
)

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


class PressingValidator:
    """Validates outputs produced by the pressing analysis pipeline.

    This module performs no detection or metrics computation; it only
    verifies correctness and consistency.

    Attributes:
        config: Configuration parameters.
    """

    def __init__(self) -> None:
        pass

    def validate_pressure_event(self, event: PressureEvent) -> ValidationReport:
        """Validate a single pressure event.

        Args:
            event: PressureEvent instance.

        Returns:
            ValidationReport.
        """
        report = ValidationReport()
        if not event.is_valid():
            report.add_error("PressureEvent failed basic validation.")
        if not event.within_bounds():
            report.add_error("PressureEvent values are outside expected bounds.")
        if event.frame_number < 0:
            report.add_error(f"Invalid frame_number {event.frame_number}.")
        if event.distance < 0 or event.closing_speed < 0:
            report.add_error("Distance and closing_speed must be non-negative.")
        report.add_pass()
        return report

    def validate_sequence(self, sequence: PressingSequence) -> ValidationReport:
        """Validate a pressing sequence.

        Args:
            sequence: PressingSequence instance.

        Returns:
            ValidationReport.
        """
        report = ValidationReport()
        if not sequence.is_valid():
            report.add_error("PressingSequence failed basic validation.")
        if sequence.start_frame > sequence.end_frame:
            report.add_error("start_frame must be <= end_frame.")
        if sequence.duration_seconds < 0:
            report.add_error("duration_seconds must be non-negative.")
        if sequence.start_time > sequence.end_time:
            report.add_error("start_time must be <= end_time.")
        report.add_pass()
        return report

    def validate_ppda_window(self, window: PPDAWindow) -> ValidationReport:
        """Validate a PPDA window.

        Args:
            window: PPDAWindow instance.

        Returns:
            ValidationReport.
        """
        report = ValidationReport()
        if not window.is_valid():
            report.add_error("PPDAWindow failed basic validation.")
        if window.passes_allowed < 0:
            report.add_error("passes_allowed must be non-negative.")
        if window.defensive_actions < 0:
            report.add_error("defensive_actions must be non-negative.")
        if window.start_time > window.end_time:
            report.add_error("start_time must be <= end_time.")
        report.add_pass()
        return report

    def validate_metrics(self, metrics: PressingMetrics) -> ValidationReport:
        """Validate pressing metrics.

        Args:
            metrics: PressingMetrics instance.

        Returns:
            ValidationReport.
        """
        report = ValidationReport()
        if not metrics.is_valid():
            report.add_error("PressingMetrics failed basic validation.")
        if metrics.total_pressures < 0 or metrics.successful_pressures < 0:
            report.add_error("Pressure counts must be non-negative.")
        if metrics.successful_pressures > metrics.total_pressures:
            report.add_error("successful_pressures cannot exceed total_pressures.")
        if not (0.0 <= metrics.pressure_success_rate <= 1.0):
            report.add_error("pressure_success_rate must be in [0, 1].")
        if metrics.high_press_count < 0 or metrics.mid_block_count < 0 or metrics.low_block_count < 0:
            report.add_error("Zone counts must be non-negative.")
        report.add_pass()
        return report

    def validate_detection(self, detection: PressingDetection) -> ValidationReport:
        """Validate pressing detection.

        Args:
            detection: PressingDetection instance.

        Returns:
            ValidationReport.
        """
        report = ValidationReport()
        if not detection.is_valid():
            report.add_error("PressingDetection failed basic validation.")
        if detection.confidence < 0 or detection.confidence > 1:
            report.add_error("Confidence must be in [0, 1].")
        if detection.frame_number < 0:
            report.add_error(f"Invalid frame_number {detection.frame_number}.")
        report.add_pass()
        return report

    def validate_analysis(self, analysis: PressingAnalysisResult) -> ValidationReport:
        """Validate a complete analysis result.

        Args:
            analysis: PressingAnalysisResult instance.

        Returns:
            ValidationReport.
        """
        report = ValidationReport()
        if not analysis.metadata:
            report.add_warning("Analysis metadata is empty.")
        if analysis.processing_time_ms < 0:
            report.add_error("processing_time_ms must be non-negative.")
        if analysis.pressing_metrics:
            metrics_report = self.validate_metrics(analysis.pressing_metrics)
            report.checked_items += metrics_report.checked_items
            report.passed_items += metrics_report.passed_items
            report.errors.extend(metrics_report.errors)
            report.warnings.extend(metrics_report.warnings)
        if analysis.pressing_detection:
            detection_report = self.validate_detection(analysis.pressing_detection)
            report.checked_items += detection_report.checked_items
            report.passed_items += detection_report.passed_items
            report.errors.extend(detection_report.errors)
            report.warnings.extend(detection_report.warnings)
        for seq in analysis.pressing_sequences:
            seq_report = self.validate_sequence(seq)
            report.checked_items += seq_report.checked_items
            report.passed_items += seq_report.passed_items
            report.errors.extend(seq_report.errors)
            report.warnings.extend(seq_report.warnings)
        if report.errors:
            report.overall_valid = False
        report.add_pass()
        return report

    def validate_batch(self, analyses: Sequence[PressingAnalysisResult]) -> ValidationReport:
        """Validate a batch of analysis results.

        Args:
            analyses: Sequence of PressingAnalysisResult instances.

        Returns:
            ValidationReport.
        """
        report = ValidationReport()
        if not analyses:
            report.add_warning("Empty analysis batch.")
        for idx, analysis in enumerate(analyses):
            sub_report = self.validate_analysis(analysis)
            report.checked_items += sub_report.checked_items
            report.passed_items += sub_report.passed_items
            report.errors.extend(sub_report.errors)
            report.warnings.extend(sub_report.warnings)
        if report.errors:
            report.overall_valid = False
        report.add_pass()
        return report

    def reset(self) -> None:
        """Reset validator state."""
        logger.info("PressingValidator reset.")