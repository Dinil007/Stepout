from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Sequence

from app.analytics.pressing_config import PressingConfig
from app.analytics.pressing_detector import PressingDetector
from app.analytics.pressing_metrics import PressingMetricsEngine
from app.analytics.pressing_types import (
    PPDAWindow,
    PressingDetection,
    PressingMetrics,
    PressingSequence,
    PressureEvent,
)

logger = logging.getLogger(__name__)


@dataclass
class PressingAnalysisResult:
    """Structured result of pressing analysis.

    Attributes:
        pressure_events: Detected pressure events.
        pressing_sequences: Detected pressing sequences.
        pressing_detection: Detected pressing style.
        pressing_metrics: Computed pressing metrics.
        metadata: Additional metadata.
        processing_time_ms: Time taken for analysis.
    """

    pressure_events: list[PressureEvent] = field(default_factory=list)
    pressing_sequences: list[PressingSequence] = field(default_factory=list)
    pressing_detection: PressingDetection | None = None
    pressing_metrics: PressingMetrics | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    processing_time_ms: float = 0.0


class PressingEngine:
    """Orchestration engine for pressing analysis.

    Coordinates pressure detection and metrics computation.

    Attributes:
        config: Configuration parameters.
        detector: Pressure detection engine.
        metrics_engine: Metrics computation engine.
    """

    def __init__(self, config: PressingConfig | None = None) -> None:
        self.config = config if config is not None else PressingConfig()
        self.detector = PressingDetector(config=self.config)
        self.metrics_engine = PressingMetricsEngine(config=self.config)
        self._results: list[PressingAnalysisResult] = []

    def reset(self) -> None:
        """Clear cached batch results."""
        self._results.clear()
        logger.info("PressingEngine results cleared.")

    def summary(self) -> dict[str, Any]:
        """Return a lightweight summary of cached batch results.

        Returns:
            Dictionary with count and aggregate statistics.
        """
        if not self._results:
            return {"count": 0, "total_pressures": 0}
        total_pressures = 0
        for result in self._results:
            if result.pressing_metrics:
                total_pressures += result.pressing_metrics.total_pressures
        return {
            "count": len(self._results),
            "total_pressures": total_pressures,
        }

    def analyze(
        self,
        attackers: Sequence[tuple[float, float, float, float]],
        defenders: Sequence[tuple[float, float, float, float]],
        frame_number: int = 0,
        timestamp: float = 0.0,
    ) -> PressingAnalysisResult:
        """Run full pressing analysis for a single frame.

        Args:
            attackers: Sequence of (x, y, vx, vy) for attacking players.
            defenders: Sequence of (x, y, vx, vy) for defending players.
            frame_number: Current frame number.
            timestamp: Current timestamp in seconds.

        Returns:
            PressingAnalysisResult with detection and metrics.
        """
        start = time.perf_counter()
        warnings: list[str] = []
        errors: list[str] = []

        logger.info("Pressing analysis started for frame=%s", frame_number)

        events: list[PressureEvent] = []
        sequences: list[PressingSequence] = []
        detection: PressingDetection | None = None
        metrics: PressingMetrics | None = None

        try:
            events = self.detector.detect_pressure_events(
                attackers, defenders, frame_number, timestamp
            )
            logger.info("Detected %d pressure events.", len(events))
        except Exception as exc:
            logger.exception("Pressure detection failed: %s", exc)
            errors.append(f"Detection failed: {exc}")

        if events:
            try:
                events_by_frame: dict[int, list[PressureEvent]] = {}
                for event in events:
                    events_by_frame.setdefault(event.frame_number, []).append(event)
                sequences = self.detector.detect_pressing_sequences(events_by_frame)
                logger.info("Detected %d pressing sequences.", len(sequences))
            except Exception as exc:
                logger.exception("Sequence detection failed: %s", exc)
                errors.append(f"Sequence detection failed: {exc}")

            try:
                confidence = self.detector.calculate_confidence(events, sequences)
                pressing_style = self.detector.classify_pressing_zone(
                    np.mean([e.distance for e in events]) if events else 5.0
                )
                from datetime import datetime, timezone
                detection = PressingDetection(
                    pressing_style=pressing_style,
                    confidence=confidence,
                    frame_number=frame_number,
                    timestamp=datetime.now(timezone.utc),
                )
            except Exception as exc:
                logger.exception("Detection classification failed: %s", exc)
                errors.append(f"Detection classification failed: {exc}")

            try:
                ppda_window = PPDAWindow(
                    team_id=0,
                    start_time=time.time(),
                    end_time=time.time(),
                    passes_allowed=0,
                    defensive_actions=len(events),
                    ppda=0.0,
                )
                metrics = self.metrics_engine.calculate_metrics(events, sequences, ppda_window)
                logger.info("Metrics computation completed.")
            except Exception as exc:
                logger.exception("Metrics computation failed: %s", exc)
                warnings.append(f"Metrics computation failed: {exc}")

        duration_ms = (time.perf_counter() - start) * 1000.0
        result = PressingAnalysisResult(
            pressure_events=events,
            pressing_sequences=sequences,
            pressing_detection=detection,
            pressing_metrics=metrics,
            metadata={"frame_number": frame_number, "timestamp": timestamp, "warnings": warnings, "errors": errors},
            processing_time_ms=duration_ms,
        )
        self._results.append(result)
        logger.info("Pressing analysis completed in %.2fms", duration_ms)
        return result

    def analyze_team(
        self,
        team_attackers: Sequence[tuple[float, float, float, float]],
        team_defenders: Sequence[tuple[float, float, float, float]],
        frame_number: int = 0,
        timestamp: float = 0.0,
    ) -> PressingAnalysisResult:
        """Analyze pressing for a specific team.

        Args:
            team_attackers: Attacking players for the team.
            team_defenders: Defending players for the team.
            frame_number: Current frame number.
            timestamp: Current timestamp in seconds.

        Returns:
            PressingAnalysisResult for the specified team.
        """
        if not team_attackers or not team_defenders:
            logger.warning("Empty player sequences for frame %s.", frame_number)
        return self.analyze(team_attackers, team_defenders, frame_number, timestamp)

    def analyze_match(
        self,
        frames: Sequence[tuple[Sequence[tuple[float, float, float, float]], Sequence[tuple[float, float, float, float]]]],
        frame_numbers: Sequence[int] | None = None,
        timestamps: Sequence[float] | None = None,
    ) -> dict[int, PressingAnalysisResult]:
        """Analyze pressing for multiple frames.

        Args:
            frames: Sequence of (attackers, defenders) tuples per frame.
            frame_numbers: Optional frame numbers.
            timestamps: Optional timestamps.

        Returns:
            Mapping from frame number to PressingAnalysisResult.
        """
        results: dict[int, PressingAnalysisResult] = {}
        for idx, (attackers, defenders) in enumerate(frames):
            fn = frame_numbers[idx] if frame_numbers and idx < len(frame_numbers) else idx
            ts = timestamps[idx] if timestamps and idx < len(timestamps) else float(idx)
            try:
                result = self.analyze(attackers, defenders, frame_number=fn, timestamp=ts)
                results[fn] = result
            except Exception as exc:
                logger.warning("Frame %s analysis failed: %s", fn, exc)
        return results

    def batch_analyze(
        self,
        frames: Sequence[tuple[Sequence[tuple[float, float, float, float]], Sequence[tuple[float, float, float, float]]]],
        frame_numbers: Sequence[int] | None = None,
    ) -> list[PressingAnalysisResult]:
        """Analyze multiple frames in batch.

        Args:
            frames: Sequence of (attackers, defenders) tuples per frame.
            frame_numbers: Optional frame numbers.

        Returns:
            List of PressingAnalysisResult instances.
        """
        results: list[PressingAnalysisResult] = []
        for idx, (attackers, defenders) in enumerate(frames):
            fn = frame_numbers[idx] if frame_numbers and idx < len(frame_numbers) else idx
            try:
                result = self.analyze(attackers, defenders, frame_number=fn)
                results.append(result)
            except Exception as exc:
                logger.warning("Frame %s analysis failed: %s", fn, exc)
        return results