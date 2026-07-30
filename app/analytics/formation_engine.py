from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Sequence

from app.analytics.formation_config import FormationConfig
from app.analytics.formation_detector import FormationDetector
from app.analytics.formation_metrics import FormationMetricsEngine
from app.analytics.formation_types import (
    FormationDetection,
    FormationMetrics,
    PlayerPosition,
)

logger = logging.getLogger(__name__)


@dataclass
class FormationAnalysisResult:
    """Lightweight result object combining detection and metrics.

    Attributes:
        team_id: Identifier of the analyzed team.
        detected_formation: Detected formation name.
        confidence: Detection confidence score.
        metrics: Calculated tactical metrics.
        frame_number: Frame number analyzed.
        timestamp: Timestamp of the analysis.
        analysis_duration_seconds: Time taken for analysis.
        warnings: List of warning messages.
        errors: List of error messages.
    """

    team_id: int
    detected_formation: str
    confidence: float
    metrics: FormationMetrics
    frame_number: int = 0
    timestamp: datetime | None = None
    analysis_duration_seconds: float = 0.0
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


class FormationEngine:
    """Orchestration engine for tactical shape analysis.

    Coordinates formation detection and metrics calculation without
    reimplementing either concern.

    Attributes:
        config: Configuration parameters.
        detector: Formation detection engine.
        metrics_engine: Metrics computation engine.
    """

    def __init__(self, config: FormationConfig | None = None) -> None:
        self.config = config if config is not None else FormationConfig()
        self.detector = FormationDetector(config=self.config)
        self.metrics_engine = FormationMetricsEngine(config=self.config)
        self._results: list[FormationAnalysisResult] = []

    def reset(self) -> None:
        """Clear cached batch results."""
        self._results.clear()
        logger.info("FormationEngine results cleared.")

    def summary(self) -> dict[str, Any]:
        """Return a lightweight summary of cached batch results.

        Returns:
            Dictionary with count and aggregate statistics.
        """
        if not self._results:
            return {"count": 0, "formations": {}}
        formations: dict[str, int] = {}
        for result in self._results:
            formations[result.detected_formation] = (
                formations.get(result.detected_formation, 0) + 1
            )
        return {
            "count": len(self._results),
            "formations": formations,
        }

    def analyze(
        self,
        players: Sequence[PlayerPosition],
        frame_number: int = 0,
        timestamp: datetime | None = None,
        team_id: int | None = None,
    ) -> FormationAnalysisResult:
        """Run full tactical analysis on a single team's players.

        Args:
            players: Sequence of PlayerPosition instances.
            frame_number: Frame number being analyzed.
            timestamp: Timestamp of the analysis.
            team_id: Optional team identifier override.

        Returns:
            FormationAnalysisResult with detection and metrics.

        Raises:
            ValueError: If analysis cannot proceed.
        """
        start = time.perf_counter()
        ts = timestamp if timestamp is not None else datetime.now(timezone.utc)
        warnings: list[str] = []
        errors: list[str] = []

        logger.info("Analysis started for frame=%s", frame_number)

        try:
            detection = self.detector.detect(
                list(players), timestamp=ts, frame_number=frame_number
            )
            logger.info(
                "Detection completed: %s (confidence=%.2f)",
                detection.detected_formation,
                detection.confidence,
            )
            resolved_team_id = team_id if team_id is not None else (
                players[0].team_id if players else 0
            )
        except Exception as exc:
            logger.exception("Detection failed: %s", exc)
            errors.append(f"Detection failed: {exc}")
            raise

        try:
            metrics = self.metrics_engine.compute_metrics(
                list(players), detection=detection
            )
            logger.info("Metrics computation completed.")
        except Exception as exc:
            logger.warning("Metrics computation failed: %s", exc)
            warnings.append(f"Metrics computation failed: {exc}")
            metrics = FormationMetrics(
                team_width=0.0,
                team_length=0.0,
                compactness=0.0,
                centroid_x=0.0,
                centroid_y=0.0,
                convex_hull_area=0.0,
                defensive_line=0.0,
                midfield_line=0.0,
                forward_line=0.0,
                vertical_stretch=0.0,
                horizontal_stretch=0.0,
            )

        duration = time.perf_counter() - start
        result = FormationAnalysisResult(
            team_id=resolved_team_id,
            detected_formation=detection.detected_formation,
            confidence=detection.confidence,
            metrics=metrics,
            frame_number=frame_number,
            timestamp=ts,
            analysis_duration_seconds=duration,
            warnings=warnings,
            errors=errors,
        )
        self._results.append(result)
        logger.info("Analysis completed in %.3fs", duration)
        return result

    def analyze_team(
        self,
        players: Sequence[PlayerPosition],
        team_id: int,
        frame_number: int = 0,
        timestamp: datetime | None = None,
    ) -> FormationAnalysisResult:
        """Analyze a specific team by filtering and delegating to analyze().

        Args:
            players: Sequence of PlayerPosition instances from all teams.
            team_id: Team identifier to analyze.
            frame_number: Frame number being analyzed.
            timestamp: Timestamp of the analysis.

        Returns:
            FormationAnalysisResult for the specified team.
        """
        team_players = [p for p in players if p.team_id == team_id]
        if not team_players:
            raise ValueError(f"No players found for team_id={team_id}.")
        return self.analyze(
            team_players,
            frame_number=frame_number,
            timestamp=timestamp,
            team_id=team_id,
        )

    def analyze_match(
        self,
        players: Sequence[PlayerPosition],
        frame_number: int = 0,
        timestamp: datetime | None = None,
    ) -> dict[int, FormationAnalysisResult]:
        """Analyze all teams present in the player list.

        Args:
            players: Sequence of PlayerPosition instances from all teams.
            frame_number: Frame number being analyzed.
            timestamp: Timestamp of the analysis.

        Returns:
            Mapping from team_id to FormationAnalysisResult.
        """
        teams: dict[int, list[PlayerPosition]] = {}
        for player in players:
            teams.setdefault(player.team_id, []).append(player)

        results: dict[int, FormationAnalysisResult] = {}
        for team_id, team_players in teams.items():
            try:
                result = self.analyze_team(
                    team_players,
                    team_id=team_id,
                    frame_number=frame_number,
                    timestamp=timestamp,
                )
                results[team_id] = result
            except Exception as exc:
                logger.warning("Skipping team_id=%s: %s", team_id, exc)
        return results

    def batch_analyze(
        self,
        frames: Sequence[Sequence[PlayerPosition]],
        frame_numbers: Sequence[int] | None = None,
    ) -> list[FormationAnalysisResult]:
        """Analyze multiple frames in batch.

        Args:
            frames: Sequence of player lists, one per frame.
            frame_numbers: Optional frame numbers for each frame.

        Returns:
            List of FormationAnalysisResult instances.
        """
        results: list[FormationAnalysisResult] = []
        for idx, frame_players in enumerate(frames):
            fn = frame_numbers[idx] if frame_numbers and idx < len(frame_numbers) else idx
            try:
                result = self.analyze(frame_players, frame_number=fn)
                results.append(result)
            except Exception as exc:
                logger.warning("Frame %s analysis failed: %s", fn, exc)
        return results