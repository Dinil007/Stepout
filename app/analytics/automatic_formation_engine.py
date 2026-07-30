"""
Automatic Formation Detection Engine

Detects team formations at regular intervals throughout a match.
Tracks formation changes and calculates tactical metrics.

Integrates with existing FormationDetector and FormationMetricsEngine.
"""

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

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
class FormationSnapshot:
    """Single formation detection snapshot.

    Attributes:
        frame_number: Frame number of detection.
        timestamp: Timestamp of detection.
        team_id: Team identifier.
        formation: Detected formation name.
        confidence: Detection confidence score.
        metrics: Calculated formation metrics.
        is_formation_change: Whether this differs from previous detection.
        change_from: Previous formation name (if changed).
    """

    frame_number: int
    timestamp: datetime
    team_id: int
    formation: str
    confidence: float
    metrics: FormationMetrics
    is_formation_change: bool = False
    change_from: Optional[str] = None


class AutomaticFormationEngine:
    """Automatic formation detection and tracking engine.

    Detects formations at regular intervals, tracks changes, and
    calculates tactical metrics for each detection.
    """

    def __init__(
        self,
        fps: float = 30.0,
        detection_interval_seconds: float = 5.0,
        min_confidence: float = 0.6,
        formation_change_threshold: float = 0.15,
        output_dir: Optional[Path] = None,
    ):
        """Initialize the automatic formation detection engine.

        Args:
            fps: Frames per second of the video.
            detection_interval_seconds: Interval between formation detections.
            min_confidence: Minimum confidence for valid detection.
            formation_change_threshold: Minimum confidence difference to register change.
            output_dir: Optional directory for JSON output.
        """
        self.fps = fps
        self.detection_interval_seconds = detection_interval_seconds
        self.detection_interval_frames = int(detection_interval_seconds * fps)
        self.min_confidence = min_confidence
        self.formation_change_threshold = formation_change_threshold
        self.output_dir = output_dir

        # Configuration
        self.config = FormationConfig(
            analysis_window_seconds=detection_interval_seconds,
            frame_stride=max(1, int(detection_interval_seconds * fps / 10)),
            minimum_confidence=min_confidence,
            formation_change_threshold=formation_change_threshold,
            minimum_tracked_players=6,
            ignore_goalkeeper=True,
        )

        # Components
        self.detector = FormationDetector(config=self.config)
        self.metrics_engine = FormationMetricsEngine(config=self.config)

        # State tracking
        self.snapshots: List[FormationSnapshot] = []
        self._last_formation: Dict[int, str] = {}  # team_id -> formation

    def reset(self) -> None:
        """Clear detection history."""
        self.snapshots.clear()
        self._last_formation.clear()
        logger.info("AutomaticFormationEngine reset.")

    def _convert_players(
        self, players: List[Dict[str, Any]], frame_number: int
    ) -> List[PlayerPosition]:
        """Convert player dictionaries to PlayerPosition instances.

        Args:
            players: List of player dicts with track_id, field_position, team_id.
            frame_number: Current frame number.

        Returns:
            List of PlayerPosition instances.
        """
        result = []
        for p in players:
            pos = p.get("field_position")
            if not pos or len(pos) < 2:
                continue
            x, y = float(pos[0]), float(pos[1])
            team_id = p.get("team_id", 0)
            track_id = p.get("track_id", 0)
            result.append(
                PlayerPosition(
                    player_id=track_id,
                    team_id=team_id,
                    team_name=str(team_id),
                    jersey_number=track_id,
                    x=x,
                    y=y,
                    frame_number=frame_number,
                    timestamp=datetime.now(timezone.utc),
                )
            )
        return result

    def _calculate_metrics(
        self, players: List[PlayerPosition]
    ) -> FormationMetrics:
        """Calculate formation metrics for a set of players.

        Args:
            players: List of PlayerPosition instances.

        Returns:
            FormationMetrics instance.
        """
        try:
            return self.metrics_engine.compute_metrics(list(players))
        except Exception as e:
            logger.warning(f"Metrics calculation failed: {e}")
            return FormationMetrics(
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

    def _detect_formation(
        self, players: List[PlayerPosition], frame_number: int
    ) -> Optional[FormationDetection]:
        """Detect formation for a single team.

        Args:
            players: List of PlayerPosition instances for one team.
            frame_number: Current frame number.

        Returns:
            FormationDetection or None if detection fails.
        """
        try:
            detection = self.detector.detect_team(
                players, frame_number=frame_number
            )
            if detection.confidence < self.min_confidence:
                return None
            return detection
        except Exception as e:
            logger.warning(f"Formation detection failed: {e}")
            return None

    def process_frame(
        self,
        frame_number: int,
        players: List[Dict[str, Any]],
        team_assignments: Dict[int, Any],
    ) -> List[FormationSnapshot]:
        """Process a single frame if it's time for detection.

        Args:
            frame_number: Current frame number.
            players: List of player dictionaries.
            team_assignments: Mapping of track_id to team_id.

        Returns:
            List of FormationSnapshot instances (empty if not detection frame).
        """
        # Check if this is a detection frame
        if frame_number % self.detection_interval_frames != 0:
            return []

        # Group players by team
        teams: Dict[int, List[Dict[str, Any]]] = {}
        for p in players:
            team_id = p.get("team_id")
            if team_id is None:
                continue
            teams.setdefault(team_id, []).append(p)

        snapshots = []
        for team_id, team_players in teams.items():
            # Convert to PlayerPosition
            player_positions = self._convert_players(team_players, frame_number)
            if len(player_positions) < self.config.minimum_tracked_players:
                continue

            # Detect formation
            detection = self._detect_formation(player_positions, frame_number)
            if detection is None:
                continue

            # Calculate metrics
            metrics = self._calculate_metrics(player_positions)

            # Check for formation change
            prev_formation = self._last_formation.get(team_id)
            is_change = False
            change_from = None
            prev_conf = 0.0
            if isinstance(prev_formation, dict):
                prev_conf = prev_formation.get("confidence", 0.0)
                prev_name = prev_formation.get("name", prev_formation.get("formation"))
            else:
                prev_conf = 0.0
                prev_name = prev_formation

            if prev_name and prev_name != detection.detected_formation:
                if detection.confidence >= prev_conf + self.formation_change_threshold:
                    is_change = True
                    change_from = prev_name
                    logger.info(
                        f"Formation change detected for team {team_id}: "
                        f"{change_from} -> {detection.detected_formation}"
                    )

            # Create snapshot
            snapshot = FormationSnapshot(
                frame_number=frame_number,
                timestamp=datetime.now(timezone.utc),
                team_id=team_id,
                formation=detection.detected_formation,
                confidence=detection.confidence,
                metrics=metrics,
                is_formation_change=is_change,
                change_from=change_from,
            )
            snapshots.append(snapshot)

            # Update last formation
            self._last_formation[team_id] = {
                "name": detection.detected_formation,
                "confidence": detection.confidence,
            }

        self.snapshots.extend(snapshots)
        return snapshots

    def get_formation_timeline(self) -> List[Dict[str, Any]]:
        """Get formation timeline as list of dictionaries.

        Returns:
            List of formation snapshots in JSON-serializable format.
        """
        timeline = []
        for snap in self.snapshots:
            timeline.append({
                "frame_number": snap.frame_number,
                "timestamp": snap.timestamp.isoformat(),
                "team_id": snap.team_id,
                "formation": snap.formation,
                "confidence": round(snap.confidence, 4),
                "is_formation_change": snap.is_formation_change,
                "change_from": snap.change_from,
                "metrics": {
                    "team_width_m": round(snap.metrics.team_width, 2),
                    "team_length_m": round(snap.metrics.team_length, 2),
                    "compactness_m": round(snap.metrics.compactness, 2),
                    "centroid_x_m": round(snap.metrics.centroid_x, 2),
                    "centroid_y_m": round(snap.metrics.centroid_y, 2),
                    "defensive_line_m": round(snap.metrics.defensive_line, 2),
                    "midfield_line_m": round(snap.metrics.midfield_line, 2),
                    "forward_line_m": round(snap.metrics.forward_line, 2),
                    "convex_hull_area_m2": round(snap.metrics.convex_hull_area, 2),
                    "vertical_stretch_m": round(snap.metrics.vertical_stretch, 2),
                    "horizontal_stretch_m": round(snap.metrics.horizontal_stretch, 2),
                },
            })
        return timeline

    def get_formation_summary(self) -> Dict[str, Any]:
        """Get summary statistics for all detections.

        Returns:
            Dictionary with formation counts, changes, and averages.
        """
        if not self.snapshots:
            return {"total_detections": 0}

        # Count formations
        formation_counts: Dict[str, int] = {}
        for snap in self.snapshots:
            formation_counts[snap.formation] = (
                formation_counts.get(snap.formation, 0) + 1
            )

        # Count changes
        changes = [s for s in self.snapshots if s.is_formation_change]
        change_count = len(changes)

        # Average confidence
        avg_confidence = (
            sum(s.confidence for s in self.snapshots) / len(self.snapshots)
        )

        # Average metrics
        avg_width = np.mean([s.metrics.team_width for s in self.snapshots])
        avg_length = np.mean([s.metrics.team_length for s in self.snapshots])
        avg_compactness = np.mean([s.metrics.compactness for s in self.snapshots])

        return {
            "total_detections": len(self.snapshots),
            "unique_formations": len(formation_counts),
            "formation_distribution": formation_counts,
            "formation_changes": change_count,
            "average_confidence": round(avg_confidence, 4),
            "average_team_width_m": round(avg_width, 2),
            "average_team_length_m": round(avg_length, 2),
            "average_compactness_m": round(avg_compactness, 2),
        }

    def validate(self) -> List[str]:
        """Validate detection results.

        Returns:
            List of validation errors (empty if valid).
        """
        errors = []
        if not self.snapshots:
            errors.append("No formation detections recorded.")
            return errors

        # Check for duplicate frame numbers per team
        seen = {}
        for snap in self.snapshots:
            key = (snap.team_id, snap.frame_number)
            if key in seen:
                errors.append(f"Duplicate detection at team={snap.team_id}, frame={snap.frame_number}")
            seen[key] = True

        # Validate confidence ranges
        for snap in self.snapshots:
            if not (0.0 <= snap.confidence <= 1.0):
                errors.append(
                    f"Invalid confidence {snap.confidence} at frame {snap.frame_number}"
                )

        # Validate metrics ranges
        for snap in self.snapshots:
            m = snap.metrics
            if m.team_width < 0 or m.team_width > 100:
                errors.append(f"Invalid team_width at frame {snap.frame_number}")
            if m.team_length < 0 or m.team_length > 120:
                errors.append(f"Invalid team_length at frame {snap.frame_number}")

        return errors

    def save(self) -> None:
        """Save results to JSON files."""
        if self.output_dir is None:
            logger.warning("No output directory configured, skipping save.")
            return

        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Save formation timeline
        timeline = self.get_formation_timeline()
        with open(self.output_dir / "formation_timeline.json", "w") as f:
            json.dump(timeline, f, indent=4)

        # Save formation analysis summary
        summary = self.get_formation_summary()
        with open(self.output_dir / "formation_analysis.json", "w") as f:
            json.dump(summary, f, indent=4)

        logger.info(
            f"Formation analysis saved: {len(timeline)} detections, "
            f"{summary.get('formation_changes', 0)} changes"
        )

    def run_match_analysis(
        self,
        frames: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Run automatic formation detection on a full match.

        Args:
            frames: List of frame data dicts with frame_number, players, team_assignments.

        Returns:
            Summary dictionary with detection results.
        """
        self.reset()
        logger.info(
            f"Starting automatic formation detection: "
            f"{len(frames)} frames, interval={self.detection_interval_seconds}s"
        )

        for frame_data in frames:
            frame_number = frame_data.get("frame_number", 0)
            players = frame_data.get("players", [])
            team_assignments = frame_data.get("team_assignments", {})

            # Assign team_id to players if missing
            for p in players:
                if "team_id" not in p:
                    track_id = p.get("track_id")
                    if track_id in team_assignments:
                        p["team_id"] = team_assignments[track_id]

            self.process_frame(frame_number, players, team_assignments)

        # Validate and save
        errors = self.validate()
        if errors:
            for err in errors:
                logger.error(f"Validation error: {err}")

        self.save()

        summary = self.get_formation_summary()
        logger.info(
            f"Formation detection complete: {summary.get('total_detections', 0)} detections, "
            f"{summary.get('formation_changes', 0)} changes"
        )
        return summary