"""Aggregate exported analytics artifacts into one match context."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Set

from app.ai.schemas import AnalyticsArtifacts, JsonDict

LOGGER = logging.getLogger(__name__)


class AnalyticsAggregator:
    """Builds a clean JSON object from generated analytics files."""

    REQUIRED_FILES = ("analytics.json",)
    OPTIONAL_FILES = (
        "pass_summary.json",
        "pass_events.json",
        "shot_summary.json",
        "shot_events.json",
        "team_possession_summary.json",
        "team_passing_summary.json",
        "average_positions.json",
        "xg_shots.json",
        "team_xg_summary.json",
        "player_xg_summary.json",
        "xg_summary.json",
        "xa_passes.json",
        "team_xa_summary.json",
        "player_xa_summary.json",
        "xa_summary.json",
        "pressing_events.json",
        "pressing_sequences.json",
        "pressing_metrics.json",
        "pressing_detection.json",
        "pressing_timeline.json",
    )

    def validate(self, output_dir: Path, artifacts: AnalyticsArtifacts) -> List[str]:
        missing = [
            filename
            for filename in self.REQUIRED_FILES
            if not (output_dir / filename).exists()
        ]
        warnings = [
            filename
            for filename in self.OPTIONAL_FILES
            if not (output_dir / filename).exists()
        ]
        messages = [f"missing required artifact: {name}" for name in missing]
        messages.extend(f"missing optional artifact: {name}" for name in warnings)
        if not artifacts.average_positions:
            messages.append("average positions unavailable; player reports are limited")
        return messages

    def aggregate(self, artifacts: AnalyticsArtifacts, match_id: str) -> JsonDict:
        analytics = artifacts.analytics
        pass_summary = artifacts.pass_summary or analytics.get("pass_summary", {})
        shot_summary = artifacts.shot_summary or analytics.get("shot_summary", {})
        possession = (
            artifacts.team_possession_summary
            or analytics.get("team_possession_summary", {})
            or analytics.get("possession_summary", {})
        )
        team_passing = artifacts.team_passing_summary
        average_positions = artifacts.average_positions

        teams = self._build_teams(pass_summary, possession, team_passing)
        players = self._build_players(
            pass_summary=pass_summary,
            average_positions=average_positions,
            pass_events=artifacts.pass_events,
            shot_events=artifacts.shot_events,
        )

        # xG data — loaded as structured analytics if available
        xg_shots = artifacts.xg_shots or []
        team_xg_summary = artifacts.team_xg_summary or {}
        player_xg_summary = artifacts.player_xg_summary or {}
        xg_summary = artifacts.xg_summary or {}

        return {
            "match_id": match_id,
            "match": {
                "frames": analytics.get("match_info", {}).get("processed_frames"),
                "fps": analytics.get("match_info", {}).get("fps"),
                "input_video": analytics.get("match_info", {}).get("input_video"),
                "player_count": analytics.get("player_count", len(players)),
                "ball_detections_count": analytics.get("ball_detections_count"),
            },
            "teams": teams,
            "players": players,
            "events": {
                "passes": artifacts.pass_events,
                "shots": artifacts.shot_events,
                "summary": {
                    "passes": pass_summary,
                    "shots": shot_summary,
                    "possession": possession,
                },
            },
            "tactical": {
                "team_shapes": {
                    team: data.get("tactical_shape", {})
                    for team, data in team_passing.items()
                    if isinstance(data, dict)
                },
                "average_positions": average_positions,
            },
            "xg": {
                "shots": xg_shots,
                "team_summary": team_xg_summary,
                "player_summary": player_xg_summary,
                "summary": xg_summary,
            },
            # xA data — loaded as structured analytics if available
        "xa": {
            "passes": artifacts.xa_passes or [],
            "team_summary": artifacts.team_xa_summary or {},
            "player_summary": artifacts.player_xa_summary or {},
            "summary": artifacts.xa_summary or {},
        },
        "xt": {
            "actions": artifacts.xt_actions or [],
            "team_summary": artifacts.team_xt_summary or {},
            "player_summary": artifacts.player_xt_summary or {},
            "summary": artifacts.xt_summary or {},
        },
        "formation": {
            "windows": artifacts.formation_windows or [],
            "transitions": artifacts.formation_transitions or [],
            "metrics": artifacts.formation_metrics or {},
        },
        "pressing": {
            "events": artifacts.pressing_events or [],
            "sequences": artifacts.pressing_sequences or [],
            "metrics": artifacts.pressing_metrics or {},
            "detection": artifacts.pressing_detection or {},
            "timeline": artifacts.pressing_timeline or [],
        },
        }

    def _build_teams(
        self,
        pass_summary: JsonDict,
        possession: JsonDict,
        team_passing: JsonDict,
    ) -> JsonDict:
        possession_pct = possession.get("team_possession_pct", {})
        possession_time = possession.get("total_possession_time_seconds", {})
        team_passes = pass_summary.get("team_pass_summary", {})
        team_names: Set[str] = set(team_passes) | set(team_passing) | set(possession_pct)
        team_names.discard("Free Ball")

        teams: JsonDict = {}
        for team in sorted(team_names):
            passing = team_passes.get(team, {})
            tactical = team_passing.get(team, {})
            teams[team] = {
                "possession_pct": possession_pct.get(team),
                "possession_time_seconds": possession_time.get(team),
                "passes_attempted": passing.get(
                    "total_passes",
                    tactical.get("total_passes_attempted"),
                ),
                "passes_completed": passing.get(
                    "completed_passes",
                    tactical.get("completed_passes"),
                ),
                "pass_accuracy_pct": passing.get(
                    "accuracy_pct",
                    tactical.get("completion_pct"),
                ),
                "progressive_passes": tactical.get("progressive_passes"),
                "avg_pass_distance_m": passing.get("avg_distance_m"),
                "total_pass_distance_m": passing.get("total_distance_m"),
                "shots": 0,
                "shots_on_target": 0,
                "average_speed_mps": None,
                "tactical_shape": tactical.get("tactical_shape", {}),
            }
        return teams

    def _build_players(
        self,
        pass_summary: JsonDict,
        average_positions: JsonDict,
        pass_events: Iterable[JsonDict],
        shot_events: Iterable[JsonDict],
    ) -> JsonDict:
        player_passes = pass_summary.get("player_pass_summary", {})
        player_ids = set(str(pid) for pid in player_passes)
        player_ids.update(str(pid) for pid in average_positions)
        for event in pass_events:
            if event.get("passer") is not None:
                player_ids.add(str(event["passer"]))
            if event.get("receiver") is not None:
                player_ids.add(str(event["receiver"]))
        for event in shot_events:
            if event.get("player_id") is not None:
                player_ids.add(str(event["player_id"]))

        shot_counts = self._count_by_key(shot_events, "player_id")
        players: JsonDict = {}
        for player_id in sorted(player_ids, key=self._sort_key):
            passing = player_passes.get(player_id, {})
            position = average_positions.get(player_id, {})
            players[player_id] = {
                "player_id": player_id,
                "team": position.get("team", "Unknown"),
                "passes_attempted": passing.get("attempted", 0),
                "passes_completed": passing.get("completed", 0),
                "pass_accuracy_pct": passing.get("accuracy_pct"),
                "avg_pass_distance_m": passing.get("avg_distance_m"),
                "total_pass_distance_m": passing.get("total_distance_m"),
                "shots": shot_counts.get(player_id, 0),
                "shots_on_target": 0,
                "distance_covered_m": None,
                "sprint_count": None,
                "average_speed_mps": None,
                "touches": None,
                "possession_seconds": None,
                "heatmap_summary": {
                    "movement_radius": position.get("movement_radius"),
                    "samples": position.get("total_samples"),
                },
                "average_position": position.get("average_position"),
                "movement_radius": position.get("movement_radius"),
                "total_samples": position.get("total_samples"),
            }
        return players

    def _count_by_key(self, events: Iterable[JsonDict], key: str) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for event in events:
            value = event.get(key)
            if value is not None:
                counts[str(value)] = counts.get(str(value), 0) + 1
        return counts

    def _sort_key(self, value: str) -> Any:
        return int(value) if value.isdigit() else value
