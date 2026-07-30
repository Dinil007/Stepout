"""Deterministic insight computation before LLM analysis."""

from __future__ import annotations

from collections import Counter
from statistics import mean
from typing import Any, Dict, Iterable, List, Optional, Tuple

from app.ai.schemas import JsonDict


class InsightEngine:
    """Computes measurable insights from aggregated analytics."""

    def compute(self, context: JsonDict) -> JsonDict:
        teams = context.get("teams", {})
        players = context.get("players", {})
        passes = context.get("events", {}).get("passes", [])
        shots = context.get("events", {}).get("shots", [])

        return {
            "passing": self._passing_insights(teams, players, passes),
            "shooting": self._shooting_insights(players, shots, context),
            "possession": self._possession_insights(teams),
            "tactical": self._tactical_insights(context),
            "movement": self._movement_insights(players),
            "data_quality": self._data_quality(context),
        }

    def _passing_insights(
        self,
        teams: JsonDict,
        players: JsonDict,
        passes: Iterable[JsonDict],
    ) -> JsonDict:
        pass_list = list(passes)
        pair_counter = Counter(
            (str(p.get("passer")), str(p.get("receiver")))
            for p in pass_list
            if p.get("successful") and p.get("passer") is not None and p.get("receiver") is not None
        )
        progressive = [
            p for p in pass_list if float(p.get("distance_m") or 0.0) >= 10.0
        ]
        return {
            "highest_team_pass_accuracy": self._max_metric(teams, "pass_accuracy_pct"),
            "highest_player_pass_accuracy": self._max_metric(
                {
                    pid: pdata
                    for pid, pdata in players.items()
                    if int(pdata.get("passes_attempted") or 0) > 0
                },
                "pass_accuracy_pct",
            ),
            "most_completed_passes": self._max_metric(players, "passes_completed"),
            "most_progressive_passes_team": self._max_metric(
                teams,
                "progressive_passes",
            ),
            "progressive_pass_count": len(progressive),
            "best_passing_pair": self._best_pair(pair_counter),
        }

    def _shooting_insights(
        self,
        players: JsonDict,
        shots: Iterable[JsonDict],
        context: JsonDict,
    ) -> JsonDict:
        shot_list = list(shots)
        distances = [float(s.get("distance_m")) for s in shot_list if s.get("distance_m") is not None]
        return {
            "most_shots": self._max_metric(players, "shots"),
            "most_shots_on_target": self._max_metric(players, "shots_on_target"),
            "average_shot_distance_m": round(mean(distances), 2) if distances else context.get("events", {}).get("summary", {}).get("shots", {}).get("average_shot_distance_m"),
            "longest_shot": max(shot_list, key=lambda s: float(s.get("distance_m") or 0.0), default=None),
        }

    def _possession_insights(self, teams: JsonDict) -> JsonDict:
        dominant = self._max_metric(teams, "possession_pct")
        return {
            "team_dominance": dominant,
            "ball_retention_proxy": self._max_metric(teams, "pass_accuracy_pct"),
            "longest_possession": None,
            "note": "Longest possession requires possession sequence exports; unavailable when only summary totals exist.",
        }

    def _tactical_insights(self, context: JsonDict) -> JsonDict:
        shapes = context.get("tactical", {}).get("team_shapes", {})
        return {
            "widest_team": self._max_nested_metric(shapes, "width_m"),
            "deepest_team": self._max_nested_metric(shapes, "depth_m"),
            "most_compact_team": self._min_nested_metric(shapes, "compactness"),
            "defensive_line": self._metric_by_team(shapes, "defensive_line_height_m"),
            "midfield_line": self._metric_by_team(shapes, "midfield_line_height_m"),
            "formation_placeholder": "Formation inference requires role/line clustering and is not asserted by the AI layer.",
        }

    def _movement_insights(self, players: JsonDict) -> JsonDict:
        return {
            "largest_movement_radius": self._max_metric(players, "movement_radius"),
            "most_position_samples": self._max_metric(players, "total_samples"),
            "distance_covered": self._max_metric(players, "distance_covered_m"),
            "sprint_count": self._max_metric(players, "sprint_count"),
            "average_speed": self._max_metric(players, "average_speed_mps"),
        }

    def _data_quality(self, context: JsonDict) -> JsonDict:
        players = context.get("players", {})
        unavailable = []
        for metric in ("distance_covered_m", "sprint_count", "average_speed_mps", "touches"):
            if all(player.get(metric) is None for player in players.values()):
                unavailable.append(metric)
        return {
            "players_detected": len(players),
            "unavailable_player_metrics": unavailable,
            "uses_raw_video": False,
        }

    def _max_metric(self, data: JsonDict, metric: str) -> Optional[JsonDict]:
        valid = [
            (key, value.get(metric))
            for key, value in data.items()
            if isinstance(value, dict) and value.get(metric) is not None
        ]
        if not valid:
            return None
        key, value = max(valid, key=lambda item: float(item[1] or 0.0))
        return {"id": key, "value": value}

    def _max_nested_metric(self, data: JsonDict, metric: str) -> Optional[JsonDict]:
        return self._nested_metric(data, metric, max)

    def _min_nested_metric(self, data: JsonDict, metric: str) -> Optional[JsonDict]:
        return self._nested_metric(data, metric, min)

    def _nested_metric(self, data: JsonDict, metric: str, selector: Any) -> Optional[JsonDict]:
        valid: List[Tuple[str, Any]] = []
        for team, shape in data.items():
            if isinstance(shape, dict) and shape.get(metric) is not None:
                valid.append((team, shape[metric]))
        if not valid:
            return None
        team, value = selector(valid, key=lambda item: float(item[1] or 0.0))
        return {"team": team, "value": value}

    def _metric_by_team(self, data: JsonDict, metric: str) -> JsonDict:
        return {
            team: shape.get(metric)
            for team, shape in data.items()
            if isinstance(shape, dict) and shape.get(metric) is not None
        }

    def _best_pair(self, pairs: Counter) -> Optional[JsonDict]:
        if not pairs:
            return None
        (passer, receiver), count = pairs.most_common(1)[0]
        return {"passer": passer, "receiver": receiver, "completed_passes": count}
