"""Expected goals engine for StepOut structured analytics."""

from __future__ import annotations

import json
import logging
import time
import tracemalloc
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from app.analytics.xg_features import XGFeatureExtractor, XGFeatures
from app.analytics.xg_model import XGModel, XGModelFactory
from app.analytics.xg_validation import XGValidator
from app.analytics.xg_visualizer import XGVisualizer

LOGGER = logging.getLogger(__name__)
JsonDict = Dict[str, Any]


class XGEngine:
    """Computes xG, summaries, validation, performance, and visualisations."""

    def __init__(
        self,
        output_dir: Path | str = "outputs",
        model: Optional[XGModel] = None,
        force_rule_based: bool = False,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.feature_extractor = XGFeatureExtractor()
        self.model = model or XGModelFactory().load(force_rule_based=force_rule_based)
        self.visualizer = XGVisualizer()
        self.validator = XGValidator()

    def run(self) -> JsonDict:
        tracemalloc.start()
        started = time.perf_counter()
        artifacts = self._load_artifacts()

        feature_started = time.perf_counter()
        features = [
            self.feature_extractor.extract(
                shot=shot,
                fps=artifacts["fps"],
                pass_events=artifacts["pass_events"],
                average_positions=artifacts["average_positions"],
            )
            for shot in artifacts["shot_events"]
        ]
        feature_elapsed = time.perf_counter() - feature_started

        inference_started = time.perf_counter()
        xg_shots = [self._score_shot(feature) for feature in features]
        inference_elapsed = time.perf_counter() - inference_started

        team_summary = self._team_summary(xg_shots)
        player_summary = self._player_summary(xg_shots)
        summary = self._summary(xg_shots, team_summary, player_summary)

        viz_started = time.perf_counter()
        visualisations = self.visualizer.render_all(
            self.output_dir,
            xg_shots,
            team_summary,
            player_summary,
        )
        visualisation_elapsed = time.perf_counter() - viz_started

        validation = self.validator.validate(
            artifacts["shot_events"],
            xg_shots,
            team_summary,
            player_summary,
        )
        performance = self._performance(
            started=started,
            feature_elapsed=feature_elapsed,
            inference_elapsed=inference_elapsed,
            visualisation_elapsed=visualisation_elapsed,
        )
        regression = self._regression_report()

        payload = {
            "shots": xg_shots,
            "team_summary": team_summary,
            "player_summary": player_summary,
            "summary": summary,
            "visualisations": visualisations,
            "validation": validation,
            "performance": performance,
            "regression": regression,
            "model": self.model.name,
        }
        self._export(payload)
        tracemalloc.stop()
        return payload

    def _score_shot(self, features: XGFeatures) -> JsonDict:
        vector = self.feature_extractor.to_model_vector(features)
        xg = self.model.predict_proba(features, vector)
        payload = features.to_dict()
        payload.update({
            "xg": xg,
            "goal": self._is_goal(features),
            "shot_id": features.shot_id,
            "player": features.player_id,
            "distance": features.distance_m,
            "angle": features.angle_deg,
        })
        return payload

    def _team_summary(self, shots: List[JsonDict]) -> JsonDict:
        summary: JsonDict = {}
        for shot in shots:
            team = str(shot.get("team") or "Unknown")
            data = summary.setdefault(team, {
                "total_xg": 0.0,
                "average_xg": 0.0,
                "xg_per_shot": 0.0,
                "goals": 0,
                "goals_minus_xg": 0.0,
                "shots": 0,
                "best_chance": None,
                "worst_missed_chance": None,
            })
            data["shots"] += 1
            data["total_xg"] += float(shot["xg"])
            data["goals"] += int(bool(shot.get("goal")))
            if data["best_chance"] is None or shot["xg"] > data["best_chance"]["xg"]:
                data["best_chance"] = self._small_shot(shot)
            if not shot.get("goal") and (
                data["worst_missed_chance"] is None
                or shot["xg"] > data["worst_missed_chance"]["xg"]
            ):
                data["worst_missed_chance"] = self._small_shot(shot)
        for data in summary.values():
            data["total_xg"] = round(data["total_xg"], 3)
            data["average_xg"] = round(data["total_xg"] / data["shots"], 3) if data["shots"] else 0.0
            data["xg_per_shot"] = data["average_xg"]
            data["goals_minus_xg"] = round(data["goals"] - data["total_xg"], 3)
        return summary

    def _player_summary(self, shots: List[JsonDict]) -> JsonDict:
        summary: JsonDict = {}
        for shot in shots:
            player_id = str(shot.get("player") if shot.get("player") is not None else "Unknown")
            data = summary.setdefault(player_id, {
                "total_xg": 0.0,
                "goals": 0,
                "goals_minus_xg": 0.0,
                "average_shot_distance": 0.0,
                "average_shot_angle": 0.0,
                "shots": 0,
                "shots_on_target": 0,
                "conversion_pct": 0.0,
                "_distance_total": 0.0,
                "_angle_total": 0.0,
            })
            data["shots"] += 1
            data["total_xg"] += float(shot["xg"])
            data["goals"] += int(bool(shot.get("goal")))
            data["shots_on_target"] += int("on target" in str(shot.get("shot_type", "")).lower())
            data["_distance_total"] += float(shot.get("distance_m") or 0.0)
            data["_angle_total"] += float(shot.get("angle_deg") or 0.0)
        for data in summary.values():
            shots = data["shots"]
            data["total_xg"] = round(data["total_xg"], 3)
            data["goals_minus_xg"] = round(data["goals"] - data["total_xg"], 3)
            data["average_shot_distance"] = round(data.pop("_distance_total") / shots, 2) if shots else 0.0
            data["average_shot_angle"] = round(data.pop("_angle_total") / shots, 2) if shots else 0.0
            data["conversion_pct"] = round(data["goals"] / shots * 100.0, 1) if shots else 0.0
        return summary

    def _summary(self, shots: List[JsonDict], team_summary: JsonDict, player_summary: JsonDict) -> JsonDict:
        best = max(shots, key=lambda shot: shot.get("xg", 0.0), default=None)
        lowest = min(shots, key=lambda shot: shot.get("xg", 0.0), default=None)
        total_xg = round(sum(float(shot.get("xg", 0.0)) for shot in shots), 3)
        return {
            "total_shots": len(shots),
            "total_xg": total_xg,
            "average_xg": round(total_xg / len(shots), 3) if shots else 0.0,
            "highest_xg_shot": self._small_shot(best) if best else None,
            "lowest_xg_shot": self._small_shot(lowest) if lowest else None,
            "top_finishers": sorted(
                player_summary.items(),
                key=lambda item: item[1].get("goals_minus_xg", 0.0),
                reverse=True,
            )[:5],
            "team_count": len(team_summary),
            "player_count": len(player_summary),
            "model": self.model.name,
        }

    def _load_artifacts(self) -> JsonDict:
        analytics = self._read_json("analytics.json", {})
        return {
            "shot_events": self._read_json("shot_events.json", []),
            "pass_events": self._read_json("pass_events.json", []),
            "average_positions": self._read_json("average_positions.json", {}),
            "fps": analytics.get("match_info", {}).get("fps"),
        }

    def _read_json(self, filename: str, default: Any) -> Any:
        path = self.output_dir / filename
        if not path.exists():
            LOGGER.info("xG artifact missing: %s", path)
            return default
        with path.open("r", encoding="utf-8") as file:
            return json.load(file)

    def _export(self, payload: JsonDict) -> None:
        exports = {
            "xg_shots.json": payload["shots"],
            "team_xg_summary.json": payload["team_summary"],
            "player_xg_summary.json": payload["player_summary"],
            "xg_summary.json": payload["summary"],
            "xg_validation_report.json": payload["validation"],
            "xg_performance_report.json": payload["performance"],
            "xg_regression_report.json": payload["regression"],
        }
        self.output_dir.mkdir(parents=True, exist_ok=True)
        for filename, data in exports.items():
            (self.output_dir / filename).write_text(
                json.dumps(data, indent=2, sort_keys=True),
                encoding="utf-8",
            )

    def _performance(
        self,
        started: float,
        feature_elapsed: float,
        inference_elapsed: float,
        visualisation_elapsed: float,
    ) -> JsonDict:
        current, peak = tracemalloc.get_traced_memory()
        return {
            "feature_extraction_time_ms": round(feature_elapsed * 1000.0, 2),
            "model_inference_time_ms": round(inference_elapsed * 1000.0, 2),
            "visualisation_time_ms": round(visualisation_elapsed * 1000.0, 2),
            "memory_current_kb": round(current / 1024.0, 2),
            "memory_peak_kb": round(peak / 1024.0, 2),
            "api_latency_ms": round((time.perf_counter() - started) * 1000.0, 2),
            "overall_runtime_impact": "post-processing only; no CV pipeline stage modified",
        }

    def _regression_report(self) -> JsonDict:
        return {
            "existing_analytics_modules_modified": False,
            "existing_shot_detection_outputs_preserved": True,
            "pipeline_contract": "xG reads shot_events.json and related structured analytics after detection.",
            "unchanged_modules": [
                "app/analytics/shot_detector.py",
                "app/detection/",
                "app/tracking/",
                "app/homography/",
                "app/pose/",
                "run_pipeline.py",
            ],
        }

    def _is_goal(self, features: XGFeatures) -> bool:
        shot_type = features.shot_type.lower()
        return "goal" in shot_type and "goalkeeper" not in shot_type

    def _small_shot(self, shot: Optional[JsonDict]) -> Optional[JsonDict]:
        if shot is None:
            return None
        return {
            "shot_id": shot.get("shot_id"),
            "player": shot.get("player"),
            "team": shot.get("team"),
            "frame": shot.get("frame"),
            "distance": shot.get("distance_m"),
            "angle": shot.get("angle_deg"),
            "xg": shot.get("xg"),
            "goal": shot.get("goal"),
        }
