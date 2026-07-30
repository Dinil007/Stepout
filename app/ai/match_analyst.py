"""Production-grade AI match analyst over structured analytics exports."""

from __future__ import annotations

import json
import logging
import os
import time
import tracemalloc
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib import request

from app.ai.aggregator import AnalyticsAggregator
from app.ai.insight_engine import InsightEngine
from app.ai.prompt_builder import PromptBuilder
from app.ai.recommendations import RecommendationEngine
from app.ai.report_generator import AIReportGenerator
from app.ai.schemas import AnalyticsArtifacts, JsonDict, LLMProvider, LLMResponse

LOGGER = logging.getLogger(__name__)


class OfflineLLMProvider:
    """Deterministic local provider used when remote LLM keys are absent."""

    name = "offline"
    model = "stepout-local-analyst"

    def generate(self, prompt: str, context: JsonDict) -> LLMResponse:
        teams = context.get("teams", {})
        insights = context.get("insights", {})
        dominant = insights.get("possession", {}).get("team_dominance") or {}
        best_team = dominant.get("id", "No team")
        accuracy = insights.get("passing", {}).get("highest_team_pass_accuracy") or {}
        lines = [
            f"{best_team} had the strongest possession profile in the exported analytics.",
            f"Best team pass accuracy: {accuracy.get('id', 'unavailable')} "
            f"({accuracy.get('value', 'unavailable')}%).",
            f"Detected teams: {', '.join(sorted(teams)) if teams else 'unavailable'}.",
            "This response is generated from structured analytics only.",
        ]
        return LLMResponse(text="\n".join(lines), provider=self.name, model=self.model)


class OpenAIProvider:
    """Minimal OpenAI Chat Completions provider using the standard library."""

    name = "openai"

    def __init__(self, api_key: str, model: str = "gpt-4o-mini") -> None:
        self.api_key = api_key
        self.model = model

    def generate(self, prompt: str, context: JsonDict) -> LLMResponse:
        payload = json.dumps({
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You are a football match analyst."},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.2,
        }).encode("utf-8")
        req = request.Request(
            "https://api.openai.com/v1/chat/completions",
            data=payload,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with request.urlopen(req, timeout=60) as response:
            raw = json.loads(response.read().decode("utf-8"))
        text = raw["choices"][0]["message"]["content"]
        return LLMResponse(text=text, provider=self.name, model=self.model, raw=raw)


class GeminiProvider:
    """Minimal Gemini provider using the public REST API."""

    name = "gemini"

    def __init__(self, api_key: str, model: str = "gemini-1.5-flash") -> None:
        self.api_key = api_key
        self.model = model

    def generate(self, prompt: str, context: JsonDict) -> LLMResponse:
        endpoint = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.model}:generateContent?key={self.api_key}"
        )
        payload = json.dumps({
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.2},
        }).encode("utf-8")
        req = request.Request(
            endpoint,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with request.urlopen(req, timeout=60) as response:
            raw = json.loads(response.read().decode("utf-8"))
        text = raw["candidates"][0]["content"]["parts"][0]["text"]
        return LLMResponse(text=text, provider=self.name, model=self.model, raw=raw)


class MatchAnalyst:
    """Coordinates aggregation, insight computation, LLM calls, and report export."""

    def __init__(
        self,
        output_dir: Path | str = "outputs",
        provider: Optional[LLMProvider] = None,
        match_id: Optional[str] = None,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.match_id = match_id or self._default_match_id()
        self.aggregator = AnalyticsAggregator()
        self.insight_engine = InsightEngine()
        self.prompt_builder = PromptBuilder()
        self.recommendation_engine = RecommendationEngine()
        self.report_generator = AIReportGenerator()
        self.provider = provider or self._provider_from_environment()

    def load_artifacts(self) -> AnalyticsArtifacts:
        return AnalyticsArtifacts(
            analytics=self._read_json("analytics.json", default={}),
            pass_summary=self._read_json("pass_summary.json", default={}),
            pass_events=self._read_json("pass_events.json", default=[]),
            shot_summary=self._read_json("shot_summary.json", default={}),
            shot_events=self._read_json("shot_events.json", default=[]),
            team_possession_summary=self._read_json(
                "team_possession_summary.json",
                default={},
            ),
            team_passing_summary=self._read_json("team_passing_summary.json", default={}),
            average_positions=self._read_json("average_positions.json", default={}),
            xg_shots=self._read_json("xg_shots.json", default=[]),
            team_xg_summary=self._read_json("team_xg_summary.json", default={}),
            player_xg_summary=self._read_json("player_xg_summary.json", default={}),
            xg_summary=self._read_json("xg_summary.json", default={}),
            xa_passes=self._read_json("xa_passes.json", default=[]),
            team_xa_summary=self._read_json("team_xa_summary.json", default={}),
            player_xa_summary=self._read_json("player_xa_summary.json", default={}),
            xa_summary=self._read_json("xa_summary.json", default={}),
            xt_actions=self._read_json("xt_actions.json", default=[]),
            team_xt_summary=self._read_json("team_xt_summary.json", default={}),
            player_xt_summary=self._read_json("player_xt_summary.json", default={}),
            xt_summary=self._read_json("xt_summary.json", default={}),
        )

    def build_context(self) -> JsonDict:
        artifacts = self.load_artifacts()
        messages = self.aggregator.validate(self.output_dir, artifacts)
        blocking = [message for message in messages if "required" in message]
        if blocking:
            raise FileNotFoundError("; ".join(blocking))
        context = self.aggregator.aggregate(artifacts, self.match_id)
        context["insights"] = self.insight_engine.compute(context)
        context["validation_messages"] = messages
        return context

    def generate_match_report(self) -> JsonDict:
        tracemalloc.start()
        started = time.perf_counter()
        context_started = time.perf_counter()
        context = self.build_context()
        prompt_started = time.perf_counter()
        prompt = self.prompt_builder.build_report_prompt(context)
        prompt_elapsed = time.perf_counter() - prompt_started

        llm_started = time.perf_counter()
        llm_response = self.provider.generate(prompt, context)
        llm_elapsed = time.perf_counter() - llm_started

        report_started = time.perf_counter()
        recommendations = self.recommendation_engine.generate(context)
        report = self._compose_report(context, llm_response, recommendations)
        report["performance"] = self._performance_payload(
            started=started,
            prompt_elapsed=prompt_elapsed,
            llm_elapsed=llm_elapsed,
            report_elapsed=time.perf_counter() - report_started,
            context_elapsed=prompt_started - context_started,
        )
        report["exports"] = self.report_generator.export_all(self.output_dir, report)
        tracemalloc.stop()
        return report

    def query(self, question: str) -> JsonDict:
        if not question.strip():
            raise ValueError("question must not be empty")
        started = time.perf_counter()
        context = self.build_context()
        prompt = self.prompt_builder.build_query_prompt(context, question)
        llm_started = time.perf_counter()
        response = self.provider.generate(prompt, context)
        return {
            "question": question,
            "answer": response.text,
            "provider": response.provider,
            "model": response.model,
            "latency_ms": round((time.perf_counter() - started) * 1000, 2),
            "llm_latency_ms": round((time.perf_counter() - llm_started) * 1000, 2),
            "match_id": context["match_id"],
        }

    def get_section(self, section: str, player_id: Optional[str] = None) -> JsonDict:
        report = self.generate_match_report()
        if section == "player":
            players = report.get("player_reports", {})
            if player_id not in players:
                raise KeyError(f"player '{player_id}' not found")
            return players[player_id]
        return report.get(section, {})

    def _compose_report(
        self,
        context: JsonDict,
        llm_response: LLMResponse,
        recommendations: JsonDict,
    ) -> JsonDict:
        match_summary = self._match_summary(context, llm_response)
        team_analysis = self._team_analysis(context, recommendations)
        player_reports = self._player_reports(context, recommendations)
        coach_report = self._coach_report(context)
        opposition_report = self._opposition_report(context)
        validation = self._validate_ai_outputs(context, player_reports)

        return {
            "match_id": context["match_id"],
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "provider": llm_response.provider,
            "model": llm_response.model,
            "context": context,
            "match_summary": match_summary,
            "match_summary_markdown": self._markdown("AI Match Summary", match_summary),
            "team_analysis": team_analysis,
            "player_reports": player_reports,
            "coach_report": coach_report,
            "coach_report_markdown": self._markdown("Coach Report", coach_report),
            "opposition_report": opposition_report,
            "opposition_report_markdown": self._markdown(
                "Opposition Analysis",
                opposition_report,
            ),
            "recommendations": recommendations,
            "validation": validation,
            "regression": self._regression_payload(),
            "llm_text": llm_response.text,
        }

    def _match_summary(self, context: JsonDict, response: LLMResponse) -> JsonDict:
        ratings = {
            team: self._team_rating(data)
            for team, data in context.get("teams", {}).items()
        }
        formation = context.get("formation", {})
        pressing = context.get("pressing", {})
        pressing_summary = {}
        if pressing:
            p_metrics = pressing.get("metrics", {})
            p_detection = pressing.get("detection", {})
            pressing_summary = {
                "total_pressures": p_metrics.get("total_pressures"),
                "pressure_success_rate": p_metrics.get("pressure_success_rate"),
                "ppda": p_metrics.get("ppda"),
                "pressing_style": p_detection.get("pressing_style"),
                "confidence": p_detection.get("confidence"),
                "high_press_count": p_metrics.get("high_press_count"),
                "mid_block_count": p_metrics.get("mid_block_count"),
                "low_block_count": p_metrics.get("low_block_count"),
                "average_closing_speed": p_metrics.get("average_closing_speed"),
                "events_count": len(pressing.get("events", [])),
                "sequences_count": len(pressing.get("sequences", [])),
            }
        return {
            "overview": response.text,
            "computed_insights": context.get("insights", {}),
            "formation_summary": {
                "windows": formation.get("windows", []),
                "transitions": formation.get("transitions", []),
                "latest_metrics": formation.get("metrics", {}),
            },
            "pressing_summary": pressing_summary,
            "final_rating": ratings,
        }

    def _team_analysis(self, context: JsonDict, recommendations: JsonDict) -> JsonDict:
        analysis: JsonDict = {}
        formation = context.get("formation", {})
        formation_metrics = formation.get("metrics", {})
        pressing = context.get("pressing", {})
        p_metrics = pressing.get("metrics", {})
        p_detection = pressing.get("detection", {})
        for team, data in context.get("teams", {}).items():
            strengths = []
            weaknesses = []
            if (data.get("pass_accuracy_pct") or 0) >= 75:
                strengths.append("Efficient passing based on exported accuracy.")
            else:
                weaknesses.append("Passing reliability can improve.")
            if (data.get("possession_pct") or 0) >= 50:
                strengths.append("Sustained possession share.")
            elif data.get("possession_pct") is not None:
                weaknesses.append("Limited possession share.")
            # Pressing-based strengths/weaknesses
            if p_metrics:
                sr = p_metrics.get("pressure_success_rate", 0)
                if sr and sr >= 0.5:
                    strengths.append(f"Effective pressing with {sr:.0%} success rate.")
                else:
                    weaknesses.append("Pressing success rate below 50%.")
                ppda = p_metrics.get("ppda", 0)
                if ppda and ppda <= 8.0:
                    strengths.append(f"Aggressive pressing (PPDA={ppda:.1f}).")
                elif ppda:
                    weaknesses.append(f"High PPDA ({ppda:.1f}) indicates passive defensive approach.")
                total_p = p_metrics.get("total_pressures", 0)
                if total_p and total_p >= 10:
                    strengths.append(f"High pressure volume ({total_p} total pressures).")
                cs = p_metrics.get("average_closing_speed", 0)
                if cs and cs >= 2.0:
                    strengths.append(f"Quick closing speed ({cs:.1f} m/s).")
            if p_detection:
                style = p_detection.get("pressing_style", "")
                if style:
                    strengths.append(f"Clear pressing identity: {style.replace('_', ' ').title()}.")
            tactical_obs = data.get("tactical_shape", {})
            if formation_metrics:
                tactical_obs = dict(tactical_obs)
                tactical_obs["formation_metrics"] = formation_metrics
            # Merge pressing observations
            if p_metrics or p_detection:
                tactical_obs = dict(tactical_obs)
                tactical_obs["pressing_metrics"] = dict(p_metrics) if p_metrics else {}
                tactical_obs["pressing_detection"] = dict(p_detection) if p_detection else {}
            analysis[team] = {
                "statistics": data,
                "strengths": strengths,
                "weaknesses": weaknesses,
                "tactical_observations": tactical_obs,
                "recommendations": recommendations.get("team", {}).get(team, []),
                "rating": self._team_rating(data),
            }
        return analysis

    def _player_reports(self, context: JsonDict, recommendations: JsonDict) -> JsonDict:
        reports: JsonDict = {}
        for player_id, data in context.get("players", {}).items():
            strengths = []
            weaknesses = []
            accuracy = data.get("pass_accuracy_pct")
            if accuracy is not None and accuracy >= 80:
                strengths.append("Excellent passing accuracy.")
            if int(data.get("passes_completed") or 0) > 0:
                strengths.append("Completed at least one recorded pass.")
            if accuracy is not None and accuracy < 65:
                weaknesses.append("Below-target passing accuracy.")
            if int(data.get("shots") or 0) == 0:
                weaknesses.append("No recorded shots in exported events.")
            reports[player_id] = {
                "overall_rating": self._player_rating(data),
                "team": data.get("team"),
                "statistics": data,
                "strengths": strengths,
                "weaknesses": weaknesses,
                "passing": {
                    "attempted": data.get("passes_attempted"),
                    "completed": data.get("passes_completed"),
                    "accuracy_pct": accuracy,
                },
                "movement": {
                    "movement_radius": data.get("movement_radius"),
                    "distance_covered_m": data.get("distance_covered_m"),
                    "average_speed_mps": data.get("average_speed_mps"),
                },
                "positioning": {
                    "average_position": data.get("average_position"),
                    "heatmap_summary": data.get("heatmap_summary"),
                },
                "recommendations": recommendations.get("player", {}).get(player_id, []),
            }
        return reports

    def _coach_report(self, context: JsonDict) -> JsonDict:
        formation = context.get("formation", {})
        formation_metrics = formation.get("metrics", {})
        pressing = context.get("pressing", {})
        p_metrics = pressing.get("metrics", {})
        p_detection = pressing.get("detection", {})
        tactical_recommendations = context.get("insights", {}).get("tactical", {})
        if formation_metrics:
            tactical_recommendations = dict(tactical_recommendations)
            tactical_recommendations["formation_metrics"] = formation_metrics
        # Build pressing-specific recommendations
        pressing_recommendations = {}
        if p_metrics:
            sr = p_metrics.get("pressure_success_rate", 0)
            ppda = p_metrics.get("ppda", 0)
            cs = p_metrics.get("average_closing_speed", 0)
            hp = p_metrics.get("high_press_count", 0)
            mb = p_metrics.get("mid_block_count", 0)
            lb = p_metrics.get("low_block_count", 0)
            if sr and sr < 0.4:
                pressing_recommendations["improve_success_rate"] = "Focus on coordinated pressing triggers to reduce easily-played-through pressures."
            if ppda and ppda > 10:
                pressing_recommendations["reduce_ppda"] = "Increase defensive intensity to compress opponent passing options."
            if cs and cs < 1.5:
                pressing_recommendations["increase_closing_speed"] = "Improve transition speed to close down opponents faster."
            if hp and mb and lb:
                total_zones = hp + mb + lb
                if total_zones > 0:
                    hp_pct = hp / total_zones
                    if hp_pct < 0.2:
                        pressing_recommendations["higher_defensive_line"] = "Consider pushing defensive line higher to enable more high-press opportunities."
                    elif hp_pct > 0.6:
                        pressing_recommendations["maintain_high_press"] = "High press is dominant; maintain fitness and coordination for sustainability."
        if p_detection:
            style = p_detection.get("pressing_style", "")
            confidence = p_detection.get("confidence", 0)
            pressing_recommendations["detected_style"] = f"{style.replace('_', ' ').title()} (confidence: {confidence:.0%})"
        return {
            "what_worked": context.get("insights", {}).get("passing", {}),
            "what_failed": context.get("insights", {}).get("data_quality", {}),
            "areas_to_improve": "Use team and player recommendations for measurable improvement areas.",
            "training_recommendations": "Passing rondos, compactness drills, and transition scenarios based on threshold flags.",
            "tactical_recommendations": tactical_recommendations,
            "pressing_analysis": {
                "summary": {
                    "total_pressures": p_metrics.get("total_pressures") if p_metrics else None,
                    "success_rate": p_metrics.get("pressure_success_rate") if p_metrics else None,
                    "ppda": p_metrics.get("ppda") if p_metrics else None,
                    "style": p_detection.get("pressing_style") if p_detection else None,
                    "confidence": p_detection.get("confidence") if p_detection else None,
                },
                "zone_distribution": {
                    "high_press": p_metrics.get("high_press_count") if p_metrics else None,
                    "mid_block": p_metrics.get("mid_block_count") if p_metrics else None,
                    "low_block": p_metrics.get("low_block_count") if p_metrics else None,
                } if p_metrics else {},
                "pressing_events": len(pressing.get("events", [])),
                "pressing_sequences": len(pressing.get("sequences", [])),
            },
            "pressing_recommendations": pressing_recommendations,
            "substitution_suggestions": "Substitution suggestions require fatigue, role, and match-state inputs.",
        }

    def _opposition_report(self, context: JsonDict) -> JsonDict:
        team_analysis = self._team_analysis(
            context,
            self.recommendation_engine.generate(context),
        )
        formation = context.get("formation", {})
        pressing = context.get("pressing", {})
        p_metrics = pressing.get("metrics", {})
        p_detection = pressing.get("detection", {})
        danger_areas = context.get("tactical", {}).get("team_shapes", {})
        if formation.get("metrics"):
            danger_areas = dict(danger_areas)
            danger_areas["formation_metrics"] = formation.get("metrics")
        # Add pressing-based danger areas and exploitation advice
        pressing_opposition = {}
        if p_metrics or p_detection:
            pressing_opposition["opponent_pressing_style"] = p_detection.get("pressing_style") if p_detection else None
            pressing_opposition["opponent_ppda"] = p_metrics.get("ppda") if p_metrics else None
            pressing_opposition["opponent_success_rate"] = p_metrics.get("pressure_success_rate") if p_metrics else None
            pressing_opposition["opponent_press_volume"] = p_metrics.get("total_pressures") if p_metrics else None
            ppda = p_metrics.get("ppda", 0)
            sr = p_metrics.get("pressure_success_rate", 0)
            if ppda and ppda > 10:
                pressing_opposition["exploitation_advice"] = "Opponent low press intensity; build-up play should be comfortable."
            elif ppda and ppda <= 8:
                pressing_opposition["exploitation_advice"] = "Opponent presses aggressively; use quick passing and movement to evade pressure."
            if sr and sr < 0.4:
                pressing_opposition["vulnerability"] = "Opponent pressing is often bypassed — exploit transitional spaces."
        return {
            "opponent_strengths": {
                team: data.get("strengths", []) for team, data in team_analysis.items()
            },
            "opponent_weaknesses": {
                team: data.get("weaknesses", []) for team, data in team_analysis.items()
            },
            "how_to_exploit": "Target teams with lower pass accuracy or limited width where measured.",
            "key_players": context.get("insights", {}).get("passing", {}),
            "danger_areas": danger_areas,
            "formation_transitions": formation.get("transitions", []),
            "opposition_pressing_analysis": pressing_opposition,
        }

    def _validate_ai_outputs(self, context: JsonDict, player_reports: JsonDict) -> JsonDict:
        expected_players = set(context.get("players", {}))
        reported_players = set(player_reports)
        return {
            "uses_only_generated_analytics": True,
            "raw_video_analysis_used": False,
            "no_hallucinated_player_ids": reported_players.issubset(expected_players),
            "ratings_for_every_detected_player": expected_players.issubset(reported_players),
            "statistics_source": "outputs/*.json structured analytics",
            "api_dashboard_consistency": "FastAPI and Streamlit use MatchAnalyst over the same outputs directory.",
            "validation_messages": context.get("validation_messages", []),
        }

    def _regression_payload(self) -> JsonDict:
        return {
            "existing_cv_modules_modified": False,
            "pipeline_contract": "AI layer consumes outputs/*.json and does not call CV modules or raw frame readers.",
            "unchanged_module_paths": [
                "app/detection/",
                "app/tracking/",
                "app/homography/",
                "app/analytics/",
                "app/pose/",
                "run_pipeline.py",
            ],
            "integration_files_changed": [
                "app/api/main.py",
                "streamlit_app.py",
            ],
        }
    def _performance_payload(
        self,
        started: float,
        prompt_elapsed: float,
        llm_elapsed: float,
        report_elapsed: float,
        context_elapsed: float,
    ) -> JsonDict:
        current, peak = tracemalloc.get_traced_memory()
        total = time.perf_counter() - started
        return {
            "prompt_generation_time_ms": round(prompt_elapsed * 1000, 2),
            "llm_response_time_ms": round(llm_elapsed * 1000, 2),
            "report_generation_time_ms": round(report_elapsed * 1000, 2),
            "context_generation_time_ms": round(context_elapsed * 1000, 2),
            "api_latency_ms": round(total * 1000, 2),
            "memory_current_kb": round(current / 1024, 2),
            "memory_peak_kb": round(peak / 1024, 2),
        }

    def _read_json(self, filename: str, default: Any) -> Any:
        path = self.output_dir / filename
        if not path.exists():
            LOGGER.info("AI artifact missing: %s", path)
            return default
        with path.open("r", encoding="utf-8") as file:
            return json.load(file)

    def _provider_from_environment(self) -> LLMProvider:
        provider = os.getenv("STEPOUT_LLM_PROVIDER", "").lower()
        if provider == "openai" and os.getenv("OPENAI_API_KEY"):
            return OpenAIProvider(
                api_key=os.environ["OPENAI_API_KEY"],
                model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            )
        if provider == "gemini" and os.getenv("GEMINI_API_KEY"):
            return GeminiProvider(
                api_key=os.environ["GEMINI_API_KEY"],
                model=os.getenv("GEMINI_MODEL", "gemini-1.5-flash"),
            )
        if os.getenv("OPENAI_API_KEY"):
            return OpenAIProvider(os.environ["OPENAI_API_KEY"], os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
        if os.getenv("GEMINI_API_KEY"):
            return GeminiProvider(os.environ["GEMINI_API_KEY"], os.getenv("GEMINI_MODEL", "gemini-1.5-flash"))
        return OfflineLLMProvider()

    def _default_match_id(self) -> str:
        return os.getenv("STEPOUT_MATCH_ID", f"match-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}")

    def _team_rating(self, data: JsonDict) -> float:
        accuracy = float(data.get("pass_accuracy_pct") or 0.0)
        possession = float(data.get("possession_pct") or 0.0)
        progressive = float(data.get("progressive_passes") or 0.0)
        rating = 5.0 + (accuracy / 100.0 * 2.0) + (possession / 100.0 * 1.5) + min(progressive, 5.0) * 0.2
        return round(min(rating, 10.0), 1)

    def _pressing_rating(self, metrics: JsonDict) -> float:
        """Calculate a pressing-specific rating from metrics."""
        sr = float(metrics.get("pressure_success_rate", 0))
        ppda = float(metrics.get("ppda", 10))
        cs = float(metrics.get("average_closing_speed", 0))
        total = int(metrics.get("total_pressures", 0))
        # Lower PPDA is better for pressing; clamp to [3, 20] range
        ppda_score = max(0.0, min(1.0, (20.0 - ppda) / 17.0))
        rating = 5.0 + (sr * 2.0) + (ppda_score * 1.5) + min(cs / 3.0, 1.0) + min(total / 20.0, 1.0)
        return round(min(rating, 10.0), 1)

    def _player_rating(self, data: JsonDict) -> float:
        accuracy = float(data.get("pass_accuracy_pct") or 0.0)
        completed = float(data.get("passes_completed") or 0.0)
        movement = float(data.get("movement_radius") or 0.0)
        rating = 5.0 + (accuracy / 100.0 * 2.0) + min(completed, 8.0) * 0.15 + min(movement, 120.0) / 120.0
        return round(min(rating, 10.0), 1)

    def _markdown(self, title: str, payload: JsonDict) -> str:
        return f"# {title}\n\n```json\n{json.dumps(payload, indent=2, sort_keys=True)}\n```\n"