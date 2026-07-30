"""
Football Intelligence Engine

Transforms raw analytics into actionable insights for coaches, scouts, and analysts.

Modules:
1. Player Performance Engine
2. Player Rating
3. Team Insights
4. Player Comparison
5. Match Summary

All inputs come from existing pipeline outputs (JSON files).
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from app.homography.field_config import FIELD_LENGTH_METERS, FIELD_WIDTH_METERS

logger = logging.getLogger(__name__)


class IntelligenceEngine:
    """Generate football intelligence from pipeline outputs."""

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.data: Dict[str, Any] = {}

    def load_data(self) -> None:
        """Load all required JSON files."""
        files = [
            "analytics.json",
            "player_statistics.csv",
            "team_statistics.csv",
            "pass_events.json",
            "shot_events.json",
            "ball_possession.json",
            "team_passing_summary.json",
            "average_positions.json",
            "player_heatmaps.json",
            "team_heatmap.json",
            "pass_network.json",
            "team_shape.json",
            "possession_summary.json",
            "territory_control.json",
            "pressing_metrics.json",
        ]
        for fname in files:
            path = self.output_dir / fname
            if path.exists():
                try:
                    if fname.endswith(".json"):
                        with open(path, "r") as f:
                            self.data[fname.replace(".json", "")] = json.load(f)
                    else:
                        # CSV handled separately
                        pass
                except Exception as e:
                    logger.warning(f"Failed to load {path}: {e}")

    # ============================================================
    # MODULE 1 - PLAYER PERFORMANCE ENGINE
    # ============================================================
    def compute_player_performance(self) -> Dict[int, Dict]:
        """Compute per-player performance metrics."""
        performance: Dict[int, Dict] = {}

        # Load player statistics CSV if exists
        csv_path = self.output_dir / "player_statistics.csv"
        if not csv_path.exists():
            return performance

        import csv
        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                tid = int(row.get("track_id", -1))
                if tid < 0:
                    continue

                performance[tid] = {
                    "track_id": tid,
                    "distance_m": float(row.get("total_distance_meters", 0)),
                    "avg_speed_kmh": float(row.get("avg_speed_kmh", 0)),
                    "max_speed_kmh": float(row.get("max_speed_kmh", 0)),
                    "sprint_count": int(row.get("sprint_count", 0)),
                    "high_intensity_distance_m": float(row.get("high_intensity_distance_m", 0)),
                    "acceleration_count": int(row.get("acceleration_count", 0)),
                    "deceleration_count": int(row.get("deceleration_count", 0)),
                    "work_rate_m_per_min": float(row.get("work_rate_m_per_min", 0)),
                    "activity_index": float(row.get("activity_index", 0)),
                    "ball_touches": int(row.get("ball_touches", 0)),
                    "pass_accuracy_pct": float(row.get("pass_accuracy_pct", 0)),
                    "shot_accuracy_pct": float(row.get("shot_accuracy_pct", 0)),
                    "xg": float(row.get("xg", 0)),
                    "xa": float(row.get("xa", 0)),
                    "xt": float(row.get("xt", 0)),
                    "defensive_actions": int(row.get("defensive_actions", 0)),
                    "heatmap_coverage_pct": float(row.get("heatmap_coverage_pct", 0)),
                }

        return performance

    # ============================================================
    # MODULE 2 - PLAYER RATING
    # ============================================================
    def compute_player_ratings(self, performance: Dict[int, Dict]) -> Dict[int, float]:
        """Compute overall player rating (0-10) using weighted metrics."""
        ratings: Dict[int, float] = {}

        # Weights (must sum to 1.0)
        W = {
            "passing": 0.20,
            "defensive": 0.20,
            "xg_xa_xt": 0.20,
            "movement": 0.20,
            "possession": 0.20,
        }

        # Helper: normalize to 0-1
        def norm(val, max_val):
            return min(max(val / max_val, 0), 1)

        for tid, p in performance.items():
            # Passing component
            pass_score = norm(p.get("pass_accuracy_pct", 0), 100)

            # Defensive component
            def_score = norm(p.get("defensive_actions", 0), 20)

            # xG/xA/xT component
            xg_score = norm(p.get("xg", 0), 0.5)
            xa_score = norm(p.get("xa", 0), 0.5)
            xt_score = norm(p.get("xt", 0), 2.0)
            xg_xa_xt_score = (xg_score + xa_score + xt_score) / 3

            # Movement component
            dist_score = norm(p.get("distance_m", 0), 12000)
            speed_score = norm(p.get("max_speed_kmh", 0), 35)
            sprint_score = norm(p.get("sprint_count", 0), 20)
            movement_score = (dist_score + speed_score + sprint_score) / 3

            # Possession contribution
            poss_score = norm(p.get("ball_touches", 0), 100)

            # Weighted sum → 0-10
            rating = (
                W["passing"] * pass_score
                + W["defensive"] * def_score
                + W["xg_xa_xt"] * xg_xa_xt_score
                + W["movement"] * movement_score
                + W["possession"] * poss_score
            ) * 10

            ratings[tid] = round(rating, 2)

        return ratings

    # ============================================================
    # MODULE 3 - TEAM INSIGHTS
    # ============================================================
    def compute_team_insights(self) -> Dict[str, Any]:
        """Detect team strengths, weaknesses, and tactical patterns."""
        insights: Dict[str, Any] = {}

        # Pass style
        pass_summary = self.data.get("team_passing_summary", {})
        total_passes = pass_summary.get("total_passes", 0)
        long_ball_pct = 0.0
        short_pass_pct = 0.0
        if total_passes > 0:
            long_ball_pct = pass_summary.get("long_balls", 0) / total_passes * 100
            short_pass_pct = pass_summary.get("short_passes", 0) / total_passes * 100

        # Territory
        territory = self.data.get("territory_control", {})
        attacking_third_pct = 0.0
        final_third_entries = 0
        for team_id, data in territory.items():
            attacking_third_pct = max(attacking_third_pct, data.get("attacking_third_pct", 0))
            final_third_entries = max(final_third_entries, data.get("final_third_entries", 0))

        # Possession
        possession = self.data.get("possession_summary", {})
        possession_pct = possession.get("possession_pct", {})

        # Determine strongest attacking side (team with more attacking third occupancy)
        strongest_attacking_side = "Unknown"
        max_att_pct = 0
        for team_id, data in territory.items():
            pct = data.get("attacking_third_pct", 0)
            if pct > max_att_pct:
                max_att_pct = pct
                strongest_attacking_side = str(team_id)

        # Weakest defensive side (team with more defensive third occupancy)
        weakest_defensive_side = "Unknown"
        max_def_pct = 0
        for team_id, data in territory.items():
            pct = data.get("defensive_third_pct", 0)
            if pct > max_def_pct:
                max_def_pct = pct
                weakest_defensive_side = str(team_id)

        # Build-up style
        build_up_style = "Long Ball" if long_ball_pct > 50 else "Short Passing"

        # Counter attacks (approximated from possession chains)
        avg_chain = possession.get("avg_possession_duration_frames", 0)
        counter_attack_freq = "Low" if avg_chain > 60 else "High"

        # Width utilization
        team_shape = self.data.get("team_shape", {})
        width_util = "Low"
        for team_id, shape in team_shape.items():
            if shape.get("avg_width_m", 0) > 50:
                width_util = "High"
                break

        # Central play %
        central_play_pct = 50.0  # Placeholder - would require pass lane analysis

        # Dangerous possessions (xG > 0.3)
        dangerous_possessions = 0  # Placeholder

        insights = {
            "strongest_attacking_side": strongest_attacking_side,
            "weakest_defensive_side": weakest_defensive_side,
            "build_up_style": build_up_style,
            "long_ball_pct": round(long_ball_pct, 2),
            "short_pass_pct": round(short_pass_pct, 2),
            "counter_attack_frequency": counter_attack_freq,
            "width_utilization": width_util,
            "central_play_pct": round(central_play_pct, 2),
            "final_third_entries": final_third_entries,
            "dangerous_possessions": dangerous_possessions,
            "possession_pct": possession_pct,
        }

        return insights

    # ============================================================
    # MODULE 4 - PLAYER COMPARISON
    # ============================================================
    def compute_player_comparison(self, performance: Dict[int, Dict], ratings: Dict[int, float]) -> Dict[str, Any]:
        """Generate percentile rankings for players."""
        if not performance:
            return {}

        df_data = []
        for tid, p in performance.items():
            df_data.append({
                "track_id": tid,
                "distance_m": p.get("distance_m", 0),
                "max_speed_kmh": p.get("max_speed_kmh", 0),
                "pass_accuracy_pct": p.get("pass_accuracy_pct", 0),
                "defensive_actions": p.get("defensive_actions", 0),
                "xg": p.get("xg", 0),
                "xa": p.get("xa", 0),
                "xt": p.get("xt", 0),
                "overall_rating": ratings.get(tid, 0),
            })

        if not df_data:
            return {}

        # Compute percentiles
        percentiles: Dict[str, Dict] = {}
        metrics = ["distance_m", "max_speed_kmh", "pass_accuracy_pct", "defensive_actions", "xg", "xa", "xt", "overall_rating"]

        for metric in metrics:
            values = [d[metric] for d in df_data]
            if not values:
                continue
            sorted_vals = sorted(values)
            pct_map = {}
            for d in df_data:
                tid = d["track_id"]
                val = d[metric]
                # Percentile = % of players with value <= this value
                pct = sum(1 for v in sorted_vals if v <= val) / len(sorted_vals) * 100
                pct_map[tid] = round(pct, 2)
            percentiles[metric] = pct_map

        return {
            "percentiles": percentiles,
            "top_players": {
                metric: max(df_data, key=lambda x: x[metric])["track_id"]
                for metric in metrics
            }
        }

    # ============================================================
    # MODULE 5 - MATCH SUMMARY
    # ============================================================
    def compute_match_summary(self, performance: Dict[int, Dict], ratings: Dict[int, float], insights: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive match summary."""
        if not performance:
            return {}

        # Best and worst performers
        best_performer = max(ratings.items(), key=lambda x: x[1]) if ratings else (-1, 0)
        worst_performer = min(ratings.items(), key=lambda x: x[1]) if ratings else (-1, 0)

        # Key statistics
        total_distance = sum(p.get("distance_m", 0) for p in performance.values())
        avg_speed = np.mean([p.get("avg_speed_kmh", 0) for p in performance.values()])
        max_speed = max(p.get("max_speed_kmh", 0) for p in performance.values())
        total_passes = self.data.get("pass_network", {}).get("total_passes", 0)
        shot_events = self.data.get("shot_events", [])
        total_shots = len(shot_events) if isinstance(shot_events, list) else len(shot_events.get("shots", []))

        # Team strengths and weaknesses
        strengths = []
        weaknesses = []

        if insights.get("short_pass_pct", 0) > 60:
            strengths.append("Strong passing game")
        if insights.get("long_ball_pct", 0) > 50:
            weaknesses.append("Over-reliance on long balls")

        if total_distance > 100000:
            strengths.append("High work rate")
        else:
            weaknesses.append("Low distance covered")

        if max_speed > 30:
            strengths.append("Fast players")
        else:
            weaknesses.append("Lack of pace")

        summary = {
            "best_performer": {
                "track_id": best_performer[0],
                "rating": best_performer[1],
            },
            "weakest_performer": {
                "track_id": worst_performer[0],
                "rating": worst_performer[1],
            },
            "key_statistics": {
                "total_distance_m": round(total_distance, 2),
                "avg_speed_kmh": round(avg_speed, 2),
                "max_speed_kmh": round(max_speed, 2),
                "total_passes": total_passes,
                "total_shots": total_shots,
            },
            "team_strengths": strengths,
            "team_weaknesses": weaknesses,
            "tactical_summary": insights,
        }

        return summary

    # ============================================================
    # MASTER EXPORT
    # ============================================================
    def compute_all(self) -> Dict[str, Any]:
        logger.info("Computing football intelligence...")
        self.load_data()

        performance = self.compute_player_performance()
        ratings = self.compute_player_ratings(performance)
        insights = self.compute_team_insights()
        comparison = self.compute_player_comparison(performance, ratings)
        match_summary = self.compute_match_summary(performance, ratings, insights)

        result = {
            "player_performance": {str(k): v for k, v in performance.items()},
            "player_ratings": {str(k): v for k, v in ratings.items()},
            "team_insights": insights,
            "player_comparison": comparison,
            "match_summary": match_summary,
        }

        logger.info("Football intelligence computation complete.")
        return result