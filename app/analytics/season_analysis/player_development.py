"""
Multi-Match Football Intelligence Platform - Player Development Tracking

Tracks player progression across matches:
- Rating progression
- Speed progression
- Tactical improvement
- Passing improvement
- Shooting improvement
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class PlayerDevelopmentTracker:
    """Track player development across matches."""

    def compute_rating_progression(self, match_ratings: List[Dict]) -> Dict[str, Any]:
        """Compute rating progression over time.

        Args:
            match_ratings: List of match ratings with dates

        Returns:
            Rating progression analysis
        """
        if not match_ratings:
            return {"trend": "no_data", "progression": []}

        df = pd.DataFrame(match_ratings)
        df = df.sort_values("date")

        ratings = df["rating"].tolist()
        dates = df["date"].tolist()

        # Linear regression
        x = np.arange(len(ratings))
        slope, intercept = np.polyfit(x, ratings, 1)

        # Classify progression
        if slope > 0.05:
            trend = "improving"
        elif slope < -0.05:
            trend = "declining"
        else:
            trend = "stable"

        return {
            "trend": trend,
            "slope": round(float(slope), 4),
            "start_rating": ratings[0],
            "current_rating": ratings[-1],
            "peak_rating": max(ratings),
            "lowest_rating": min(ratings),
            "average_rating": round(np.mean(ratings), 2),
            "progression": [{"date": d, "rating": r} for d, r in zip(dates, ratings)]
        }

    def compute_speed_progression(self, match_speeds: List[Dict]) -> Dict[str, Any]:
        """Compute speed progression over time.

        Args:
            match_speeds: List of match speed data with dates

        Returns:
            Speed progression analysis
        """
        if not match_speeds:
            return {"trend": "no_data", "progression": []}

        df = pd.DataFrame(match_speeds)
        df = df.sort_values("date")

        max_speeds = df["max_speed_kmh"].tolist()
        avg_speeds = df["avg_speed_kmh"].tolist()
        dates = df["date"].tolist()

        # Max speed trend
        x = np.arange(len(max_speeds))
        slope, _ = np.polyfit(x, max_speeds, 1)
        trend = "improving" if slope > 0.1 else "declining" if slope < -0.1 else "stable"

        return {
            "max_speed_trend": trend,
            "max_speed_slope": round(float(slope), 4),
            "current_max_speed": max_speeds[-1],
            "peak_max_speed": max(max_speeds),
            "average_max_speed": round(np.mean(max_speeds), 2),
            "progression": [
                {
                    "date": d,
                    "max_speed_kmh": ms,
                    "avg_speed_kmh": avg_s
                }
                for d, ms, avg_s in zip(dates, max_speeds, avg_speeds)
            ]
        }

    def compute_tactical_improvement(self, match_tactics: List[Dict]) -> Dict[str, Any]:
        """Compute tactical improvement over time.

        Args:
            match_tactics: List of match tactical data with dates

        Returns:
            Tactical improvement analysis
        """
        if not match_tactics:
            return {"trend": "no_data"}

        df = pd.DataFrame(match_tactics)
        df = df.sort_values("date")

        # Analyze passing accuracy trend
        if "pass_accuracy_pct" in df.columns:
            pass_acc = df["pass_accuracy_pct"].tolist()
            x = np.arange(len(pass_acc))
            slope, _ = np.polyfit(x, pass_acc, 1)
            pass_trend = "improving" if slope > 0.5 else "declining" if slope < -0.5 else "stable"
        else:
            pass_trend = "unknown"
            slope = 0.0

        # Analyze decision making (possession efficiency)
        if "possession_pct" in df.columns:
            possession = df["possession_pct"].tolist()
            avg_possession = round(np.mean(possession), 2)
        else:
            avg_possession = 0.0

        return {
            "passing_trend": pass_trend,
            "passing_improvement_slope": round(float(slope), 4),
            "average_possession_pct": avg_possession,
            "matches_analyzed": len(match_tactics)
        }

    def compute_shooting_improvement(self, match_shooting: List[Dict]) -> Dict[str, Any]:
        """Compute shooting improvement over time.

        Args:
            match_shooting: List of match shooting data with dates

        Returns:
            Shooting improvement analysis
        """
        if not match_shooting:
            return {"trend": "no_data"}

        df = pd.DataFrame(match_shooting)
        df = df.sort_values("date")

        # Analyze shot accuracy trend
        if "shot_accuracy_pct" in df.columns:
            shot_acc = df["shot_accuracy_pct"].tolist()
            x = np.arange(len(shot_acc))
            slope, _ = np.polyfit(x, shot_acc, 1)
            accuracy_trend = "improving" if slope > 0.5 else "declining" if slope < -0.5 else "stable"
        else:
            accuracy_trend = "unknown"
            slope = 0.0

        # Analyze xG conversion
        if "xg" in df.columns and "shots" in df.columns:
            xg = df["xg"].tolist()
            shots = df["shots"].tolist()
            xg_per_shot = [x/s if s > 0 else 0 for x, s in zip(xg, shots)]
            avg_xg_per_shot = round(np.mean(xg_per_shot), 4) if xg_per_shot else 0.0
        else:
            avg_xg_per_shot = 0.0

        return {
            "accuracy_trend": accuracy_trend,
            "accuracy_improvement_slope": round(float(slope), 4),
            "average_xg_per_shot": avg_xg_per_shot,
            "matches_analyzed": len(match_shooting)
        }

    def generate_player_development_report(self, player_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive player development report.

        Args:
            player_data: Dictionary containing all player match data

        Returns:
            Complete development report
        """
        report = {
            "player_id": player_data.get("player_id"),
            "player_name": player_data.get("player_name"),
            "position": player_data.get("position"),
            "matches_played": player_data.get("matches_played", 0),
            "rating_progression": self.compute_rating_progression(player_data.get("match_ratings", [])),
            "speed_progression": self.compute_speed_progression(player_data.get("match_speeds", [])),
            "tactical_improvement": self.compute_tactical_improvement(player_data.get("match_tactics", [])),
            "shooting_improvement": self.compute_shooting_improvement(player_data.get("match_shooting", [])),
        }

        # Overall development score
        scores = []
        if report["rating_progression"].get("trend") == "improving":
            scores.append(1.0)
        elif report["rating_progression"].get("trend") == "stable":
            scores.append(0.5)

        if report["speed_progression"].get("max_speed_trend") == "improving":
            scores.append(1.0)
        elif report["speed_progression"].get("max_speed_trend") == "stable":
            scores.append(0.5)

        if report["tactical_improvement"].get("passing_trend") == "improving":
            scores.append(1.0)
        elif report["tactical_improvement"].get("passing_trend") == "stable":
            scores.append(0.5)

        if report["shooting_improvement"].get("accuracy_trend") == "improving":
            scores.append(1.0)
        elif report["shooting_improvement"].get("accuracy_trend") == "stable":
            scores.append(0.5)

        report["overall_development_score"] = round(np.mean(scores), 2) if scores else 0.0

        return report