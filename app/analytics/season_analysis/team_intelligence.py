"""
Multi-Match Football Intelligence Platform - Team Intelligence

Automatically identifies team characteristics:
- Preferred formation
- Tactical evolution
- Strongest attacking pattern
- Weakest defensive pattern
- Pressing trend
- Possession trend
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class TeamIntelligenceAnalyzer:
    """Analyze team intelligence across multiple matches."""

    def identify_preferred_formation(self, formation_history: List[Dict]) -> Dict[str, Any]:
        """Identify team's preferred formation.

        Args:
            formation_history: List of formation detections with dates

        Returns:
            Preferred formation analysis
        """
        if not formation_history:
            return {"preferred_formation": "unknown", "confidence": 0.0}

        df = pd.DataFrame(formation_history)
        formation_counts = df["formation"].value_counts()

        preferred = formation_counts.index[0] if len(formation_counts) > 0 else "unknown"
        count = formation_counts.iloc[0] if len(formation_counts) > 0 else 0
        total = len(formation_history)
        confidence = count / total if total > 0 else 0.0

        return {
            "preferred_formation": preferred,
            "usage_count": count,
            "total_detections": total,
            "usage_pct": round(confidence * 100, 2),
            "confidence": round(confidence, 4),
            "all_formations": formation_counts.to_dict()
        }

    def analyze_tactical_evolution(self, match_data: List[Dict]) -> Dict[str, Any]:
        """Analyze tactical evolution over the season.

        Args:
            match_data: List of match statistics

        Returns:
            Tactical evolution analysis
        """
        if not match_data:
            return {"evolution": "no_data"}

        df = pd.DataFrame(match_data)
        df = df.sort_values("date") if "date" in df.columns else df

        # Analyze formation changes
        if "formation" in df.columns:
            formation_changes = (df["formation"] != df["formation"].shift()).sum() - 1
            formation_evolution = "stable" if formation_changes <= 2 else "evolving"
        else:
            formation_changes = 0
            formation_evolution = "unknown"

        # Analyze possession trend
        if "possession_pct" in df.columns:
            possession_trend = "improving" if df["possession_pct"].iloc[-1] > df["possession_pct"].iloc[0] else "declining" if df["possession_pct"].iloc[-1] < df["possession_pct"].iloc[0] else "stable"
        else:
            possession_trend = "unknown"

        # Analyze pressing trend
        if "ppda" in df.columns:
            ppda_trend = "improving" if df["ppda"].iloc[-1] < df["ppda"].iloc[0] else "declining" if df["ppda"].iloc[-1] > df["ppda"].iloc[0] else "stable"
        else:
            ppda_trend = "unknown"

        return {
            "formation_changes": int(formation_changes),
            "formation_evolution": formation_evolution,
            "possession_trend": possession_trend,
            "pressing_trend": ppda_trend,
            "matches_analyzed": len(match_data)
        }

    def identify_strongest_attacking_pattern(self, match_data: List[Dict]) -> Dict[str, Any]:
        """Identify team's strongest attacking pattern.

        Args:
            match_data: List of match statistics

        Returns:
            Strongest attacking pattern analysis
        """
        if not match_data:
            return {"pattern": "unknown"}

        df = pd.DataFrame(match_data)

        # Analyze xG trends
        if "xg" in df.columns:
            avg_xg = df["xg"].mean()
            xg_trend = "improving" if df["xg"].iloc[-1] > df["xg"].iloc[0] else "stable"
        else:
            avg_xg = 0.0
            xg_trend = "unknown"

        # Analyze shot patterns
        if "shots" in df.columns and "xg" in df.columns:
            shot_quality = df["xg"].sum() / max(df["shots"].sum(), 1)
        else:
            shot_quality = 0.0

        # Analyze possession-based attacking
        if "possession_pct" in df.columns and "xg" in df.columns:
            high_possession_xg = df[df["possession_pct"] > 50]["xg"].mean() if len(df[df["possession_pct"] > 50]) > 0 else 0
            counter_xg = df[df["possession_pct"] <= 50]["xg"].mean() if len(df[df["possession_pct"] <= 50]) > 0 else 0
        else:
            high_possession_xg = 0.0
            counter_xg = 0.0

        # Determine strongest pattern
        if shot_quality > 0.15:
            pattern = "High-Quality Shots"
        elif high_possession_xg > counter_xg:
            pattern = "Possession-Based"
        elif counter_xg > high_possession_xg:
            pattern = "Counter-Attacking"
        else:
            pattern = "Balanced"

        return {
            "pattern": pattern,
            "avg_xg": round(avg_xg, 4),
            "xg_trend": xg_trend,
            "shot_quality": round(shot_quality, 4),
            "high_possession_xg": round(high_possession_xg, 4),
            "counter_attack_xg": round(counter_xg, 4)
        }

    def identify_weakest_defensive_pattern(self, match_data: List[Dict]) -> Dict[str, Any]:
        """Identify team's weakest defensive pattern.

        Args:
            match_data: List of match statistics

        Returns:
            Weakest defensive pattern analysis
        """
        if not match_data:
            return {"pattern": "unknown"}

        df = pd.DataFrame(match_data)

        # Analyze goals conceded
        if "goals_conceded" in df.columns:
            avg_goals_conceded = df["goals_conceded"].mean()
            goals_trend = "improving" if df["goals_conceded"].iloc[-1] < df["goals_conceded"].iloc[0] else "declining" if df["goals_conceded"].iloc[-1] > df["goals_conceded"].iloc[0] else "stable"
        else:
            avg_goals_conceded = 0.0
            goals_trend = "unknown"

        # Analyze defensive actions
        if "defensive_actions" in df.columns:
            avg_def_actions = df["defensive_actions"].mean()
            def_trend = "improving" if df["defensive_actions"].iloc[-1] > df["defensive_actions"].iloc[0] else "declining"
        else:
            avg_def_actions = 0.0
            def_trend = "unknown"

        # Analyze PPDA (pressing efficiency)
        if "ppda" in df.columns:
            avg_ppda = df["ppda"].mean()
            ppda_trend = "improving" if df["ppda"].iloc[-1] < df["ppda"].iloc[0] else "declining"
        else:
            avg_ppda = 0.0
            ppda_trend = "unknown"

        # Determine weakest pattern
        if avg_goals_conceded > 2.0:
            pattern = "High Concession Rate"
        elif avg_def_actions < 10:
            pattern = "Low Defensive Engagement"
        elif avg_ppda > 15:
            pattern = "Poor Pressing"
        else:
            pattern = "Set Piece Vulnerability"

        return {
            "pattern": pattern,
            "avg_goals_conceded": round(avg_goals_conceded, 2),
            "goals_trend": goals_trend,
            "avg_defensive_actions": round(avg_def_actions, 2),
            "defense_trend": def_trend,
            "avg_ppda": round(avg_ppda, 2),
            "pressing_trend": ppda_trend
        }

    def analyze_pressing_trend(self, match_data: List[Dict]) -> Dict[str, Any]:
        """Analyze pressing trend over time.

        Args:
            match_data: List of match statistics

        Returns:
            Pressing trend analysis
        """
        if not match_data:
            return {"trend": "no_data"}

        df = pd.DataFrame(match_data)

        if "ppda" not in df.columns:
            return {"trend": "no_data"}

        ppda_values = df["ppda"].tolist()
        avg_ppda = np.mean(ppda_values)
        latest_ppda = ppda_values[-1] if ppda_values else 0

        # PPDA trend (lower is better)
        x = np.arange(len(ppda_values))
        slope, _ = np.polyfit(x, ppda_values, 1)

        if slope < -0.5:
            trend = "improving"
        elif slope > 0.5:
            trend = "declining"
        else:
            trend = "stable"

        return {
            "trend": trend,
            "avg_ppda": round(avg_ppda, 2),
            "latest_ppda": round(latest_ppda, 2),
            "ppda_slope": round(float(slope), 4),
            "matches_analyzed": len(match_data)
        }

    def analyze_possession_trend(self, match_data: List[Dict]) -> Dict[str, Any]:
        """Analyze possession trend over time.

        Args:
            match_data: List of match statistics

        Returns:
            Possession trend analysis
        """
        if not match_data:
            return {"trend": "no_data"}

        df = pd.DataFrame(match_data)

        if "possession_pct" not in df.columns:
            return {"trend": "no_data"}

        possession_values = df["possession_pct"].tolist()
        avg_possession = np.mean(possession_values)
        latest_possession = possession_values[-1] if possession_values else 0

        x = np.arange(len(possession_values))
        slope, _ = np.polyfit(x, possession_values, 1)

        if slope > 1.0:
            trend = "improving"
        elif slope < -1.0:
            trend = "declining"
        else:
            trend = "stable"

        return {
            "trend": trend,
            "avg_possession_pct": round(avg_possession, 2),
            "latest_possession_pct": round(latest_possession, 2),
            "possession_slope": round(float(slope), 4),
            "matches_analyzed": len(match_data)
        }

    def generate_team_intelligence_report(self, team_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive team intelligence report.

        Args:
            team_data: Dictionary containing all team data

        Returns:
            Complete team intelligence report
        """
        match_data = team_data.get("match_data", [])

        report = {
            "team_id": team_data.get("team_id"),
            "team_name": team_data.get("team_name"),
            "competition": team_data.get("competition"),
            "season": team_data.get("season"),
            "matches_analyzed": len(match_data),
            "preferred_formation": self.identify_preferred_formation(team_data.get("formation_history", [])),
            "tactical_evolution": self.analyze_tactical_evolution(match_data),
            "strongest_attacking_pattern": self.identify_strongest_attacking_pattern(match_data),
            "weakest_defensive_pattern": self.identify_weakest_defensive_pattern(match_data),
            "pressing_trend": self.analyze_pressing_trend(match_data),
            "possession_trend": self.analyze_possession_trend(match_data),
        }

        return report