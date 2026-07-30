"""
Multi-Match Football Intelligence Platform - Trend Analytics

Computes rolling averages, recent form, and performance trends.
"""

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class TrendConfig:
    """Configuration for trend analysis."""
    rolling_window_last_5: int = 5
    rolling_window_last_10: int = 10
    min_matches_for_trend: int = 3


class TrendAnalyzer:
    """Analyze performance trends across multiple matches."""

    def __init__(self, config: Optional[TrendConfig] = None):
        self.config = config or TrendConfig()

    def compute_rolling_averages(self, match_data: List[Dict], windows: Optional[List[int]] = None) -> Dict[str, Any]:
        """Compute rolling averages for specified windows.

        Args:
            match_data: List of match statistics in chronological order
            windows: List of window sizes (default: [5, 10])

        Returns:
            Dictionary with rolling averages for each metric
        """
        if not match_data:
            return {}

        windows = windows or [5, 10]
        df = pd.DataFrame(match_data)

        results = {}
        for window in windows:
            if len(df) >= window:
                rolling = df.rolling(window=window, min_periods=1).mean()
                results[f"last_{window}"] = {
                    col: rolling[col].iloc[-1] if col in rolling.columns else 0.0
                    for col in df.columns
                    if df[col].dtype in [np.float64, np.int64]
                }

        return results

    def analyze_home_away_split(self, match_data: List[Dict]) -> Dict[str, Any]:
        """Analyze performance split between home and away matches.

        Args:
            match_data: List of match statistics with 'venue' field

        Returns:
            Dictionary with home/away statistics
        """
        home_matches = [m for m in match_data if m.get("venue") == "Home"]
        away_matches = [m for m in match_data if m.get("venue") == "Away"]

        def aggregate(matches: List[Dict]) -> Dict[str, float]:
            if not matches:
                return {}
            df = pd.DataFrame(matches)
            return {
                col: df[col].mean()
                for col in df.columns
                if df[col].dtype in [np.float64, np.int64]
            }

        return {
            "home": {
                "matches_played": len(home_matches),
                "averages": aggregate(home_matches),
            },
            "away": {
                "matches_played": len(away_matches),
                "averages": aggregate(away_matches),
            },
        }

    def compare_to_opponent(self, match_data: List[Dict], opponent_id: str) -> Dict[str, Any]:
        """Compare performance against specific opponent.

        Args:
            match_data: List of match statistics
            opponent_id: Opponent team identifier

        Returns:
            Dictionary with comparison metrics
        """
        opponent_matches = [m for m in match_data if m.get("opponent") == opponent_id]
        other_matches = [m for m in match_data if m.get("opponent") != opponent_id]

        def compute_avg(matches: List[Dict]) -> Dict[str, float]:
            if not matches:
                return {}
            df = pd.DataFrame(matches)
            return {
                col: df[col].mean()
                for col in df.columns
                if df[col].dtype in [np.float64, np.int64]
            }

        opponent_avg = compute_avg(opponent_matches)
        other_avg = compute_avg(other_matches)

        # Compute differential
        differential = {}
        for key in opponent_avg:
            if key in other_avg:
                differential[key] = opponent_avg[key] - other_avg[key]

        return {
            "opponent_id": opponent_id,
            "matches_played": len(opponent_matches),
            "opponent_averages": opponent_avg,
            "other_averages": other_avg,
            "differential": differential,
        }

    def detect_performance_trend(self, match_data: List[Dict], metric: str) -> Dict[str, Any]:
        """Detect trend in specific metric over time.

        Args:
            match_data: List of match statistics
            metric: Metric to analyze

        Returns:
            Dictionary with trend analysis
        """
        if len(match_data) < self.config.min_matches_for_trend:
            return {"trend": "insufficient_data", "slope": 0.0}

        values = [m.get(metric, 0) for m in match_data]
        x = np.arange(len(values))

        # Linear regression
        slope, intercept = np.polyfit(x, values, 1)

        # Classify trend
        if slope > 0.05:
            trend = "improving"
        elif slope < -0.05:
            trend = "declining"
        else:
            trend = "stable"

        return {
            "trend": trend,
            "slope": round(float(slope), 4),
            "intercept": round(float(intercept), 4),
            "start_value": values[0],
            "end_value": values[-1],
            "change_pct": round((values[-1] - values[0]) / max(abs(values[0]), 0.001) * 100, 2),
        }

    def generate_trend_report(self, match_data: List[Dict]) -> Dict[str, Any]:
        """Generate comprehensive trend report.

        Args:
            match_data: List of match statistics

        Returns:
            Comprehensive trend analysis report
        """
        if not match_data:
            return {"error": "No match data provided"}

        # Extract numeric columns
        df = pd.DataFrame(match_data)
        numeric_cols = [col for col in df.columns if df[col].dtype in [np.float64, np.int64]]

        report = {
            "total_matches": len(match_data),
            "date_range": {
                "first": match_data[0].get("date"),
                "last": match_data[-1].get("date"),
            },
            "rolling_averages": self.compute_rolling_averages(match_data),
            "home_away_split": self.analyze_home_away_split(match_data),
            "metric_trends": {},
        }

        # Analyze trends for key metrics
        key_metrics = ["xg", "xa", "xt", "possession_pct", "ppda", "distance_m", "max_speed_kmh"]
        for metric in key_metrics:
            if metric in numeric_cols:
                report["metric_trends"][metric] = self.detect_performance_trend(match_data, metric)

        return report