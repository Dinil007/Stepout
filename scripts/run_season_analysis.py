"""
Multi-Match Football Intelligence Platform - Season Analysis Pipeline

Aggregates match analytics into season-long statistics.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

from app.analytics.season_analysis.database import (
    MatchRecord,
    PlayerDatabase,
    PlayerRecord,
    SeasonDatabase,
    TeamDatabase,
    TeamRecord,
)
from app.analytics.season_analysis.player_development import PlayerDevelopmentTracker
from app.analytics.season_analysis.team_intelligence import TeamIntelligenceAnalyzer
from app.analytics.season_analysis.trend_analytics import TrendAnalyzer
from app.analytics.season_analysis.season_engine import SeasonAggregationEngine, SeasonConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("SeasonAnalysis")


def run_season_analysis(
    db_path: Path,
    output_dir: Path,
    season: str = "2024-25",
    competition: str = "EPL",
) -> Dict[str, Any]:
    """Run season-long aggregation analysis.

    Args:
        db_path: Path to season database directory
        output_dir: Path to output directory
        season: Season identifier
        competition: Competition name

    Returns:
        Summary dictionary
    """
    logger.info(f"Starting season analysis: {season} {competition}")

    # Initialize engine
    engine = SeasonAggregationEngine(
        db_path=db_path,
        config=SeasonConfig(
            season=season,
            competition=competition,
            output_dir=output_dir,
        ),
    )

    # Add sample match (in real usage, this would iterate over all matches)
    # For now, we just generate reports from existing data
    engine.generate_all_reports()

    logger.info("Season analysis complete.")

    return {
        "season": season,
        "competition": competition,
        "output_dir": str(output_dir),
        "files_generated": [
            "season_summary.json",
            "player_season_stats.json",
            "team_season_stats.json",
            "trend_analysis.json",
            "player_progression.json",
            "team_intelligence.json",
        ],
    }


if __name__ == "__main__":
    import sys

    db_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("outputs/season_db")
    output_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("outputs/season")

    run_season_analysis(db_path, output_dir)