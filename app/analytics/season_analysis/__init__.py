"""
Multi-Match Football Intelligence Platform - Season Analysis

Provides season-long analytics aggregation:
- Match database
- Player profiles and statistics
- Team statistics and trends
- Trend analytics
- Player development tracking
- Team intelligence
"""

from app.analytics.season_analysis.database import (
    MatchRecord,
    MatchDatabase,
    PlayerDatabase,
    PlayerRecord,
    SeasonDatabase,
    TeamDatabase,
    TeamRecord,
)
from app.analytics.season_analysis.trend_analytics import TrendAnalyzer, TrendConfig
from app.analytics.season_analysis.player_development import PlayerDevelopmentTracker
from app.analytics.season_analysis.team_intelligence import TeamIntelligenceAnalyzer
from app.analytics.season_analysis.season_engine import SeasonAggregationEngine, SeasonConfig

__all__ = [
    "MatchRecord",
    "MatchDatabase",
    "PlayerDatabase",
    "PlayerRecord",
    "SeasonDatabase",
    "TeamDatabase",
    "TeamRecord",
    "TrendAnalyzer",
    "TrendConfig",
    "PlayerDevelopmentTracker",
    "TeamIntelligenceAnalyzer",
    "SeasonAggregationEngine",
    "SeasonConfig",
]