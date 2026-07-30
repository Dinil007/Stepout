"""
Multi-Match Football Intelligence Platform - Season Aggregation Engine

Aggregates match data into season statistics:
- Season summaries
- Player season stats
- Team season stats
- Trend analysis
- Player progression
- Team intelligence
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

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

logger = logging.getLogger(__name__)


@dataclass
class SeasonConfig:
    """Configuration for season analysis."""
    season: str
    competition: str
    output_dir: Path
    rolling_windows: List[int] = None

    def __post_init__(self):
        if self.rolling_windows is None:
            self.rolling_windows = [5, 10]


class SeasonAggregationEngine:
    """Aggregate match data into season-long statistics."""

    def __init__(self, db_path: Path, config: Optional[SeasonConfig] = None):
        self.db = SeasonDatabase(db_path)
        self.config = config or SeasonConfig(season="2024-25", competition="EPL", output_dir=Path("outputs/season"))
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

        self.trend_analyzer = TrendAnalyzer()
        self.player_development = PlayerDevelopmentTracker()
        self.team_intelligence = TeamIntelligenceAnalyzer()

    def add_match_to_season(self, match: MatchRecord, match_stats: Dict[str, Any]) -> None:
        """Add a completed match to the season database.

        Args:
            match: MatchRecord with metadata
            match_stats: Dictionary containing all match analytics
        """
        # Register match
        self.db.matches.add_match(match)

        # Update player statistics
        for player_stats in match_stats.get("player_stats", []):
            player_id = player_stats.get("player_id")
            if player_id:
                if player_id not in self.db.players.players:
                    self.db.players.players[player_id] = PlayerRecord(
                        player_id=player_id,
                        team_id=player_stats.get("team_id", ""),
                        team_name=player_stats.get("team_name", ""),
                        position=player_stats.get("position", "Unknown"),
                    )
                self.db.players.add_match_stats(player_id, player_stats)

        # Update team statistics
        for team_id, team_stats in match_stats.get("team_stats", {}).items():
            if team_id not in self.db.teams.teams:
                self.db.teams.teams[team_id] = TeamRecord(
                    team_id=team_id,
                    team_name=team_stats.get("team_name", ""),
                    competition=self.config.competition,
                    season=self.config.season,
                )
            self.db.teams.add_match_stats(team_id, team_stats)

    def compute_season_summary(self) -> Dict[str, Any]:
        """Compute overall season summary.

        Returns:
            Season summary statistics
        """
        matches = self.db.matches.get_matches_by_season(self.config.season)
        completed_matches = [m for m in matches if m.processing_status == "completed"]

        total_goals = sum(m.metadata.get("goals", 0) for m in completed_matches)
        total_matches = len(completed_matches)

        return {
            "season": self.config.season,
            "competition": self.config.competition,
            "total_matches_processed": total_matches,
            "total_goals": total_goals,
            "avg_goals_per_match": round(total_goals / max(total_matches, 1), 2),
            "teams_tracked": len(self.db.teams.teams),
            "players_tracked": len(self.db.players.players),
        }

    def compute_player_season_stats(self) -> Dict[str, Any]:
        """Compute player season statistics.

        Returns:
            Dictionary mapping player_id to season stats
        """
        return {pid: player.to_dict() for pid, player in self.db.players.players.items()}

    def compute_team_season_stats(self) -> Dict[str, Any]:
        """Compute team season statistics.

        Returns:
            Dictionary mapping team_id to season stats
        """
        return {tid: team.to_dict() for tid, team in self.db.teams.teams.items()}

    def compute_trend_analysis(self, team_id: str) -> Dict[str, Any]:
        """Compute trend analysis for a team.

        Args:
            team_id: Team identifier

        Returns:
            Trend analysis report
        """
        matches = self.db.matches.get_matches_by_team(team_id)
        match_data = []

        for match in matches:
            if match.processing_status == "completed":
                # Load match analytics
                analytics_file = Path(match.output_dir) / "analytics.json"
                if analytics_file.exists():
                    with open(analytics_file, "r") as f:
                        analytics = json.load(f)
                        match_data.append({
                            "date": match.date,
                            "opponent": match.away_team if match.home_team == team_id else match.home_team,
                            "venue": "Home" if match.home_team == team_id else "Away",
                            **analytics.get("team_stats", {}).get(team_id, {})
                        })

        return self.trend_analyzer.generate_trend_report(match_data)

    def compute_player_progression(self, player_id: str) -> Dict[str, Any]:
        """Compute player development progression.

        Args:
            player_id: Player identifier

        Returns:
            Player development report
        """
        player = self.db.players.players.get(player_id)
        if not player:
            return {"error": f"Player {player_id} not found"}

        # Gather match data
        match_ratings = [{"date": m.get("date", ""), "rating": r} for m, r in zip(player.match_stats, player.match_ratings)]
        match_speeds = [{"date": m.get("date", ""), "max_speed_kmh": m.get("max_speed_kmh", 0), "avg_speed_kmh": m.get("avg_speed_kmh", 0)} for m in player.match_stats]
        match_tactics = [{"date": m.get("date", ""), "pass_accuracy_pct": m.get("pass_accuracy_pct", 0), "possession_pct": m.get("possession_pct", 0)} for m in player.match_stats]
        match_shooting = [{"date": m.get("date", ""), "shot_accuracy_pct": m.get("shot_accuracy_pct", 0), "xg": m.get("xg", 0), "shots": m.get("shots", 0)} for m in player.match_stats]

        player_data = {
            "player_id": player_id,
            "player_name": player.team_name,
            "position": player.position,
            "matches_played": player.matches_played,
            "match_ratings": match_ratings,
            "match_speeds": match_speeds,
            "match_tactics": match_tactics,
            "match_shooting": match_shooting,
        }

        return self.player_development.generate_player_development_report(player_data)

    def compute_team_intelligence(self, team_id: str) -> Dict[str, Any]:
        """Compute team intelligence report.

        Args:
            team_id: Team identifier

        Returns:
            Team intelligence report
        """
        team = self.db.teams.teams.get(team_id)
        if not team:
            return {"error": f"Team {team_id} not found"}

        # Gather match data
        matches = self.db.matches.get_matches_by_team(team_id)
        match_data = []

        for match in matches:
            if match.processing_status == "completed":
                analytics_file = Path(match.output_dir) / "analytics.json"
                if analytics_file.exists():
                    with open(analytics_file, "r") as f:
                        analytics = json.load(f)
                        match_data.append({
                            "date": match.date,
                            "opponent": match.away_team if match.home_team == team_id else match.home_team,
                            "venue": "Home" if match.home_team == team_id else "Away",
                            **analytics.get("team_stats", {}).get(team_id, {})
                        })

        # Load formation history
        formation_history = team.formation_history

        team_data = {
            "team_id": team_id,
            "team_name": team.team_name,
            "competition": team.competition,
            "season": team.season,
            "match_data": match_data,
            "formation_history": formation_history,
        }

        return self.team_intelligence.generate_team_intelligence_report(team_data)

    def validate_season_stats(self) -> List[str]:
        """Validate that season statistics equal sum of individual matches.

        Returns:
            List of validation errors
        """
        errors = []

        # Validate player stats
        for player_id, player in self.db.players.players.items():
            computed_totals = {
                "goals": sum(m.get("goals", 0) for m in player.match_stats),
                "assists": sum(m.get("assists", 0) for m in player.match_stats),
                "shots": sum(m.get("shots", 0) for m in player.match_stats),
                "distance_m": sum(m.get("distance_m", 0) for m in player.match_stats),
            }
            for key, computed in computed_totals.items():
                stored = getattr(player, key)
                if abs(stored - computed) > 0.01:
                    errors.append(f"Player {player_id}: {key} mismatch (stored={stored}, computed={computed})")

        # Validate team stats
        for team_id, team in self.db.teams.teams.items():
            matches = self.db.matches.get_matches_by_team(team_id)
            computed_wins = sum(1 for m in matches if m.metadata.get("result") == "win")
            if abs(team.wins - computed_wins) > 0:
                errors.append(f"Team {team_id}: wins mismatch (stored={team.wins}, computed={computed_wins})")

        return errors

    def generate_all_reports(self) -> None:
        """Generate all season reports."""
        logger.info("Generating season reports...")

        # Season summary
        season_summary = self.compute_season_summary()
        with open(self.config.output_dir / "season_summary.json", "w") as f:
            json.dump(season_summary, f, indent=4)

        # Player season stats
        player_stats = self.compute_player_season_stats()
        with open(self.config.output_dir / "player_season_stats.json", "w") as f:
            json.dump(player_stats, f, indent=4)

        # Team season stats
        team_stats = self.compute_team_season_stats()
        with open(self.config.output_dir / "team_season_stats.json", "w") as f:
            json.dump(team_stats, f, indent=4)

        # Trend analysis (for each team)
        trend_reports = {}
        for team_id in self.db.teams.teams:
            trend_reports[team_id] = self.compute_trend_analysis(team_id)
        with open(self.config.output_dir / "trend_analysis.json", "w") as f:
            json.dump(trend_reports, f, indent=4)

        # Player progression (for each player)
        progression_reports = {}
        for player_id in self.db.players.players:
            progression_reports[player_id] = self.compute_player_progression(player_id)
        with open(self.config.output_dir / "player_progression.json", "w") as f:
            json.dump(progression_reports, f, indent=4)

        # Team intelligence (for each team)
        team_intelligence_reports = {}
        for team_id in self.db.teams.teams:
            team_intelligence_reports[team_id] = self.compute_team_intelligence(team_id)
        with open(self.config.output_dir / "team_intelligence.json", "w") as f:
            json.dump(team_intelligence_reports, f, indent=4)

        # Validate
        errors = self.validate_season_stats()
        if errors:
            logger.warning(f"Season validation found {len(errors)} errors")
            with open(self.config.output_dir / "season_validation_errors.json", "w") as f:
                json.dump(errors, f, indent=4)

        logger.info("Season reports generated successfully.")