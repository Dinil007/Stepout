"""
Multi-Match Football Intelligence Platform - Database Layer

Provides persistent storage for:
- Match metadata
- Player profiles and statistics
- Team statistics and trends
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class MatchRecord:
    """Single match record."""
    match_id: str
    competition: str
    season: str
    home_team: str
    away_team: str
    date: str
    venue: str
    video_path: str
    duration_seconds: float = 0.0
    processing_status: str = "pending"  # pending, processing, completed, failed
    analytics_version: str = "1.0.0"
    processing_time_seconds: float = 0.0
    output_dir: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "match_id": self.match_id,
            "competition": self.competition,
            "season": self.season,
            "home_team": self.home_team,
            "away_team": self.away_team,
            "date": self.date,
            "venue": self.venue,
            "video_path": self.video_path,
            "duration_seconds": self.duration_seconds,
            "processing_status": self.processing_status,
            "analytics_version": self.analytics_version,
            "processing_time_seconds": self.processing_time_seconds,
            "output_dir": self.output_dir,
            "metadata": self.metadata,
        }


@dataclass
class PlayerRecord:
    """Player profile with season statistics."""
    player_id: str
    team_id: str
    team_name: str
    position: str
    matches_played: int = 0
    minutes_played: float = 0.0
    average_rating: float = 0.0
    total_xg: float = 0.0
    total_xa: float = 0.0
    total_xt: float = 0.0
    total_distance_m: float = 0.0
    total_sprint_count: int = 0
    max_speed_kmh: float = 0.0
    goals: int = 0
    assists: int = 0
    shots: int = 0
    passes_completed: int = 0
    passes_attempted: int = 0
    defensive_actions: int = 0
    match_ratings: List[float] = field(default_factory=list)
    match_stats: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "player_id": self.player_id,
            "team_id": self.team_id,
            "team_name": self.team_name,
            "position": self.position,
            "matches_played": self.matches_played,
            "minutes_played": round(self.minutes_played, 2),
            "average_rating": round(self.average_rating, 2),
            "total_xg": round(self.total_xg, 4),
            "total_xa": round(self.total_xa, 4),
            "total_xt": round(self.total_xt, 4),
            "total_distance_m": round(self.total_distance_m, 2),
            "total_sprint_count": self.total_sprint_count,
            "max_speed_kmh": round(self.max_speed_kmh, 2),
            "goals": self.goals,
            "assists": self.assists,
            "shots": self.shots,
            "passes_completed": self.passes_completed,
            "passes_attempted": self.passes_attempted,
            "pass_accuracy_pct": round(self.passes_completed / self.passes_attempted * 100, 2) if self.passes_attempted > 0 else 0.0,
            "defensive_actions": self.defensive_actions,
        }


@dataclass
class TeamRecord:
    """Team season statistics and trends."""
    team_id: str
    team_name: str
    competition: str
    season: str
    matches_played: int = 0
    wins: int = 0
    draws: int = 0
    losses: int = 0
    goals_scored: int = 0
    goals_conceded: int = 0
    total_xg: float = 0.0
    total_xa: float = 0.0
    total_xt: float = 0.0
    possession_avg: float = 0.0
    ppda_avg: float = 0.0
    formation_history: List[Dict[str, Any]] = field(default_factory=list)
    pressing_trends: List[Dict[str, Any]] = field(default_factory=list)
    tactical_evolution: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "team_id": self.team_id,
            "team_name": self.team_name,
            "competition": self.competition,
            "season": self.season,
            "matches_played": self.matches_played,
            "wins": self.wins,
            "draws": self.draws,
            "losses": self.losses,
            "goals_scored": self.goals_scored,
            "goals_conceded": self.goals_conceded,
            "goal_difference": self.goals_scored - self.goals_conceded,
            "points": self.wins * 3 + self.draws,
            "total_xg": round(self.total_xg, 4),
            "total_xa": round(self.total_xa, 4),
            "total_xt": round(self.total_xt, 4),
            "possession_avg": round(self.possession_avg, 2),
            "ppda_avg": round(self.ppda_avg, 2),
        }


class MatchDatabase:
    """Persistent match metadata storage."""

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.matches_file = db_path / "matches.json"
        self.matches: Dict[str, MatchRecord] = {}
        self._load()

    def _load(self) -> None:
        if self.matches_file.exists():
            with open(self.matches_file, "r") as f:
                data = json.load(f)
                for m in data:
                    record = MatchRecord(
                        match_id=m["match_id"],
                        competition=m["competition"],
                        season=m["season"],
                        home_team=m["home_team"],
                        away_team=m["away_team"],
                        date=m["date"],
                        venue=m["venue"],
                        video_path=m["video_path"],
                        duration_seconds=m.get("duration_seconds", 0.0),
                        processing_status=m.get("processing_status", "pending"),
                        analytics_version=m.get("analytics_version", "1.0.0"),
                        processing_time_seconds=m.get("processing_time_seconds", 0.0),
                        output_dir=m.get("output_dir", ""),
                        metadata=m.get("metadata", {}),
                    )
                    self.matches[record.match_id] = record

    def save(self) -> None:
        data = [m.to_dict() for m in self.matches.values()]
        with open(self.matches_file, "w") as f:
            json.dump(data, f, indent=4)

    def add_match(self, match: MatchRecord) -> None:
        self.matches[match.match_id] = match
        self.save()

    def update_status(self, match_id: str, status: str) -> None:
        if match_id in self.matches:
            self.matches[match_id].processing_status = status
            self.save()

    def get_matches_by_season(self, season: str) -> List[MatchRecord]:
        return [m for m in self.matches.values() if m.season == season]

    def get_matches_by_team(self, team_name: str) -> List[MatchRecord]:
        return [m for m in self.matches.values() if m.home_team == team_name or m.away_team == team_name]


class PlayerDatabase:
    """Persistent player profile storage."""

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.players_file = db_path / "players.json"
        self.players: Dict[str, PlayerRecord] = {}
        self._load()

    def _load(self) -> None:
        if self.players_file.exists():
            with open(self.players_file, "r") as f:
                data = json.load(f)
                for p in data:
                    record = PlayerRecord(
                        player_id=p["player_id"],
                        team_id=p["team_id"],
                        team_name=p["team_name"],
                        position=p.get("position", "Unknown"),
                        matches_played=p.get("matches_played", 0),
                        minutes_played=p.get("minutes_played", 0.0),
                        average_rating=p.get("average_rating", 0.0),
                        total_xg=p.get("total_xg", 0.0),
                        total_xa=p.get("total_xa", 0.0),
                        total_xt=p.get("total_xt", 0.0),
                        total_distance_m=p.get("total_distance_m", 0.0),
                        total_sprint_count=p.get("total_sprint_count", 0),
                        max_speed_kmh=p.get("max_speed_kmh", 0.0),
                    )
                    self.players[record.player_id] = record

    def save(self) -> None:
        data = [p.to_dict() for p in self.players.values()]
        with open(self.players_file, "w") as f:
            json.dump(data, f, indent=4)

    def add_match_stats(self, player_id: str, stats: Dict[str, Any]) -> None:
        if player_id not in self.players:
            logger.warning(f"Player {player_id} not found in database")
            return
        player = self.players[player_id]
        player.matches_played += 1
        player.minutes_played += stats.get("minutes_played", 0.0)
        player.total_xg += stats.get("xg", 0.0)
        player.total_xa += stats.get("xa", 0.0)
        player.total_xt += stats.get("xt", 0.0)
        player.total_distance_m += stats.get("distance_m", 0.0)
        player.total_sprint_count += stats.get("sprint_count", 0)
        player.max_speed_kmh = max(player.max_speed_kmh, stats.get("max_speed_kmh", 0.0))
        player.goals += stats.get("goals", 0)
        player.assists += stats.get("assists", 0)
        player.shots += stats.get("shots", 0)
        player.passes_completed += stats.get("passes_completed", 0)
        player.passes_attempted += stats.get("passes_attempted", 0)
        player.defensive_actions += stats.get("defensive_actions", 0)
        if "rating" in stats:
            player.match_ratings.append(stats["rating"])
        player.match_stats.append(stats)
        player.average_rating = sum(player.match_ratings) / len(player.match_ratings) if player.match_ratings else 0.0
        self.save()

    def get_top_players(self, metric: str, limit: int = 10) -> List[PlayerRecord]:
        sorted_players = sorted(self.players.values(), key=lambda p: getattr(p, metric, 0), reverse=True)
        return sorted_players[:limit]


class TeamDatabase:
    """Persistent team statistics storage."""

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.teams_file = db_path / "teams.json"
        self.teams: Dict[str, TeamRecord] = {}
        self._load()

    def _load(self) -> None:
        if self.teams_file.exists():
            with open(self.teams_file, "r") as f:
                data = json.load(f)
                for t in data:
                    record = TeamRecord(
                        team_id=t["team_id"],
                        team_name=t["team_name"],
                        competition=t["competition"],
                        season=t["season"],
                        matches_played=t.get("matches_played", 0),
                        wins=t.get("wins", 0),
                        draws=t.get("draws", 0),
                        losses=t.get("losses", 0),
                        goals_scored=t.get("goals_scored", 0),
                        goals_conceded=t.get("goals_conceded", 0),
                    )
                    self.teams[record.team_id] = record

    def save(self) -> None:
        data = [t.to_dict() for t in self.teams.values()]
        with open(self.teams_file, "w") as f:
            json.dump(data, f, indent=4)

    def add_match_stats(self, team_id: str, stats: Dict[str, Any]) -> None:
        if team_id not in self.teams:
            logger.warning(f"Team {team_id} not found in database")
            return
        team = self.teams[team_id]
        team.matches_played += 1
        team.wins += stats.get("wins", 0)
        team.draws += stats.get("draws", 0)
        team.losses += stats.get("losses", 0)
        team.goals_scored += stats.get("goals_scored", 0)
        team.goals_conceded += stats.get("goals_conceded", 0)
        team.total_xg += stats.get("xg", 0.0)
        team.total_xa += stats.get("xa", 0.0)
        team.total_xt += stats.get("xt", 0.0)
        team.possession_avg = (team.possession_avg * (team.matches_played - 1) + stats.get("possession_pct", 0)) / team.matches_played
        team.ppda_avg = (team.ppda_avg * (team.matches_played - 1) + stats.get("ppda", 0)) / team.matches_played
        self.save()


class SeasonDatabase:
    """Unified database for season-long analysis."""

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.db_path.mkdir(parents=True, exist_ok=True)
        self.matches = MatchDatabase(db_path)
        self.players = PlayerDatabase(db_path)
        self.teams = TeamDatabase(db_path)

    def register_match(self, match: MatchRecord) -> None:
        self.matches.add_match(match)

    def get_season_matches(self, season: str) -> List[MatchRecord]:
        return self.matches.get_matches_by_season(season)

    def validate(self) -> List[str]:
        """Validate database consistency."""
        errors = []
        for match_id, match in self.matches.matches.items():
            if match.processing_status == "completed" and not Path(match.output_dir).exists():
                errors.append(f"Match {match_id} marked completed but output dir missing")
        return errors