"""
Team Statistics Module

Aggregates per-player analytics into team-level summaries including total distance,
average speed, sprint counts, possession percentages, and pass statistics.
"""

import logging
from dataclasses import dataclass, asdict
from typing import Dict, List, Any, Optional
import pandas as pd

from app.analytics.player_statistics import PlayerStats

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)


@dataclass
class TeamStats:
    """
    Aggregated statistics for a single team across all players.
    """
    team_id: Any
    player_count: int = 0
    total_distance_m: float = 0.0
    avg_distance_m: float = 0.0
    total_sprint_distance_m: float = 0.0
    top_speed_kmh: float = 0.0
    avg_speed_kmh: float = 0.0
    total_sprint_count: int = 0
    total_possession_frames: int = 0
    possession_percentage: float = 0.0

    def to_dict(self) -> Dict:
        return asdict(self)


class TeamStatisticsAggregator:
    """
    Computes team-level analytics summaries from individual player statistics.
    """

    def __init__(
        self,
        player_stats: List[PlayerStats],
        possession_summary: Optional[Dict] = None,
        pass_summary: Optional[Dict[Any, Dict]] = None
    ):
        """
        Initializes the TeamStatisticsAggregator.

        Args:
            player_stats: List of per-player PlayerStats objects.
            possession_summary: Optional dict with team possession percentages.
            pass_summary: Optional dict of team_id -> pass statistics dict.
        """
        self._player_stats = player_stats
        self._possession_summary = possession_summary or {}
        self._pass_summary = pass_summary or {}

    def _group_by_team(self) -> Dict[Any, List[PlayerStats]]:
        """Groups player stats by team_id."""
        groups: Dict[Any, List[PlayerStats]] = {}
        for ps in self._player_stats:
            key = ps.team_id if ps.team_id is not None else "Unknown"
            groups.setdefault(key, []).append(ps)
        return groups

    def build_team_stats(self, team_id: Any, players: List[PlayerStats]) -> TeamStats:
        """
        Computes TeamStats from a list of players in that team.

        Args:
            team_id: Team identifier.
            players: List of PlayerStats belonging to this team.

        Returns:
            Populated TeamStats instance.
        """
        count = len(players)
        total_dist = sum(p.total_distance_m for p in players)
        total_sprint_dist = sum(p.sprint_distance_m for p in players)
        top_speed = max((p.max_speed_kmh for p in players), default=0.0)
        avg_speed = sum(p.avg_speed_kmh for p in players) / count if count > 0 else 0.0
        total_sprints = sum(p.sprint_count for p in players)
        total_poss = sum(p.possession_frames for p in players)
        poss_pct = float(self._possession_summary.get(f"team_{team_id}_pct", 0.0))

        return TeamStats(
            team_id=team_id,
            player_count=count,
            total_distance_m=round(total_dist, 2),
            avg_distance_m=round(total_dist / count, 2) if count > 0 else 0.0,
            total_sprint_distance_m=round(total_sprint_dist, 2),
            top_speed_kmh=round(top_speed, 2),
            avg_speed_kmh=round(avg_speed, 2),
            total_sprint_count=total_sprints,
            total_possession_frames=total_poss,
            possession_percentage=poss_pct
        )

    def build_all_team_stats(self) -> List[TeamStats]:
        """Computes and returns TeamStats for every team."""
        groups = self._group_by_team()
        team_stats_list = [
            self.build_team_stats(team_id, players)
            for team_id, players in groups.items()
        ]
        logger.info(f"Built team stats for {len(team_stats_list)} teams.")
        return team_stats_list

    def to_dataframe(self) -> pd.DataFrame:
        """Returns all team statistics as a pandas DataFrame."""
        stats_list = self.build_all_team_stats()
        records = [s.to_dict() for s in stats_list]
        return pd.DataFrame(records)

    def print_summary(self) -> None:
        """Logs a formatted team-level analytics summary."""
        for ts in self.build_all_team_stats():
            logger.info(
                f"\n Team {ts.team_id}:"
                f"\n   Players:         {ts.player_count}"
                f"\n   Total Distance:  {ts.total_distance_m} m"
                f"\n   Avg Distance:    {ts.avg_distance_m} m"
                f"\n   Top Speed:       {ts.top_speed_kmh} km/h"
                f"\n   Avg Speed:       {ts.avg_speed_kmh} km/h"
                f"\n   Total Sprints:   {ts.total_sprint_count}"
                f"\n   Possession:      {ts.possession_percentage}%"
            )

    def save_csv(self, output_path: str) -> str:
        """Saves team statistics as a CSV file."""
        df = self.to_dataframe()
        df.to_csv(output_path, index=False)
        logger.info(f"Team statistics saved to: {output_path}")
        return output_path
