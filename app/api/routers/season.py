"""
Season Statistics Router
"""

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from typing import List, Optional
from app.api.database import get_db
from app.api.models import User, SeasonStat
from app.api.schemas import SeasonSummaryResponse, PlayerSeasonStatsResponse, TeamSeasonStatsResponse
from app.api.dependencies import get_current_active_user
from app.api.logging_config import get_logger

router = APIRouter(prefix="/season", tags=["Season"])
logger = get_logger("season")


@router.get("/{season}/summary", response_model=SeasonSummaryResponse)
async def get_season_summary(
    season: str,
    competition: str = Query(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Get season summary statistics."""
    # Count matches
    from app.api.models import Match
    total_matches = db.query(Match).filter(
        Match.season == season,
        Match.competition == competition,
        Match.processing_status.value == "completed"
    ).count()
    
    # TODO: Compute total goals, avg goals, teams tracked, players tracked
    # For now, return placeholder
    
    return {
        "season": season,
        "competition": competition,
        "total_matches": total_matches,
        "total_goals": 0,
        "avg_goals_per_match": 0.0,
        "teams_tracked": 0,
        "players_tracked": 0
    }


@router.get("/{season}/players")
async def get_season_players(
    season: str,
    competition: str = Query(...),
    team_id: Optional[int] = None,
    position: Optional[str] = None,
    skip: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Get player season statistics."""
    query = db.query(SeasonStat).filter(
        SeasonStat.season == season,
        SeasonStat.competition == competition,
        SeasonStat.entity_type == "player"
    )
    
    # Apply filters
    if team_id:
        # TODO: Filter by team through player relationship
        pass
    if position:
        # TODO: Filter by position through player relationship
        pass
    
    total = query.count()
    stats = query.offset(skip).limit(limit).all()
    
    # TODO: Join with players table to get names
    
    return {
        "total": total,
        "limit": limit,
        "offset": skip,
        "players": []
    }


@router.get("/{season}/teams")
async def get_season_teams(
    season: str,
    competition: str = Query(...),
    skip: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Get team season statistics."""
    query = db.query(SeasonStat).filter(
        SeasonStat.season == season,
        SeasonStat.competition == competition,
        SeasonStat.entity_type == "team"
    )
    
    total = query.count()
    stats = query.offset(skip).limit(limit).all()
    
    return {
        "total": total,
        "limit": limit,
        "offset": skip,
        "teams": []
    }