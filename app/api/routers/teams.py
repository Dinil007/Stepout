"""
Teams Router
"""

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from typing import List, Optional
from app.api.database import get_db
from app.api.models import Team, User
from app.api.schemas import TeamResponse, TeamListResponse
from app.api.dependencies import get_current_active_user
from app.api.logging_config import get_logger

router = APIRouter(prefix="/teams", tags=["Teams"])
logger = get_logger("teams")


@router.get("", response_model=TeamListResponse)
async def get_teams(
    skip: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100),
    name: Optional[str] = None,
    competition: Optional[str] = None,
    season: Optional[str] = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Get list of teams with filtering and pagination."""
    query = db.query(Team)
    
    # Apply filters
    if name:
        query = query.filter(Team.team_name.ilike(f"%{name}%"))
    if competition:
        query = query.filter(Team.competition == competition)
    
    # Order by name
    query = query.order_by(Team.team_name)
    
    total = query.count()
    teams = query.offset(skip).limit(limit).all()
    
    logger.info(f"Retrieved {len(teams)} teams for user {current_user.email}")
    
    return {
        "total": total,
        "limit": limit,
        "offset": skip,
        "teams": [TeamResponse.from_orm(t) for t in teams]
    }


@router.get("/{team_id}")
async def get_team(
    team_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Get team profile with stats."""
    team = db.query(Team).filter(Team.team_id == team_id).first()
    if not team:
        raise HTTPException(status_code=404, detail="Team not found")
    
    # TODO: Load season stats, formation history, tactical trends
    
    return {
        "team_id": team.team_id,
        "team_name": team.team_name,
        "season_stats": {},  # TODO: Load from season_stats
        "formation_history": [],  # TODO: Load from formation_history
        "tactical_trends": {},  # TODO: Compute from match data
        "pressing_metrics": {},  # TODO: Compute
        "possession_stats": {},  # TODO: Compute
        "strengths": [],  # TODO: Implement
        "weaknesses": []  # TODO: Implement
    }


@router.get("/{team_id}/matches")
async def get_team_matches(
    team_id: int,
    season: Optional[str] = None,
    skip: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Get team match history."""
    team = db.query(Team).filter(Team.team_id == team_id).first()
    if not team:
        raise HTTPException(status_code=404, detail="Team not found")
    
    # Query matches where team is home or away
    from app.api.models import Match
    query = db.query(Match).filter(
        (Match.home_team_id == team_id) | (Match.away_team_id == team_id)
    )
    
    if season:
        query = query.filter(Match.season == season)
    
    query = query.order_by(Match.match_date.desc())
    
    total = query.count()
    matches = query.offset(skip).limit(limit).all()
    
    # Format matches
    match_list = []
    for match in matches:
        is_home = match.home_team_id == team_id
        opponent = match.away_team.team_name if is_home else match.home_team.team_name
        venue = "Home" if is_home else "Away"
        
        team_score = match.home_score if is_home else match.away_score
        opponent_score = match.away_score if is_home else match.home_score
        result = "W" if team_score > opponent_score else "L" if team_score < opponent_score else "D"
        
        match_list.append({
            "match_id": match.match_id,
            "date": match.match_date,
            "opponent": opponent,
            "venue": venue,
            "result": result,
            "score": f"{team_score}-{opponent_score}",
            "competition": match.competition
        })
    
    return {
        "team_id": team_id,
        "total": total,
        "matches": match_list
    }