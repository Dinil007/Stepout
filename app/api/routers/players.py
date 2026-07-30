"""
Players Router
"""

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from typing import List, Optional
from app.api.database import get_db
from app.api.models import Player, User
from app.api.schemas import PlayerResponse, PlayerListResponse, PlayerComparisonResponse
from app.api.dependencies import get_current_active_user
from app.api.logging_config import get_logger

router = APIRouter(prefix="/players", tags=["Players"])
logger = get_logger("players")


@router.get("", response_model=PlayerListResponse)
async def get_players(
    skip: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100),
    name: Optional[str] = None,
    team_id: Optional[int] = None,
    position: Optional[str] = None,
    season: Optional[str] = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Get list of players with filtering and pagination."""
    query = db.query(Player)
    
    # Apply filters
    if name:
        query = query.filter(Player.full_name.ilike(f"%{name}%"))
    if team_id:
        query = query.filter(Player.team_id == team_id)
    if position:
        query = query.filter(Player.position == position)
    
    # Order by name
    query = query.order_by(Player.full_name)
    
    total = query.count()
    players = query.offset(skip).limit(limit).all()
    
    logger.info(f"Retrieved {len(players)} players for user {current_user.email}")
    
    return {
        "total": total,
        "limit": limit,
        "offset": skip,
        "players": [PlayerResponse.from_orm(p) for p in players]
    }


@router.get("/{player_id}")
async def get_player(
    player_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Get player profile with stats."""
    player = db.query(Player).filter(Player.player_id == player_id).first()
    if not player:
        raise HTTPException(status_code=404, detail="Player not found")
    
    # Get season stats (simplified - would query season_stats table)
    # Get match history (simplified - would query player_match_stats)
    
    return {
        "player_id": player.player_id,
        "profile": {
            "full_name": player.full_name,
            "team_name": player.team.team_name if player.team else None,
            "position": player.position,
            "date_of_birth": player.date_of_birth,
            "nationality": player.nationality,
            "height_cm": player.height_cm,
            "preferred_foot": player.preferred_foot
        },
        "season_stats": {},  # TODO: Load from season_stats table
        "match_history": [],  # TODO: Load from player_match_stats
        "development_trends": {},  # TODO: Compute from match history
        "strengths": [],  # TODO: Implement
        "weaknesses": []  # TODO: Implement
    }


@router.get("/{player_id}/matches")
async def get_player_matches(
    player_id: int,
    season: Optional[str] = None,
    skip: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Get player match history."""
    player = db.query(Player).filter(Player.player_id == player_id).first()
    if not player:
        raise HTTPException(status_code=404, detail="Player not found")
    
    # TODO: Query player_match_stats with filters
    matches = []
    
    return {
        "player_id": player_id,
        "matches": matches
    }


@router.get("/compare", response_model=PlayerComparisonResponse)
async def compare_players(
    player_ids: str = Query(..., description="Comma-separated player IDs"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Compare multiple players."""
    try:
        ids = [int(pid.strip()) for pid in player_ids.split(",")]
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid player IDs format")
    
    if len(ids) < 2 or len(ids) > 5:
        raise HTTPException(status_code=400, detail="Must provide 2-5 player IDs")
    
    players = db.query(Player).filter(Player.player_id.in_(ids)).all()
    
    if not players:
        raise HTTPException(status_code=404, detail="No players found")
    
    # Build comparison metrics
    comparison_metrics = {}
    
    return {
        "players": [PlayerResponse.from_orm(p) for p in players],
        "comparison_metrics": comparison_metrics
    }