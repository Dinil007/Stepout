"""
Players API Router Module

CRUD REST endpoints for querying Player entities stored in PostgreSQL.
"""

import logging
from typing import List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from app.api.dependencies import PaginationParams, get_db, get_pagination_params
from app.db import crud
from app.db.schemas import PlayerCreate, PlayerDetailResponse, PlayerResponse, PlayerUpdate

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Players"])


@router.post(
    "/",
    response_model=PlayerResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create Player",
    description="Creates a new player record linked to a team."
)
def create_player(player_in: PlayerCreate, db: Session = Depends(get_db)) -> PlayerResponse:
    """Creates and returns a new Player."""
    try:
        db_player = crud.create_player(db, player_in)
        db.commit()
        return db_player
    except Exception as exc:
        logger.error("Failed to create player: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to create player.")


@router.get(
    "/",
    response_model=List[PlayerResponse],
    status_code=status.HTTP_200_OK,
    summary="List Players",
    description="Returns players, optionally filtered by team_id."
)
def list_players(
    team_id: Optional[UUID] = Query(None, description="Filter by parent Team UUID"),
    pagination: PaginationParams = Depends(get_pagination_params),
    db: Session = Depends(get_db)
) -> List[PlayerResponse]:
    """Returns all players, optionally scoped to a team."""
    return crud.get_players(db, team_id=team_id, skip=pagination.skip, limit=pagination.limit)


@router.get(
    "/{player_id}",
    response_model=PlayerDetailResponse,
    status_code=status.HTTP_200_OK,
    summary="Get Player",
    description="Returns a player by UUID including nested analytics and pose metrics."
)
def get_player(player_id: UUID, db: Session = Depends(get_db)) -> PlayerDetailResponse:
    """Retrieves full player detail including analytics and pose."""
    db_player = crud.get_player(db, player_id)
    if not db_player:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Player '{player_id}' not found.")
    return db_player


@router.put(
    "/{player_id}",
    response_model=PlayerResponse,
    status_code=status.HTTP_200_OK,
    summary="Update Player",
    description="Updates player fields by UUID."
)
def update_player(player_id: UUID, player_in: PlayerUpdate, db: Session = Depends(get_db)) -> PlayerResponse:
    """Updates and returns the modified player."""
    db_player = crud.update_player(db, player_id, player_in)
    if not db_player:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Player '{player_id}' not found.")
    db.commit()
    return db_player


@router.delete(
    "/{player_id}",
    status_code=status.HTTP_200_OK,
    summary="Delete Player",
    description="Deletes a player and cascading analytics and pose data."
)
def delete_player(player_id: UUID, db: Session = Depends(get_db)) -> dict:
    """Deletes player and cascades to analytics/pose."""
    deleted = crud.delete_player(db, player_id)
    if not deleted:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Player '{player_id}' not found.")
    db.commit()
    return {"success": True, "message": f"Player '{player_id}' deleted successfully."}
