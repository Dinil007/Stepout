"""
Teams API Router Module

CRUD REST endpoints for querying Team entities stored in PostgreSQL.
"""

import logging
from typing import List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from app.api.dependencies import PaginationParams, get_db, get_pagination_params
from app.db import crud
from app.db.schemas import TeamCreate, TeamDetailResponse, TeamResponse, TeamUpdate

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Teams"])


@router.post(
    "/",
    response_model=TeamResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create Team",
    description="Creates a new team record linked to a match."
)
def create_team(team_in: TeamCreate, db: Session = Depends(get_db)) -> TeamResponse:
    """Creates and returns a new Team."""
    try:
        db_team = crud.create_team(db, team_in)
        db.commit()
        return db_team
    except Exception as exc:
        logger.error("Failed to create team: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to create team.")


@router.get(
    "/",
    response_model=List[TeamResponse],
    status_code=status.HTTP_200_OK,
    summary="List Teams",
    description="Returns teams, optionally filtered by match_id."
)
def list_teams(
    match_id: Optional[UUID] = Query(None, description="Filter by parent Match UUID"),
    pagination: PaginationParams = Depends(get_pagination_params),
    db: Session = Depends(get_db)
) -> List[TeamResponse]:
    """Returns all teams, optionally scoped to a match."""
    return crud.get_teams(db, match_id=match_id, skip=pagination.skip, limit=pagination.limit)


@router.get(
    "/{team_id}",
    response_model=TeamDetailResponse,
    status_code=status.HTTP_200_OK,
    summary="Get Team",
    description="Returns a team by UUID including nested players."
)
def get_team(team_id: UUID, db: Session = Depends(get_db)) -> TeamDetailResponse:
    """Retrieves a full team detail by UUID."""
    db_team = crud.get_team(db, team_id)
    if not db_team:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Team '{team_id}' not found.")
    return db_team


@router.put(
    "/{team_id}",
    response_model=TeamResponse,
    status_code=status.HTTP_200_OK,
    summary="Update Team",
    description="Updates team fields by UUID."
)
def update_team(team_id: UUID, team_in: TeamUpdate, db: Session = Depends(get_db)) -> TeamResponse:
    """Updates and returns the modified team."""
    db_team = crud.update_team(db, team_id, team_in)
    if not db_team:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Team '{team_id}' not found.")
    db.commit()
    return db_team


@router.delete(
    "/{team_id}",
    status_code=status.HTTP_200_OK,
    summary="Delete Team",
    description="Deletes a team and all cascading players."
)
def delete_team(team_id: UUID, db: Session = Depends(get_db)) -> dict:
    """Deletes team and cascades to players/analytics/pose."""
    deleted = crud.delete_team(db, team_id)
    if not deleted:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Team '{team_id}' not found.")
    db.commit()
    return {"success": True, "message": f"Team '{team_id}' deleted successfully."}
