"""
Matches API Router Module

CRUD REST endpoints for querying Match entities stored in PostgreSQL.
"""

import logging
from typing import List
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.api.dependencies import PaginationParams, get_db, get_pagination_params
from app.db import crud
from app.db.schemas import MatchCreate, MatchDetailResponse, MatchResponse, MatchUpdate

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Matches"])


@router.post(
    "/",
    response_model=MatchResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create Match",
    description="Creates a new match record."
)
def create_match(
    match_in: MatchCreate,
    db: Session = Depends(get_db)
) -> MatchResponse:
    """Creates a new Match record."""
    try:
        db_match = crud.create_match(db, match_in)
        db.commit()
        return db_match
    except Exception as exc:
        logger.error("Failed to create match: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to create match.")


@router.get(
    "/",
    response_model=List[MatchResponse],
    status_code=status.HTTP_200_OK,
    summary="List Matches",
    description="Returns a paginated list of all matches."
)
def list_matches(
    pagination: PaginationParams = Depends(get_pagination_params),
    db: Session = Depends(get_db)
) -> List[MatchResponse]:
    """Returns all matches ordered by created_at descending."""
    return crud.get_matches(db, skip=pagination.skip, limit=pagination.limit)


@router.get(
    "/{match_id}",
    response_model=MatchDetailResponse,
    status_code=status.HTTP_200_OK,
    summary="Get Match",
    description="Returns a match by UUID including nested teams and players."
)
def get_match(match_id: UUID, db: Session = Depends(get_db)) -> MatchDetailResponse:
    """Retrieves a full match detail by UUID."""
    db_match = crud.get_match(db, match_id)
    if not db_match:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Match '{match_id}' not found.")
    return db_match


@router.put(
    "/{match_id}",
    response_model=MatchResponse,
    status_code=status.HTTP_200_OK,
    summary="Update Match",
    description="Updates match fields by UUID."
)
def update_match(
    match_id: UUID,
    match_in: MatchUpdate,
    db: Session = Depends(get_db)
) -> MatchResponse:
    """Updates and returns the modified match."""
    db_match = crud.update_match(db, match_id, match_in)
    if not db_match:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Match '{match_id}' not found.")
    db.commit()
    return db_match


@router.delete(
    "/{match_id}",
    status_code=status.HTTP_200_OK,
    summary="Delete Match",
    description="Deletes a match and all cascading entities."
)
def delete_match(match_id: UUID, db: Session = Depends(get_db)) -> dict:
    """Deletes a match and all cascaded teams/players/analytics/pose."""
    deleted = crud.delete_match(db, match_id)
    if not deleted:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Match '{match_id}' not found.")
    db.commit()
    return {"success": True, "message": f"Match '{match_id}' deleted successfully."}
