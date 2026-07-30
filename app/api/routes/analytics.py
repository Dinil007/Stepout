"""
Analytics API Router Module

REST endpoints for querying and managing PlayerAnalytics entities stored in PostgreSQL.
"""

import logging
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.api.dependencies import get_db
from app.db import crud
from app.db.schemas import (
    PlayerAnalyticsCreate, PlayerAnalyticsResponse, PlayerAnalyticsUpdate
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Analytics"])


@router.post(
    "/",
    response_model=PlayerAnalyticsResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Save Player Analytics",
    description="Creates or upserts player analytics metrics for a player."
)
def save_player_analytics(
    analytics_in: PlayerAnalyticsCreate,
    db: Session = Depends(get_db)
) -> PlayerAnalyticsResponse:
    """Upserts PlayerAnalytics record for the given player_id."""
    try:
        result = crud.save_player_analytics(db, analytics_in)
        db.commit()
        return result
    except Exception as exc:
        logger.error("Failed to save player analytics: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to save player analytics.")


@router.get(
    "/{player_id}",
    response_model=PlayerAnalyticsResponse,
    status_code=status.HTTP_200_OK,
    summary="Get Player Analytics",
    description="Returns Phase 1 physical analytics for a player by player UUID."
)
def get_player_analytics(player_id: UUID, db: Session = Depends(get_db)) -> PlayerAnalyticsResponse:
    """Retrieves PlayerAnalytics for the given player_id."""
    result = crud.get_player_analytics(db, player_id)
    if not result:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Analytics not found for player '{player_id}'."
        )
    return result


@router.put(
    "/{player_id}",
    response_model=PlayerAnalyticsResponse,
    status_code=status.HTTP_200_OK,
    summary="Update Player Analytics",
    description="Updates analytics metrics for a player by player UUID."
)
def update_player_analytics(
    player_id: UUID,
    analytics_in: PlayerAnalyticsUpdate,
    db: Session = Depends(get_db)
) -> PlayerAnalyticsResponse:
    """Updates PlayerAnalytics for the given player_id."""
    result = crud.update_player_analytics(db, player_id, analytics_in)
    if not result:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Analytics not found for player '{player_id}'."
        )
    db.commit()
    return result
