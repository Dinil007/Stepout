"""
Pose API Router Module

REST endpoints for querying and managing PlayerPose biomechanics entities stored in PostgreSQL.
"""

import logging
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.api.dependencies import get_db
from app.db import crud
from app.db.schemas import PlayerPoseCreate, PlayerPoseResponse, PlayerPoseUpdate

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Pose"])


@router.post(
    "/",
    response_model=PlayerPoseResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Save Player Pose",
    description="Creates or upserts Phase 2 pose and biomechanics metrics for a player."
)
def save_player_pose(
    pose_in: PlayerPoseCreate,
    db: Session = Depends(get_db)
) -> PlayerPoseResponse:
    """Upserts PlayerPose record for the given player_id."""
    try:
        result = crud.save_player_pose(db, pose_in)
        db.commit()
        return result
    except Exception as exc:
        logger.error("Failed to save player pose: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to save player pose.")


@router.get(
    "/{player_id}",
    response_model=PlayerPoseResponse,
    status_code=status.HTTP_200_OK,
    summary="Get Player Pose",
    description="Returns Phase 2 biomechanics and pose metrics for a player by player UUID."
)
def get_player_pose(player_id: UUID, db: Session = Depends(get_db)) -> PlayerPoseResponse:
    """Retrieves PlayerPose metrics for the given player_id."""
    result = crud.get_player_pose(db, player_id)
    if not result:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Pose metrics not found for player '{player_id}'."
        )
    return result


@router.put(
    "/{player_id}",
    response_model=PlayerPoseResponse,
    status_code=status.HTTP_200_OK,
    summary="Update Player Pose",
    description="Updates pose and biomechanics metrics for a player by player UUID."
)
def update_player_pose(
    player_id: UUID,
    pose_in: PlayerPoseUpdate,
    db: Session = Depends(get_db)
) -> PlayerPoseResponse:
    """Updates PlayerPose for the given player_id."""
    result = crud.update_player_pose(db, player_id, pose_in)
    if not result:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Pose metrics not found for player '{player_id}'."
        )
    db.commit()
    return result
