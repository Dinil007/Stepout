"""
CRUD Operations Module

Provides production-ready Data Access Object (DAO) functions for Match, Team,
Player, PlayerAnalytics, and PlayerPose using SQLAlchemy 2.x ORM sessions.
Includes an atomic transaction pipeline for saving a complete match with all nested entities.
"""

import logging
from typing import Any, Dict, List, Optional
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.db.models import Match, Player, PlayerAnalytics, PlayerPose, Team
from app.db.schemas import (
    MatchCreate, MatchUpdate,
    PlayerAnalyticsCreate, PlayerAnalyticsUpdate,
    PlayerCreate, PlayerPoseCreate, PlayerPoseUpdate, PlayerUpdate,
    TeamCreate, TeamUpdate
)

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )


# ==========================================
# 1. MATCH CRUD
# ==========================================
def create_match(db: Session, match_in: MatchCreate) -> Match:
    """
    Creates a new Match record.

    Args:
        db: Active SQLAlchemy Session.
        match_in: Match creation schema.

    Returns:
        Created Match ORM object.
    """
    try:
        db_match = Match(**match_in.model_dump(exclude_unset=True))
        db.add(db_match)
        db.flush()
        db.refresh(db_match)
        logger.info("Created match: %s (id=%s)", db_match.match_name, db_match.id)
        return db_match
    except SQLAlchemyError as exc:
        logger.error("Failed to create match: %s", exc)
        db.rollback()
        raise


def get_match(db: Session, match_id: UUID) -> Optional[Match]:
    """
    Retrieves a Match by its UUID.

    Args:
        db: Active SQLAlchemy Session.
        match_id: UUID of the match.

    Returns:
        Match ORM object or None if not found.
    """
    stmt = select(Match).where(Match.id == match_id)
    return db.execute(stmt).scalar_one_or_none()


def get_matches(db: Session, skip: int = 0, limit: int = 100) -> List[Match]:
    """
    Retrieves a paginated list of Match records ordered by created_at descending.

    Args:
        db: Active SQLAlchemy Session.
        skip: Pagination offset.
        limit: Max records to return.

    Returns:
        List of Match ORM objects.
    """
    stmt = select(Match).order_by(Match.created_at.desc()).offset(skip).limit(limit)
    return list(db.execute(stmt).scalars().all())


def update_match(db: Session, match_id: UUID, match_in: MatchUpdate) -> Optional[Match]:
    """
    Updates an existing Match record.

    Args:
        db: Active SQLAlchemy Session.
        match_id: UUID of the match to update.
        match_in: Match update schema with optional fields.

    Returns:
        Updated Match ORM object or None if not found.
    """
    try:
        db_match = get_match(db, match_id)
        if not db_match:
            return None

        update_data = match_in.model_dump(exclude_unset=True)
        for field, value in update_data.items():
            setattr(db_match, field, value)

        db.flush()
        db.refresh(db_match)
        logger.info("Updated match: %s (id=%s)", db_match.match_name, db_match.id)
        return db_match
    except SQLAlchemyError as exc:
        logger.error("Failed to update match %s: %s", match_id, exc)
        db.rollback()
        raise


def delete_match(db: Session, match_id: UUID) -> bool:
    """
    Deletes a Match record. Cascades to teams, players, analytics, and pose records.

    Args:
        db: Active SQLAlchemy Session.
        match_id: UUID of the match to delete.

    Returns:
        True if deleted, False if not found.
    """
    try:
        db_match = get_match(db, match_id)
        if not db_match:
            return False

        db.delete(db_match)
        db.flush()
        logger.info("Deleted match id=%s", match_id)
        return True
    except SQLAlchemyError as exc:
        logger.error("Failed to delete match %s: %s", match_id, exc)
        db.rollback()
        raise


# ==========================================
# 2. TEAM CRUD
# ==========================================
def create_team(db: Session, team_in: TeamCreate) -> Team:
    """
    Creates a new Team record linked to a Match.

    Args:
        db: Active SQLAlchemy Session.
        team_in: Team creation schema.

    Returns:
        Created Team ORM object.
    """
    try:
        db_team = Team(**team_in.model_dump(exclude_unset=True))
        db.add(db_team)
        db.flush()
        db.refresh(db_team)
        logger.info("Created team: %s (id=%s)", db_team.team_name, db_team.id)
        return db_team
    except SQLAlchemyError as exc:
        logger.error("Failed to create team: %s", exc)
        db.rollback()
        raise


def get_team(db: Session, team_id: UUID) -> Optional[Team]:
    """Retrieves a Team by UUID."""
    stmt = select(Team).where(Team.id == team_id)
    return db.execute(stmt).scalar_one_or_none()


def get_teams(
    db: Session,
    match_id: Optional[UUID] = None,
    skip: int = 0,
    limit: int = 100
) -> List[Team]:
    """
    Retrieves teams, optionally filtered by match_id.

    Args:
        db: Active SQLAlchemy Session.
        match_id: Optional parent match UUID filter.
        skip: Pagination offset.
        limit: Max records.

    Returns:
        List of Team ORM objects.
    """
    stmt = select(Team)
    if match_id:
        stmt = stmt.where(Team.match_id == match_id)
    stmt = stmt.order_by(Team.created_at.asc()).offset(skip).limit(limit)
    return list(db.execute(stmt).scalars().all())


def update_team(db: Session, team_id: UUID, team_in: TeamUpdate) -> Optional[Team]:
    """Updates an existing Team record."""
    try:
        db_team = get_team(db, team_id)
        if not db_team:
            return None

        update_data = team_in.model_dump(exclude_unset=True)
        for field, value in update_data.items():
            setattr(db_team, field, value)

        db.flush()
        db.refresh(db_team)
        logger.info("Updated team: %s (id=%s)", db_team.team_name, db_team.id)
        return db_team
    except SQLAlchemyError as exc:
        logger.error("Failed to update team %s: %s", team_id, exc)
        db.rollback()
        raise


def delete_team(db: Session, team_id: UUID) -> bool:
    """Deletes a Team record and its associated players."""
    try:
        db_team = get_team(db, team_id)
        if not db_team:
            return False

        db.delete(db_team)
        db.flush()
        logger.info("Deleted team id=%s", team_id)
        return True
    except SQLAlchemyError as exc:
        logger.error("Failed to delete team %s: %s", team_id, exc)
        db.rollback()
        raise


# ==========================================
# 3. PLAYER CRUD
# ==========================================
def create_player(db: Session, player_in: PlayerCreate) -> Player:
    """Creates a new Player record linked to a Team."""
    try:
        db_player = Player(**player_in.model_dump(exclude_unset=True))
        db.add(db_player)
        db.flush()
        db.refresh(db_player)
        logger.info("Created player track_id=%d (id=%s)", db_player.track_id, db_player.id)
        return db_player
    except SQLAlchemyError as exc:
        logger.error("Failed to create player: %s", exc)
        db.rollback()
        raise


def get_player(db: Session, player_id: UUID) -> Optional[Player]:
    """Retrieves a Player by UUID."""
    stmt = select(Player).where(Player.id == player_id)
    return db.execute(stmt).scalar_one_or_none()


def get_players(
    db: Session,
    team_id: Optional[UUID] = None,
    skip: int = 0,
    limit: int = 100
) -> List[Player]:
    """Retrieves players, optionally filtered by team_id."""
    stmt = select(Player)
    if team_id:
        stmt = stmt.where(Player.team_id == team_id)
    stmt = stmt.order_by(Player.track_id.asc()).offset(skip).limit(limit)
    return list(db.execute(stmt).scalars().all())


def update_player(db: Session, player_id: UUID, player_in: PlayerUpdate) -> Optional[Player]:
    """Updates an existing Player record."""
    try:
        db_player = get_player(db, player_id)
        if not db_player:
            return None

        update_data = player_in.model_dump(exclude_unset=True)
        for field, value in update_data.items():
            setattr(db_player, field, value)

        db.flush()
        db.refresh(db_player)
        logger.info("Updated player id=%s", db_player.id)
        return db_player
    except SQLAlchemyError as exc:
        logger.error("Failed to update player %s: %s", player_id, exc)
        db.rollback()
        raise


def delete_player(db: Session, player_id: UUID) -> bool:
    """Deletes a Player record and associated analytics/pose data."""
    try:
        db_player = get_player(db, player_id)
        if not db_player:
            return False

        db.delete(db_player)
        db.flush()
        logger.info("Deleted player id=%s", player_id)
        return True
    except SQLAlchemyError as exc:
        logger.error("Failed to delete player %s: %s", player_id, exc)
        db.rollback()
        raise


# ==========================================
# 4. PLAYER ANALYTICS CRUD
# ==========================================
def get_player_analytics(db: Session, player_id: UUID) -> Optional[PlayerAnalytics]:
    """Retrieves PlayerAnalytics record by parent player_id."""
    stmt = select(PlayerAnalytics).where(PlayerAnalytics.player_id == player_id)
    return db.execute(stmt).scalar_one_or_none()


def save_player_analytics(db: Session, analytics_in: PlayerAnalyticsCreate) -> PlayerAnalytics:
    """
    Saves a PlayerAnalytics record. Upserts if a record for the player already exists.
    """
    try:
        existing = get_player_analytics(db, analytics_in.player_id)
        if existing:
            update_data = analytics_in.model_dump(exclude_unset=True, exclude={"player_id"})
            for field, value in update_data.items():
                setattr(existing, field, value)
            db.flush()
            db.refresh(existing)
            logger.info("Updated existing PlayerAnalytics for player_id=%s", analytics_in.player_id)
            return existing

        db_analytics = PlayerAnalytics(**analytics_in.model_dump(exclude_unset=True))
        db.add(db_analytics)
        db.flush()
        db.refresh(db_analytics)
        logger.info("Created PlayerAnalytics for player_id=%s", analytics_in.player_id)
        return db_analytics
    except SQLAlchemyError as exc:
        logger.error("Failed to save player analytics for player %s: %s", analytics_in.player_id, exc)
        db.rollback()
        raise


def update_player_analytics(
    db: Session,
    player_id: UUID,
    analytics_in: PlayerAnalyticsUpdate
) -> Optional[PlayerAnalytics]:
    """Updates an existing PlayerAnalytics record."""
    try:
        existing = get_player_analytics(db, player_id)
        if not existing:
            return None

        update_data = analytics_in.model_dump(exclude_unset=True)
        for field, value in update_data.items():
            setattr(existing, field, value)

        db.flush()
        db.refresh(existing)
        logger.info("Updated PlayerAnalytics for player_id=%s", player_id)
        return existing
    except SQLAlchemyError as exc:
        logger.error("Failed to update player analytics for player %s: %s", player_id, exc)
        db.rollback()
        raise


# ==========================================
# 5. PLAYER POSE CRUD
# ==========================================
def get_player_pose(db: Session, player_id: UUID) -> Optional[PlayerPose]:
    """Retrieves PlayerPose record by parent player_id."""
    stmt = select(PlayerPose).where(PlayerPose.player_id == player_id)
    return db.execute(stmt).scalar_one_or_none()


def save_player_pose(db: Session, pose_in: PlayerPoseCreate) -> PlayerPose:
    """
    Saves a PlayerPose record. Upserts if a record for the player already exists.
    """
    try:
        existing = get_player_pose(db, pose_in.player_id)
        if existing:
            update_data = pose_in.model_dump(exclude_unset=True, exclude={"player_id"})
            for field, value in update_data.items():
                setattr(existing, field, value)
            db.flush()
            db.refresh(existing)
            logger.info("Updated existing PlayerPose for player_id=%s", pose_in.player_id)
            return existing

        db_pose = PlayerPose(**pose_in.model_dump(exclude_unset=True))
        db.add(db_pose)
        db.flush()
        db.refresh(db_pose)
        logger.info("Created PlayerPose for player_id=%s", pose_in.player_id)
        return db_pose
    except SQLAlchemyError as exc:
        logger.error("Failed to save player pose for player %s: %s", pose_in.player_id, exc)
        db.rollback()
        raise


def update_player_pose(
    db: Session,
    player_id: UUID,
    pose_in: PlayerPoseUpdate
) -> Optional[PlayerPose]:
    """Updates an existing PlayerPose record."""
    try:
        existing = get_player_pose(db, player_id)
        if not existing:
            return None

        update_data = pose_in.model_dump(exclude_unset=True)
        for field, value in update_data.items():
            setattr(existing, field, value)

        db.flush()
        db.refresh(existing)
        logger.info("Updated PlayerPose for player_id=%s", player_id)
        return existing
    except SQLAlchemyError as exc:
        logger.error("Failed to update player pose for player %s: %s", player_id, exc)
        db.rollback()
        raise


# ==========================================
# 6. HIGH-LEVEL MATCH PIPELINE TRANSACTION
# ==========================================
def save_complete_match(db: Session, match_data: Dict[str, Any]) -> Match:
    """
    Atomically saves a complete Match hierarchy into the database in ONE transaction:
    Match -> Teams -> Players -> Analytics -> Pose

    If any operation fails, the ENTIRE transaction is rolled back cleanly.

    Args:
        db: Active SQLAlchemy Session.
        match_data: Nested dictionary structure:
            {
                "match": {...},
                "teams": [
                    {
                        "team": {...},
                        "players": [
                            {
                                "player": {...},
                                "analytics": {...},
                                "pose": {...}
                            }
                        ]
                    }
                ]
            }

    Returns:
        Fully populated Match ORM object.

    Raises:
        SQLAlchemyError: If any sub-entity creation fails, triggering full rollback.
    """
    logger.info("Beginning atomic complete match transaction...")

    try:
        # 1. Create Match
        match_info = match_data.get("match", {})
        match_in = MatchCreate(**match_info)
        db_match = Match(**match_in.model_dump(exclude_unset=True))
        db.add(db_match)
        db.flush()  # Generates db_match.id

        # 2. Iterate Teams
        teams_list = match_data.get("teams", [])
        for t_entry in teams_list:
            team_info = t_entry.get("team", {})
            team_in = TeamCreate(match_id=db_match.id, **team_info)
            db_team = Team(**team_in.model_dump(exclude_unset=True))
            db.add(db_team)
            db.flush()  # Generates db_team.id

            # 3. Iterate Players
            players_list = t_entry.get("players", [])
            for p_entry in players_list:
                player_info = p_entry.get("player", {})
                player_in = PlayerCreate(team_id=db_team.id, **player_info)
                db_player = Player(**player_in.model_dump(exclude_unset=True))
                db.add(db_player)
                db.flush()  # Generates db_player.id

                # 4. Save Analytics if present
                analytics_info = p_entry.get("analytics")
                if analytics_info:
                    analytics_in = PlayerAnalyticsCreate(player_id=db_player.id, **analytics_info)
                    db_analytics = PlayerAnalytics(**analytics_in.model_dump(exclude_unset=True))
                    db.add(db_analytics)

                # 5. Save Pose if present
                pose_info = p_entry.get("pose")
                if pose_info:
                    pose_in = PlayerPoseCreate(player_id=db_player.id, **pose_info)
                    db_pose = PlayerPose(**pose_in.model_dump(exclude_unset=True))
                    db.add(db_pose)

        # Commit entire atomic graph
        db.commit()
        db.refresh(db_match)
        logger.info(
            "Successfully saved complete match '%s' (id=%s) with %d teams.",
            db_match.match_name, db_match.id, len(db_match.teams)
        )
        return db_match

    except Exception as exc:
        logger.error("Complete match transaction failed. Rolling back all changes! Error: %s", exc)
        db.rollback()
        raise
