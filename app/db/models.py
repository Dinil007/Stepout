"""
SQLAlchemy 2.x ORM Models

Defines the complete database schema for the StepOut football analytics platform.
All models use UUID primary keys, typed Mapped columns, and cascading relationships.
"""

import logging
import uuid
from datetime import datetime, timezone
from typing import List, Optional

from sqlalchemy import (
    DateTime, Float, ForeignKey, Index,
    Integer, String, Text
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.database import Base

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )


# ==========================================
# Utility: UTC timestamp default
# ==========================================
def _utcnow() -> datetime:
    """Returns the current UTC-aware datetime."""
    return datetime.now(timezone.utc)


# ==========================================
# Model: Match
# ==========================================
class Match(Base):
    """
    Represents a single football match event.

    Each Match may contain multiple Team records and is the root
    entity of the analytics hierarchy.
    """

    __tablename__ = "matches"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        index=True,
        comment="Unique match identifier (UUID)"
    )
    match_name: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        comment="Human-readable match name (e.g., 'Arsenal vs Chelsea')"
    )
    competition: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
        comment="Competition or league name"
    )
    season: Mapped[Optional[str]] = mapped_column(
        String(50),
        nullable=True,
        comment="Season identifier (e.g., '2024-25')"
    )
    match_date: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        comment="Scheduled match date and time"
    )
    stadium: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
        comment="Venue name"
    )
    duration_seconds: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True,
        comment="Total match video duration in seconds"
    )
    video_path: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
        comment="Filesystem path to the source match video"
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=_utcnow,
        comment="Record creation timestamp (UTC)"
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=_utcnow,
        onupdate=_utcnow,
        comment="Record last-modified timestamp (UTC)"
    )

    # Relationships
    teams: Mapped[List["Team"]] = relationship(
        "Team",
        back_populates="match",
        cascade="all, delete-orphan",
        lazy="select"
    )

    def __repr__(self) -> str:
        return f"<Match id={self.id} name='{self.match_name}'>"


# ==========================================
# Model: Team
# ==========================================
class Team(Base):
    """
    Represents one of the competing teams within a Match.

    A Team belongs to exactly one Match and contains multiple Players.
    """

    __tablename__ = "teams"

    __table_args__ = (
        Index("ix_teams_team_name", "team_name"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        index=True,
        comment="Unique team identifier (UUID)"
    )
    match_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("matches.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        comment="Foreign key to parent Match"
    )
    team_name: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        comment="Display name of the team"
    )
    team_color: Mapped[Optional[str]] = mapped_column(
        String(50),
        nullable=True,
        comment="Primary jersey color used for classification (hex or name)"
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=_utcnow,
        comment="Record creation timestamp (UTC)"
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=_utcnow,
        onupdate=_utcnow,
        comment="Record last-modified timestamp (UTC)"
    )

    # Relationships
    match: Mapped["Match"] = relationship(
        "Match",
        back_populates="teams"
    )
    players: Mapped[List["Player"]] = relationship(
        "Player",
        back_populates="team",
        cascade="all, delete-orphan",
        lazy="select"
    )

    def __repr__(self) -> str:
        return f"<Team id={self.id} name='{self.team_name}'>"


# ==========================================
# Model: Player
# ==========================================
class Player(Base):
    """
    Represents a tracked player within a Team.

    A Player is linked to a ByteTrack track_id from the CV pipeline
    and owns associated analytics and pose metrics.
    """

    __tablename__ = "players"

    __table_args__ = (
        Index("ix_players_track_id", "track_id"),
        Index("ix_players_player_name", "player_name"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        index=True,
        comment="Unique player record identifier (UUID)"
    )
    team_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("teams.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        comment="Foreign key to parent Team"
    )
    track_id: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        comment="ByteTrack assigned tracking ID from the CV pipeline"
    )
    jersey_number: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True,
        comment="Player jersey number (if detected via OCR or known)"
    )
    player_name: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
        comment="Player's full name"
    )
    position: Mapped[Optional[str]] = mapped_column(
        String(100),
        nullable=True,
        comment="Player's field position (e.g., 'Forward', 'Midfielder')"
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=_utcnow,
        comment="Record creation timestamp (UTC)"
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=_utcnow,
        onupdate=_utcnow,
        comment="Record last-modified timestamp (UTC)"
    )

    # Relationships
    team: Mapped["Team"] = relationship(
        "Team",
        back_populates="players"
    )
    analytics: Mapped[Optional["PlayerAnalytics"]] = relationship(
        "PlayerAnalytics",
        back_populates="player",
        cascade="all, delete-orphan",
        uselist=False,
        lazy="select"
    )
    pose: Mapped[Optional["PlayerPose"]] = relationship(
        "PlayerPose",
        back_populates="player",
        cascade="all, delete-orphan",
        uselist=False,
        lazy="select"
    )

    def __repr__(self) -> str:
        return f"<Player id={self.id} track_id={self.track_id} name='{self.player_name}'>"


# ==========================================
# Model: PlayerAnalytics
# ==========================================
class PlayerAnalytics(Base):
    """
    Stores Phase 1 movement analytics for a Player across a Match.

    One-to-one relationship with Player. Contains speed, distance,
    acceleration, sprint, and passing metrics derived from the CV pipeline.
    """

    __tablename__ = "player_analytics"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        index=True,
        comment="Unique analytics record identifier (UUID)"
    )
    player_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("players.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
        index=True,
        comment="Foreign key to parent Player (one-to-one)"
    )

    # Movement metrics
    total_distance: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
        comment="Total distance covered in meters"
    )
    average_speed: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
        comment="Time-averaged speed in km/h"
    )
    top_speed: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
        comment="Maximum recorded speed in km/h"
    )
    acceleration: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
        comment="Peak positive acceleration in m/s²"
    )
    deceleration: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
        comment="Peak deceleration (most negative) in m/s²"
    )
    sprint_count: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True,
        comment="Number of frames exceeding the sprint speed threshold"
    )
    possession_time: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
        comment="Total time in possession (seconds)"
    )

    # Passing metrics
    passes_completed: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True,
        comment="Number of successfully completed passes"
    )
    passes_attempted: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True,
        comment="Total number of pass attempts"
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=_utcnow,
        comment="Record creation timestamp (UTC)"
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=_utcnow,
        onupdate=_utcnow,
        comment="Record last-modified timestamp (UTC)"
    )

    # Relationships
    player: Mapped["Player"] = relationship(
        "Player",
        back_populates="analytics"
    )

    def __repr__(self) -> str:
        return (
            f"<PlayerAnalytics id={self.id} player_id={self.player_id} "
            f"top_speed={self.top_speed}km/h>"
        )


# ==========================================
# Model: PlayerPose
# ==========================================
class PlayerPose(Base):
    """
    Stores Phase 2 biomechanical and gait metrics for a Player.

    One-to-one relationship with Player. Contains cadence, stride,
    joint angles, vertical oscillation, injury risk, and gait pattern
    derived from MediaPipe Pose and the biomechanics pipeline.
    """

    __tablename__ = "player_pose"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        index=True,
        comment="Unique pose record identifier (UUID)"
    )
    player_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("players.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
        index=True,
        comment="Foreign key to parent Player (one-to-one)"
    )

    # Gait & Running Mechanics
    cadence: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
        comment="Steps per minute (spm)"
    )
    stride_length: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
        comment="Normalized stride length (body-proportion units)"
    )
    knee_drive: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
        comment="Average knee drive angle in degrees"
    )
    hip_extension: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
        comment="Average hip extension angle in degrees"
    )
    trunk_lean: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
        comment="Forward trunk lean angle from vertical in degrees (0-180°)"
    )
    vertical_oscillation: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
        comment="Vertical oscillation of hip midpoint (normalized units)"
    )
    ground_contact_time: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
        comment="Estimated ground contact time as percentage of gait cycle"
    )
    running_efficiency: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
        comment="Composite running efficiency score (0–100)"
    )

    # Joint Angles
    left_knee_angle: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
        comment="Left knee joint angle in degrees"
    )
    right_knee_angle: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
        comment="Right knee joint angle in degrees"
    )
    left_hip_angle: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
        comment="Left hip joint angle in degrees"
    )
    right_hip_angle: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
        comment="Right hip joint angle in degrees"
    )

    # Gait Classification
    gait_pattern: Mapped[Optional[str]] = mapped_column(
        String(50),
        nullable=True,
        comment="Classified gait pattern: Balanced, Left-Dominant, Right-Dominant, Irregular"
    )

    # Injury Risk
    injury_risk: Mapped[Optional[str]] = mapped_column(
        String(20),
        nullable=True,
        comment="Injury risk classification level: LOW, MEDIUM, HIGH"
    )
    risk_score: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
        comment="Numeric injury risk score (0–100)"
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=_utcnow,
        comment="Record creation timestamp (UTC)"
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=_utcnow,
        onupdate=_utcnow,
        comment="Record last-modified timestamp (UTC)"
    )

    # Relationships
    player: Mapped["Player"] = relationship(
        "Player",
        back_populates="pose"
    )

    def __repr__(self) -> str:
        return (
            f"<PlayerPose id={self.id} player_id={self.player_id} "
            f"risk={self.injury_risk} eff={self.running_efficiency}>"
        )
