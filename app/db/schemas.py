"""
Pydantic v2 Schemas Module

Defines request validation and response serialization DTOs for the
StepOut Football Analytics Platform API endpoints.
All response schemas use `ConfigDict(from_attributes=True)` for SQLAlchemy 2.x compatibility.
"""

from datetime import datetime
from typing import List, Optional
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


# ==========================================
# 1. Player Pose Schemas
# ==========================================
class PlayerPoseBase(BaseModel):
    """Base fields for player pose and biomechanics metrics."""
    cadence: Optional[float] = Field(None, description="Cadence in steps per minute (spm)", ge=0.0)
    stride_length: Optional[float] = Field(None, description="Normalized stride length", ge=0.0)
    knee_drive: Optional[float] = Field(None, description="Average knee drive angle in degrees", ge=0.0, le=180.0)
    hip_extension: Optional[float] = Field(None, description="Average hip extension angle in degrees", ge=0.0, le=180.0)
    trunk_lean: Optional[float] = Field(None, description="Forward trunk lean angle from vertical in degrees", ge=0.0, le=180.0)
    vertical_oscillation: Optional[float] = Field(None, description="Vertical oscillation of hip midpoint", ge=0.0)
    ground_contact_time: Optional[float] = Field(None, description="Estimated ground contact time percentage", ge=0.0, le=100.0)
    running_efficiency: Optional[float] = Field(None, description="Composite running efficiency score (0-100)", ge=0.0, le=100.0)

    left_knee_angle: Optional[float] = Field(None, description="Left knee joint angle in degrees", ge=0.0, le=180.0)
    right_knee_angle: Optional[float] = Field(None, description="Right knee joint angle in degrees", ge=0.0, le=180.0)
    left_hip_angle: Optional[float] = Field(None, description="Left hip joint angle in degrees", ge=0.0, le=180.0)
    right_hip_angle: Optional[float] = Field(None, description="Right hip joint angle in degrees", ge=0.0, le=180.0)

    gait_pattern: Optional[str] = Field(None, description="Gait pattern classification: Balanced, Left-Dominant, Right-Dominant, Irregular")
    injury_risk: Optional[str] = Field(None, description="Injury risk level: LOW, MEDIUM, HIGH")
    risk_score: Optional[float] = Field(None, description="Injury risk score (0-100)", ge=0.0, le=100.0)


class PlayerPoseCreate(PlayerPoseBase):
    """Schema for creating a PlayerPose record."""
    player_id: UUID = Field(..., description="Foreign key UUID of the parent Player")


class PlayerPoseUpdate(BaseModel):
    """Schema for updating an existing PlayerPose record (all fields optional)."""
    cadence: Optional[float] = Field(None, ge=0.0)
    stride_length: Optional[float] = Field(None, ge=0.0)
    knee_drive: Optional[float] = Field(None, ge=0.0, le=180.0)
    hip_extension: Optional[float] = Field(None, ge=0.0, le=180.0)
    trunk_lean: Optional[float] = Field(None, ge=0.0, le=180.0)
    vertical_oscillation: Optional[float] = Field(None, ge=0.0)
    ground_contact_time: Optional[float] = Field(None, ge=0.0, le=100.0)
    running_efficiency: Optional[float] = Field(None, ge=0.0, le=100.0)
    left_knee_angle: Optional[float] = Field(None, ge=0.0, le=180.0)
    right_knee_angle: Optional[float] = Field(None, ge=0.0, le=180.0)
    left_hip_angle: Optional[float] = Field(None, ge=0.0, le=180.0)
    right_hip_angle: Optional[float] = Field(None, ge=0.0, le=180.0)
    gait_pattern: Optional[str] = None
    injury_risk: Optional[str] = None
    risk_score: Optional[float] = Field(None, ge=0.0, le=100.0)


class PlayerPoseResponse(PlayerPoseBase):
    """Response schema for PlayerPose including ORM metadata."""
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    player_id: UUID
    created_at: datetime
    updated_at: datetime


# ==========================================
# 2. Player Analytics Schemas
# ==========================================
class PlayerAnalyticsBase(BaseModel):
    """Base fields for player physical and tactical analytics."""
    total_distance: Optional[float] = Field(None, description="Total distance covered in meters", ge=0.0)
    average_speed: Optional[float] = Field(None, description="Time-averaged speed in km/h", ge=0.0)
    top_speed: Optional[float] = Field(None, description="Maximum recorded speed in km/h", ge=0.0)
    acceleration: Optional[float] = Field(None, description="Peak positive acceleration in m/s²")
    deceleration: Optional[float] = Field(None, description="Peak deceleration in m/s²")
    sprint_count: Optional[int] = Field(None, description="Number of sprint occurrences", ge=0)
    possession_time: Optional[float] = Field(None, description="Total time in possession in seconds", ge=0.0)
    passes_completed: Optional[int] = Field(None, description="Number of completed passes", ge=0)
    passes_attempted: Optional[int] = Field(None, description="Total pass attempts", ge=0)


class PlayerAnalyticsCreate(PlayerAnalyticsBase):
    """Schema for creating a PlayerAnalytics record."""
    player_id: UUID = Field(..., description="Foreign key UUID of the parent Player")


class PlayerAnalyticsUpdate(BaseModel):
    """Schema for updating an existing PlayerAnalytics record."""
    total_distance: Optional[float] = Field(None, ge=0.0)
    average_speed: Optional[float] = Field(None, ge=0.0)
    top_speed: Optional[float] = Field(None, ge=0.0)
    acceleration: Optional[float] = None
    deceleration: Optional[float] = None
    sprint_count: Optional[int] = Field(None, ge=0)
    possession_time: Optional[float] = Field(None, ge=0.0)
    passes_completed: Optional[int] = Field(None, ge=0)
    passes_attempted: Optional[int] = Field(None, ge=0)


class PlayerAnalyticsResponse(PlayerAnalyticsBase):
    """Response schema for PlayerAnalytics including ORM metadata."""
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    player_id: UUID
    created_at: datetime
    updated_at: datetime


# ==========================================
# 3. Player Schemas
# ==========================================
class PlayerBase(BaseModel):
    """Base fields for player core entity."""
    track_id: int = Field(..., description="ByteTrack assigned tracking ID from CV pipeline", ge=0)
    jersey_number: Optional[int] = Field(None, description="Player jersey number", ge=1, le=99)
    player_name: Optional[str] = Field(None, description="Full name of the player", max_length=255)
    position: Optional[str] = Field(None, description="Field position (e.g., Forward, Midfielder, Defender, Goalkeeper)", max_length=100)


class PlayerCreate(PlayerBase):
    """Schema for creating a new Player."""
    team_id: UUID = Field(..., description="Foreign key UUID of the parent Team")


class PlayerUpdate(BaseModel):
    """Schema for updating a Player record."""
    track_id: Optional[int] = Field(None, ge=0)
    jersey_number: Optional[int] = Field(None, ge=1, le=99)
    player_name: Optional[str] = Field(None, max_length=255)
    position: Optional[str] = Field(None, max_length=100)


class PlayerResponse(PlayerBase):
    """Standard response schema for Player without nested objects."""
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    team_id: UUID
    created_at: datetime
    updated_at: datetime


class PlayerDetailResponse(PlayerResponse):
    """Detailed response schema for Player including nested Analytics and Pose metrics."""
    analytics: Optional[PlayerAnalyticsResponse] = Field(None, description="Phase 1 physical analytics")
    pose: Optional[PlayerPoseResponse] = Field(None, description="Phase 2 biomechanics & pose metrics")


# ==========================================
# 4. Team Schemas
# ==========================================
class TeamBase(BaseModel):
    """Base fields for Team entity."""
    team_name: str = Field(..., description="Display name of the team", min_length=1, max_length=255)
    team_color: Optional[str] = Field(None, description="Primary jersey color code or name", max_length=50)


class TeamCreate(TeamBase):
    """Schema for creating a new Team."""
    match_id: UUID = Field(..., description="Foreign key UUID of the parent Match")


class TeamUpdate(BaseModel):
    """Schema for updating a Team record."""
    team_name: Optional[str] = Field(None, min_length=1, max_length=255)
    team_color: Optional[str] = Field(None, max_length=50)


class TeamResponse(TeamBase):
    """Standard response schema for Team without nested players."""
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    match_id: UUID
    created_at: datetime
    updated_at: datetime


class TeamDetailResponse(TeamResponse):
    """Detailed response schema for Team including nested players."""
    players: List[PlayerDetailResponse] = Field(default_factory=list, description="List of players belonging to this team")


# ==========================================
# 5. Match Schemas
# ==========================================
class MatchBase(BaseModel):
    """Base fields for Match entity."""
    match_name: str = Field(..., description="Match title (e.g. 'Arsenal vs Chelsea')", min_length=1, max_length=255)
    competition: Optional[str] = Field(None, description="Competition or tournament name", max_length=255)
    season: Optional[str] = Field(None, description="Season identifier (e.g. '2024-25')", max_length=50)
    match_date: Optional[datetime] = Field(None, description="Scheduled match timestamp (UTC)")
    stadium: Optional[str] = Field(None, description="Venue name", max_length=255)
    duration_seconds: Optional[int] = Field(None, description="Total video duration in seconds", ge=0)
    video_path: Optional[str] = Field(None, description="Filesystem path to source video")


class MatchCreate(MatchBase):
    """Schema for creating a new Match."""
    pass


class MatchUpdate(BaseModel):
    """Schema for updating a Match record."""
    match_name: Optional[str] = Field(None, min_length=1, max_length=255)
    competition: Optional[str] = Field(None, max_length=255)
    season: Optional[str] = Field(None, max_length=50)
    match_date: Optional[datetime] = None
    stadium: Optional[str] = Field(None, max_length=255)
    duration_seconds: Optional[int] = Field(None, ge=0)
    video_path: Optional[str] = None


class MatchResponse(MatchBase):
    """Standard response schema for Match without nested teams."""
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    created_at: datetime
    updated_at: datetime


class MatchDetailResponse(MatchResponse):
    """Detailed response schema for Match including nested Teams, Players, Analytics, and Pose."""
    teams: List[TeamDetailResponse] = Field(default_factory=list, description="List of competing teams in this match")
