"""
Pydantic Schemas for Request/Response Validation
"""

from pydantic import BaseModel, Field, validator
from datetime import datetime, date
from typing import Optional, List, Dict, Any
from app.api.models import UserRole, ProcessingStatus


# ==========================================
# Auth Schemas
# ==========================================
class Token(BaseModel):
    access_token: str
    token_type: str


class TokenData(BaseModel):
    email: Optional[str] = None


class UserLogin(BaseModel):
    email: str
    password: str


class UserCreate(BaseModel):
    email: str = Field(..., min_length=5, max_length=255)
    password: str = Field(..., min_length=8, max_length=100)
    full_name: str = Field(..., min_length=2, max_length=255)
    role: UserRole = UserRole.ANALYST
    team_id: Optional[int] = None


class UserResponse(BaseModel):
    user_id: int
    email: str
    full_name: str
    role: UserRole
    team_id: Optional[int]
    created_at: datetime
    last_login: Optional[datetime]
    is_active: bool

    class Config:
        from_attributes = True


class UserUpdate(BaseModel):
    full_name: Optional[str] = None
    role: Optional[UserRole] = None
    team_id: Optional[int] = None
    is_active: Optional[bool] = None


# ==========================================
# Team Schemas
# ==========================================
class TeamBase(BaseModel):
    team_name: str = Field(..., min_length=2, max_length=255)
    short_name: Optional[str] = Field(None, max_length=50)
    country: Optional[str] = Field(None, max_length=100)
    competition: Optional[str] = Field(None, max_length=100)
    founded_year: Optional[int] = Field(None, ge=1800, le=2100)
    stadium: Optional[str] = Field(None, max_length=255)
    manager: Optional[str] = Field(None, max_length=255)


class TeamCreate(TeamBase):
    pass


class TeamUpdate(BaseModel):
    team_name: Optional[str] = Field(None, min_length=2, max_length=255)
    short_name: Optional[str] = Field(None, max_length=50)
    country: Optional[str] = Field(None, max_length=100)
    competition: Optional[str] = Field(None, max_length=100)
    founded_year: Optional[int] = Field(None, ge=1800, le=2100)
    stadium: Optional[str] = Field(None, max_length=255)
    manager: Optional[str] = Field(None, max_length=255)


class TeamResponse(TeamBase):
    team_id: int
    created_at: datetime

    class Config:
        from_attributes = True


class TeamListResponse(BaseModel):
    total: int
    limit: int
    offset: int
    teams: List[TeamResponse]


# ==========================================
# Player Schemas
# ==========================================
class PlayerBase(BaseModel):
    full_name: str = Field(..., min_length=2, max_length=255)
    short_name: Optional[str] = Field(None, max_length=100)
    position: Optional[str] = Field(None, max_length=50)
    date_of_birth: Optional[date] = None
    nationality: Optional[str] = Field(None, max_length=100)
    height_cm: Optional[int] = Field(None, ge=100, le=250)
    preferred_foot: Optional[str] = Field(None, max_length=10)
    shirt_number: Optional[int] = Field(None, ge=1, le=99)
    team_id: int


class PlayerCreate(PlayerBase):
    pass


class PlayerUpdate(BaseModel):
    full_name: Optional[str] = Field(None, min_length=2, max_length=255)
    short_name: Optional[str] = Field(None, max_length=100)
    position: Optional[str] = Field(None, max_length=50)
    date_of_birth: Optional[date] = None
    nationality: Optional[str] = Field(None, max_length=100)
    height_cm: Optional[int] = Field(None, ge=100, le=250)
    preferred_foot: Optional[str] = Field(None, max_length=10)
    shirt_number: Optional[int] = Field(None, ge=1, le=99)
    team_id: Optional[int] = None


class PlayerResponse(PlayerBase):
    player_id: int
    team_name: Optional[str] = None
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class PlayerListResponse(BaseModel):
    total: int
    limit: int
    offset: int
    players: List[PlayerResponse]


class PlayerComparisonResponse(BaseModel):
    players: List[PlayerResponse]
    comparison_metrics: Dict[str, Any]


# ==========================================
# Match Schemas
# ==========================================
class MatchBase(BaseModel):
    home_team_id: int
    away_team_id: int
    competition: str = Field(..., min_length=2, max_length=100)
    season: str = Field(..., min_length=2, max_length=20)
    match_date: date
    venue: Optional[str] = Field(None, max_length=255)


class MatchCreate(MatchBase):
    pass


class MatchUpdate(BaseModel):
    home_score: Optional[int] = Field(None, ge=0)
    away_score: Optional[int] = Field(None, ge=0)
    venue: Optional[str] = Field(None, max_length=255)
    processing_status: Optional[ProcessingStatus] = None
    metadata: Optional[Dict[str, Any]] = None


class MatchResponse(MatchBase):
    match_id: int
    home_score: int
    away_score: int
    processing_status: ProcessingStatus
    created_at: datetime
    completed_at: Optional[datetime]

    class Config:
        from_attributes = True


class MatchListResponse(BaseModel):
    total: int
    limit: int
    offset: int
    matches: List[MatchResponse]


class MatchStatusResponse(BaseModel):
    match_id: int
    status: str
    progress_pct: float
    current_stage: str
    estimated_completion_seconds: int


# ==========================================
# Season Schemas
# ==========================================
class SeasonSummaryResponse(BaseModel):
    season: str
    competition: str
    total_matches: int
    total_goals: int
    avg_goals_per_match: float
    teams_tracked: int
    players_tracked: int


class PlayerSeasonStatsResponse(BaseModel):
    player_id: int
    full_name: str
    team_name: str
    position: str
    matches_played: int
    goals: int
    assists: int
    shots: int
    passes_completed: int
    passes_attempted: int
    defensive_actions: int
    distance_m: float
    max_speed_kmh: float
    xg: float
    xa: float
    xt: float
    average_rating: float


class TeamSeasonStatsResponse(BaseModel):
    team_id: int
    team_name: str
    matches_played: int
    wins: int
    draws: int
    losses: int
    points: int
    goal_difference: int
    total_xg: float
    total_xa: float
    total_xt: float


# ==========================================
# Report Schemas
# ==========================================
class MatchReportResponse(BaseModel):
    match_id: int
    report: Dict[str, Any]


# ==========================================
# Pagination
# ==========================================
class PaginationParams(BaseModel):
    skip: int = Field(0, ge=0)
    limit: int = Field(20, ge=1, le=100)


# ==========================================
# Filter Schemas
# ==========================================
class MatchFilters(BaseModel):
    season: Optional[str] = None
    competition: Optional[str] = None
    team_id: Optional[int] = None
    status: Optional[ProcessingStatus] = None
    start_date: Optional[date] = None
    end_date: Optional[date] = None


class PlayerFilters(BaseModel):
    name: Optional[str] = None
    team_id: Optional[int] = None
    position: Optional[str] = None
    season: Optional[str] = None