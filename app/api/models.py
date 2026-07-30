"""
SQLAlchemy Database Models
"""

from sqlalchemy import (
    Column, Integer, String, Float, Boolean,
    Date, DateTime, ForeignKey, Enum, JSON, Index
)
from sqlalchemy.orm import relationship, DeclarativeBase
from sqlalchemy.sql import func
import enum


class Base(DeclarativeBase):
    pass


class UserRole(str, enum.Enum):
    ADMIN = "admin"
    ANALYST = "analyst"
    COACH = "coach"
    SCOUT = "scout"


class ProcessingStatus(str, enum.Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class User(Base):
    __tablename__ = "users"
    
    user_id = Column(Integer, primary_key=True, index=True)
    email = Column(String(255), unique=True, nullable=False, index=True)
    password_hash = Column(String(255), nullable=False)
    full_name = Column(String(255), nullable=False)
    role = Column(Enum(UserRole), nullable=False)
    team_id = Column(Integer, ForeignKey("teams.team_id"), nullable=True)
    created_at = Column(DateTime, server_default=func.now())
    last_login = Column(DateTime, nullable=True)
    is_active = Column(Boolean, default=True)
    
    # Relationships
    team = relationship("Team", back_populates="users")


class Team(Base):
    __tablename__ = "teams"
    
    team_id = Column(Integer, primary_key=True, index=True)
    team_name = Column(String(255), unique=True, nullable=False)
    short_name = Column(String(50))
    country = Column(String(100))
    competition = Column(String(100), index=True)
    founded_year = Column(Integer)
    stadium = Column(String(255))
    manager = Column(String(255))
    created_at = Column(DateTime, server_default=func.now())
    
    # Relationships
    users = relationship("User", back_populates="team")
    players = relationship("Player", back_populates="team")
    home_matches = relationship("Match", foreign_keys="Match.home_team_id", back_populates="home_team")
    away_matches = relationship("Match", foreign_keys="Match.away_team_id", back_populates="away_team")


class Player(Base):
    __tablename__ = "players"
    
    player_id = Column(Integer, primary_key=True, index=True)
    team_id = Column(Integer, ForeignKey("teams.team_id"), nullable=False)
    full_name = Column(String(255), nullable=False, index=True)
    short_name = Column(String(100))
    position = Column(String(50), index=True)  # GK, DEF, MID, FWD
    date_of_birth = Column(Date)
    nationality = Column(String(100))
    height_cm = Column(Integer)
    preferred_foot = Column(String(10))
    shirt_number = Column(Integer)
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())
    
    # Relationships
    team = relationship("Team", back_populates="players")
    match_stats = relationship("PlayerMatchStat", back_populates="player")
    events = relationship("MatchEvent", back_populates="player")


class Match(Base):
    __tablename__ = "matches"
    
    match_id = Column(Integer, primary_key=True, index=True)
    home_team_id = Column(Integer, ForeignKey("teams.team_id"), nullable=False)
    away_team_id = Column(Integer, ForeignKey("teams.team_id"), nullable=False)
    competition = Column(String(100), nullable=False, index=True)
    season = Column(String(20), nullable=False, index=True)
    match_date = Column(Date, nullable=False, index=True)
    venue = Column(String(255))
    home_score = Column(Integer, default=0)
    away_score = Column(Integer, default=0)
    video_path = Column(String(500))
    duration_seconds = Column(Integer)
    processing_status = Column(Enum(ProcessingStatus), default=ProcessingStatus.PENDING, index=True)
    analytics_version = Column(String(50))
    processing_time_seconds = Column(Integer)
    output_dir = Column(String(500))
    metadata = Column(JSON)
    created_at = Column(DateTime, server_default=func.now())
    completed_at = Column(DateTime, nullable=True)
    
    # Relationships
    home_team = relationship("Team", foreign_keys=[home_team_id], back_populates="home_matches")
    away_team = relationship("Team", foreign_keys=[away_team_id], back_populates="away_matches")
    player_stats = relationship("PlayerMatchStat", back_populates="match", cascade="all, delete-orphan")
    team_stats = relationship("TeamMatchStat", back_populates="match", cascade="all, delete-orphan")
    events = relationship("MatchEvent", back_populates="match", cascade="all, delete-orphan")
    formations = relationship("FormationHistory", back_populates="match", cascade="all, delete-orphan")


class PlayerMatchStat(Base):
    __tablename__ = "player_match_stats"
    
    stat_id = Column(Integer, primary_key=True, index=True)
    match_id = Column(Integer, ForeignKey("matches.match_id"), nullable=False)
    player_id = Column(Integer, ForeignKey("players.player_id"), nullable=False)
    team_id = Column(Integer, ForeignKey("teams.team_id"), nullable=False)
    minutes_played = Column(Float, default=0)
    goals = Column(Integer, default=0)
    assists = Column(Integer, default=0)
    shots = Column(Integer, default=0)
    shots_on_target = Column(Integer, default=0)
    passes_completed = Column(Integer, default=0)
    passes_attempted = Column(Integer, default=0)
    pass_accuracy_pct = Column(Float, default=0)
    defensive_actions = Column(Integer, default=0)
    distance_m = Column(Float, default=0)
    max_speed_kmh = Column(Float, default=0)
    avg_speed_kmh = Column(Float, default=0)
    sprint_count = Column(Integer, default=0)
    xg = Column(Float, default=0)
    xa = Column(Float, default=0)
    xt = Column(Float, default=0)
    rating = Column(Float, default=0)
    heatmap_path = Column(String(500))
    created_at = Column(DateTime, server_default=func.now())
    
    # Relationships
    match = relationship("Match", back_populates="player_stats")
    player = relationship("Player", back_populates="match_stats")
    
    __table_args__ = (
        Index("idx_player_match", "match_id", "player_id", unique=True),
    )


class TeamMatchStat(Base):
    __tablename__ = "team_match_stats"
    
    stat_id = Column(Integer, primary_key=True, index=True)
    match_id = Column(Integer, ForeignKey("matches.match_id"), nullable=False)
    team_id = Column(Integer, ForeignKey("teams.team_id"), nullable=False)
    possession_pct = Column(Float, default=0)
    shots = Column(Integer, default=0)
    shots_on_target = Column(Integer, default=0)
    passes_completed = Column(Integer, default=0)
    passes_attempted = Column(Integer, default=0)
    pass_accuracy_pct = Column(Float, default=0)
    corners = Column(Integer, default=0)
    fouls = Column(Integer, default=0)
    ppda = Column(Float, default=0)
    xg = Column(Float, default=0)
    xa = Column(Float, default=0)
    xt = Column(Float, default=0)
    formation_detected = Column(String(50))
    formation_confidence = Column(Float, default=0)
    created_at = Column(DateTime, server_default=func.now())
    
    # Relationships
    match = relationship("Match", back_populates="team_stats")
    
    __table_args__ = (
        Index("idx_team_match", "match_id", "team_id", unique=True),
    )


class MatchEvent(Base):
    __tablename__ = "match_events"
    
    event_id = Column(Integer, primary_key=True, index=True)
    match_id = Column(Integer, ForeignKey("matches.match_id"), nullable=False, index=True)
    frame_number = Column(Integer, nullable=False)
    event_type = Column(String(50), nullable=False, index=True)
    team_id = Column(Integer, ForeignKey("teams.team_id"), nullable=True)
    player_id = Column(Integer, ForeignKey("players.player_id"), nullable=True)
    x = Column(Float)
    y = Column(Float)
    metadata = Column(JSON)
    created_at = Column(DateTime, server_default=func.now())
    
    # Relationships
    match = relationship("Match", back_populates="events")
    player = relationship("Player", back_populates="events")


class FormationHistory(Base):
    __tablename__ = "formation_history"
    
    formation_id = Column(Integer, primary_key=True, index=True)
    match_id = Column(Integer, ForeignKey("matches.match_id"), nullable=False, index=True)
    team_id = Column(Integer, ForeignKey("teams.team_id"), nullable=False)
    frame_number = Column(Integer, nullable=False)
    formation = Column(String(50), nullable=False)
    confidence = Column(Float, default=0)
    team_width_m = Column(Float, default=0)
    team_length_m = Column(Float, default=0)
    compactness_m = Column(Float, default=0)
    defensive_line_m = Column(Float, default=0)
    midfield_line_m = Column(Float, default=0)
    forward_line_m = Column(Float, default=0)
    is_formation_change = Column(Boolean, default=False)
    change_from = Column(String(50))
    created_at = Column(DateTime, server_default=func.now())
    
    # Relationships
    match = relationship("Match", back_populates="formations")


class SeasonStat(Base):
    __tablename__ = "season_stats"
    
    stat_id = Column(Integer, primary_key=True, index=True)
    season = Column(String(20), nullable=False, index=True)
    competition = Column(String(100), nullable=False, index=True)
    entity_type = Column(String(20), nullable=False)  # 'player' or 'team'
    entity_id = Column(Integer, nullable=False)
    matches_played = Column(Integer, default=0)
    minutes_played = Column(Float, default=0)
    goals = Column(Integer, default=0)
    assists = Column(Integer, default=0)
    shots = Column(Integer, default=0)
    passes_completed = Column(Integer, default=0)
    passes_attempted = Column(Integer, default=0)
    defensive_actions = Column(Integer, default=0)
    distance_m = Column(Float, default=0)
    max_speed_kmh = Column(Float, default=0)
    xg = Column(Float, default=0)
    xa = Column(Float, default=0)
    xt = Column(Float, default=0)
    average_rating = Column(Float, default=0)
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())
    
    __table_args__ = (
        Index("idx_season_stats_entity", "season", "competition", "entity_type", "entity_id", unique=True),
    )


class AuditLog(Base):
    __tablename__ = "audit_logs"
    
    log_id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.user_id"), nullable=True)
    action = Column(String(255), nullable=False)
    resource_type = Column(String(100))
    resource_id = Column(Integer)
    metadata = Column(JSON)
    ip_address = Column(String(45))
    user_agent = Column(String(500))
    created_at = Column(DateTime, server_default=func.now())
    
    __table_args__ = (
        Index("idx_audit_user", "user_id"),
        Index("idx_audit_action", "action"),
        Index("idx_audit_created", "created_at"),
    )