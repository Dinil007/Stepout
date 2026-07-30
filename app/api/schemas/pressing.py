from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field


class PlayerPositionSchema(BaseModel):
    """Player position schema for pressing API requests."""

    player_id: int = Field(..., description="Unique player identifier")
    x: float = Field(..., ge=0.0, le=1.0, description="Normalized x coordinate")
    y: float = Field(..., ge=0.0, le=1.0, description="Normalized y coordinate")
    vx: float = Field(default=0.0, description="Velocity in x direction")
    vy: float = Field(default=0.0, description="Velocity in y direction")
    team_id: int = Field(default=0, description="Team identifier")


class AnalyzePressingRequest(BaseModel):
    """Request model for single pressing analysis."""

    attackers: list[PlayerPositionSchema] = Field(
        ..., min_length=1, description="Attacking player positions with velocities"
    )
    defenders: list[PlayerPositionSchema] = Field(
        ..., min_length=1, description="Defending player positions with velocities"
    )
    frame_number: int = Field(default=0, description="Frame number")
    timestamp: float = Field(default=0.0, description="Timestamp in seconds")


class AnalyzeTeamRequest(BaseModel):
    """Request model for team-specific pressing analysis."""

    team_attackers: list[PlayerPositionSchema] = Field(
        ..., min_length=1, description="Attacking players for the team"
    )
    team_defenders: list[PlayerPositionSchema] = Field(
        ..., min_length=1, description="Defending players for the team"
    )
    frame_number: int = Field(default=0, description="Frame number")
    timestamp: float = Field(default=0.0, description="Timestamp in seconds")


class MatchFrameSchema(BaseModel):
    """A single frame with attackers and defenders for match analysis."""

    attackers: list[PlayerPositionSchema] = Field(
        ..., min_length=1, description="Attacking player positions"
    )
    defenders: list[PlayerPositionSchema] = Field(
        ..., min_length=1, description="Defending player positions"
    )


class AnalyzeMatchRequest(BaseModel):
    """Request model for multi-frame match analysis."""

    frames: list[MatchFrameSchema] = Field(
        ..., min_length=1, description="Sequence of frames"
    )
    frame_numbers: Optional[list[int]] = Field(
        default=None, description="Optional frame numbers"
    )
    timestamps: Optional[list[float]] = Field(
        default=None, description="Optional timestamps"
    )


class BatchAnalyzeRequest(BaseModel):
    """Request model for batch pressing analysis."""

    frames: list[MatchFrameSchema] = Field(
        ..., min_length=1, description="Sequence of frames"
    )
    frame_numbers: Optional[list[int]] = Field(
        default=None, description="Optional frame numbers"
    )


class PressureEventSchema(BaseModel):
    """Schema for a single pressure event."""

    attacker_id: int = Field(..., description="ID of the pressing attacker")
    defender_id: int = Field(..., description="ID of the defender being pressed")
    team_id: int = Field(..., description="ID of the pressing team")
    frame_number: int = Field(..., description="Frame number")
    distance: float = Field(..., ge=0.0, description="Distance between players")
    closing_speed: float = Field(..., ge=0.0, description="Closing speed")
    pressure_angle: float = Field(..., description="Angle of approach (radians)")
    successful: bool = Field(..., description="Whether pressure was successful")


class PressingSequenceSchema(BaseModel):
    """Schema for a pressing sequence."""

    sequence_id: int = Field(..., description="Unique sequence identifier")
    team_id: int = Field(..., description="Team identifier")
    start_frame: int = Field(..., description="Starting frame")
    end_frame: int = Field(..., description="Ending frame")
    duration_seconds: float = Field(..., ge=0.0, description="Duration in seconds")
    event_count: int = Field(..., ge=0, description="Number of pressure events")


class PressingMetricsSchema(BaseModel):
    """Schema for pressing metrics."""

    total_pressures: int = Field(..., ge=0, description="Total pressure events")
    successful_pressures: int = Field(..., ge=0, description="Successful pressures")
    pressure_success_rate: float = Field(
        ..., ge=0.0, le=1.0, description="Success rate (0-1)"
    )
    average_pressure_time: float = Field(
        ..., ge=0.0, description="Average sequence duration (s)"
    )
    average_closing_speed: float = Field(
        ..., ge=0.0, description="Average closing speed"
    )
    ppda: float = Field(..., ge=0.0, description="Passes per defensive action")
    high_press_count: int = Field(..., ge=0, description="High press events")
    mid_block_count: int = Field(..., ge=0, description="Mid block events")
    low_block_count: int = Field(..., ge=0, description="Low block events")


class PressingDetectionSchema(BaseModel):
    """Schema for pressing style detection."""

    pressing_style: str = Field(..., description="Detected pressing style")
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Detection confidence"
    )
    frame_number: int = Field(..., description="Frame number")


class PressingAnalysisResponse(BaseModel):
    """Response model for pressing analysis."""

    pressure_events: list[PressureEventSchema] = Field(
        default_factory=list, description="Detected pressure events"
    )
    pressing_sequences: list[PressingSequenceSchema] = Field(
        default_factory=list, description="Detected pressing sequences"
    )
    pressing_detection: Optional[PressingDetectionSchema] = Field(
        default=None, description="Pressing style detection"
    )
    pressing_metrics: Optional[PressingMetricsSchema] = Field(
        default=None, description="Aggregate pressing metrics"
    )
    processing_time_ms: float = Field(
        ..., ge=0.0, description="Analysis time in milliseconds"
    )
    frame_number: int = Field(..., description="Frame number")


class ValidationResponse(BaseModel):
    """Response model for validation."""

    overall_valid: bool = Field(..., description="Overall validation status")
    errors: list[str] = Field(default_factory=list, description="List of errors")
    warnings: list[str] = Field(default_factory=list, description="List of warnings")
    checked_items: int = Field(..., ge=0, description="Total checks performed")
    passed_items: int = Field(..., ge=0, description="Checks passed")


class VisualizationRequest(BaseModel):
    """Request model for pressing visualization."""

    player_positions: Optional[list[list[float]]] = Field(
        default=None,
        description="List of [player_id, x, y, team_id] tuples",
    )
    ball_position: Optional[list[float]] = Field(
        default=None,
        description="Ball position [x, y]",
    )
    pressure_events: Optional[list[dict[str, Any]]] = Field(
        default=None,
        description="Pressure event data",
    )
    attacker_positions: Optional[dict[str, list[float]]] = Field(
        default=None,
        description="Mapping of attacker_id -> [x, y]",
    )
    defender_positions: Optional[dict[str, list[float]]] = Field(
        default=None,
        description="Mapping of defender_id -> [x, y]",
    )
    metrics: Optional[dict[str, Any]] = Field(
        default=None,
        description="Pressing metrics data",
    )
    detection: Optional[dict[str, Any]] = Field(
        default=None,
        description="Pressing detection data",
    )
    visualization_options: Optional[dict[str, Any]] = Field(
        default=None,
        description="VisualizerConfig overrides",
    )


class VisualizationResponse(BaseModel):
    """Response model for visualization."""

    success: bool = Field(..., description="Rendering success status")
    width: int = Field(..., ge=0, description="Image width")
    height: int = Field(..., ge=0, description="Image height")
    render_time: float = Field(..., ge=0.0, description="Rendering time in seconds")
    error: Optional[str] = Field(default=None, description="Error message if failed")


class ConfigResponse(BaseModel):
    """Response model for pressing configuration."""

    pressure_distance_threshold: float
    high_press_line_y: float
    mid_block_line_y: float
    low_block_line_y: float
    minimum_pressure_duration: float
    minimum_closing_speed: float
    ppda_window_seconds: float
    confidence_threshold: float
    smoothing_window: int
    enable_validation: bool
    enable_logging: bool


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = Field(..., description="Service status")
    module: str = Field(..., description="Module name")
    version: str = Field(..., description="Module version")