from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field


class PlayerSchema(BaseModel):
    """Player position schema for API requests."""

    player_id: int = Field(..., description="Unique player identifier")
    team_id: int = Field(..., description="Team identifier")
    team_name: str = Field(..., description="Team name")
    jersey_number: int = Field(..., description="Jersey number")
    x: float = Field(..., ge=0.0, le=1.0, description="Normalized x coordinate")
    y: float = Field(..., ge=0.0, le=1.0, description="Normalized y coordinate")
    frame_number: int = Field(..., description="Frame number")
    timestamp: Optional[datetime] = Field(default=None, description="Event timestamp")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Detection confidence")
    is_goalkeeper: bool = Field(default=False, description="Goalkeeper flag")
    is_visible: bool = Field(default=True, description="Visibility flag")


class AnalyzeFormationRequest(BaseModel):
    """Request model for single analysis."""

    match_id: str = Field(..., description="Match identifier")
    frame_number: int = Field(..., description="Frame number")
    timestamp: Optional[datetime] = Field(default=None, description="Analysis timestamp")
    players: list[PlayerSchema] = Field(..., min_length=1, description="Player positions")


class AnalyzeTeamRequest(BaseModel):
    """Request model for team-specific analysis."""

    team_id: int = Field(..., description="Team identifier")
    frame_number: int = Field(..., description="Frame number")
    players: list[PlayerSchema] = Field(..., min_length=1, description="Player positions")


class BatchFormationRequest(BaseModel):
    """Request model for batch analysis."""

    frames: list[list[PlayerSchema]] = Field(..., min_length=1, description="List of frames")


class VisualizationRequest(BaseModel):
    """Request model for visualization."""

    analysis_result: dict[str, Any] = Field(..., description="Analysis result data")
    visualization_options: Optional[dict[str, Any]] = Field(default=None, description="Visualization options")


class MetricsResponse(BaseModel):
    """Response model for formation metrics."""

    team_width: float
    team_length: float
    compactness: float
    centroid_x: float
    centroid_y: float
    convex_hull_area: float
    defensive_line: float
    midfield_line: float
    forward_line: float
    vertical_stretch: float
    horizontal_stretch: float


class FormationResponse(BaseModel):
    """Response model for formation analysis."""

    detected_formation: str = Field(..., description="Detected formation name")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Detection confidence")
    metrics: MetricsResponse = Field(..., description="Calculated metrics")
    frame_number: int = Field(..., description="Frame number")
    timestamp: Optional[datetime] = Field(default=None, description="Analysis timestamp")
    team_id: int = Field(..., description="Team identifier")
    analysis_duration_seconds: float = Field(..., description="Time taken for analysis")


class ValidationResponse(BaseModel):
    """Response model for validation."""

    overall_valid: bool = Field(..., description="Overall validation status")
    errors: list[str] = Field(default_factory=list, description="List of errors")
    warnings: list[str] = Field(default_factory=list, description="List of warnings")
    checked_items: int = Field(..., description="Total checks performed")
    passed_items: int = Field(..., description="Checks passed")


class VisualizationResponse(BaseModel):
    """Response model for visualization."""

    success: bool = Field(..., description="Rendering success status")
    width: int = Field(..., description="Image width")
    height: int = Field(..., description="Image height")
    render_time: float = Field(..., description="Rendering time in seconds")
    image_base64: Optional[str] = Field(default=None, description="Base64 encoded image")
    error: Optional[str] = Field(default=None, description="Error message if failed")


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = Field(..., description="Service status")
    module: str = Field(..., description="Module name")
    version: str = Field(..., description="Module version")


class TemplateListResponse(BaseModel):
    """Response model for template list."""

    templates: list[str] = Field(..., description="List of formation names")
    count: int = Field(..., description="Number of templates")


class ConfigResponse(BaseModel):
    """Response model for configuration."""

    analysis_window_seconds: float
    frame_stride: int
    minimum_confidence: float
    formation_change_threshold: float
    minimum_tracked_players: int
    ignore_goalkeeper: bool
    interpolate_missing_tracks: bool
    maximum_missing_frames: int
    clustering_algorithm: str
    kmeans_random_state: int
    dbscan_eps: float
    dbscan_min_samples: int
    pitch_length: float
    pitch_width: float
    smoothing_window: int
    exponential_smoothing_alpha: float
    minimum_player_visibility: float
    minimum_team_players: int
    maximum_team_players: int
    enable_parallel_processing: bool
    cache_templates: bool
    profiling_enabled: bool
    enable_logging: bool
    save_intermediate_results: bool