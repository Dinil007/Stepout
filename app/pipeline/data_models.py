"""
Pipeline Data Models

Pydantic-style data models (using dataclasses for simplicity) that define
the structured data exchanged between pipeline stages.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import numpy as np


@dataclass
class FrameData:
    """Represents a single video frame with its metadata."""
    frame_number: int
    image: Optional[np.ndarray] = None
    timestamp: float = 0.0
    raw_detections: List[Dict[str, Any]] = field(default_factory=list)
    is_key_frame: bool = False


@dataclass
class VideoMetadata:
    """Metadata about the input video."""
    input_path: Path
    total_frames: int
    fps: float
    width: int
    height: int
    duration_s: float
    codec: str = ""


@dataclass
class DetectionData:
    """Output from the Detection stage."""
    frame_number: int
    timestamp: float
    player_detections: List[Dict[str, Any]] = field(default_factory=list)
    ball_detections: List[Dict[str, Any]] = field(default_factory=list)
    detection_count: int = 0
    ball_confidence: float = 0.0


@dataclass
class TrackData:
    """Output from the Tracking stage."""
    frame_number: int
    timestamp: float
    player_tracks: Dict[int, Dict[str, Any]] = field(default_factory=dict)
    ball_track: Optional[Dict[str, Any]] = None
    track_count: int = 0
    ball_tracked: bool = False


@dataclass
class TeamData:
    """Output from the Team Classification stage."""
    frame_number: int
    timestamp: float
    team_assignments: Dict[int, str] = field(default_factory=dict)
    team_colors: Dict[str, Tuple[int, int, int]] = field(default_factory=dict)
    confidence_scores: Dict[int, float] = field(default_factory=dict)


@dataclass
class PoseData:
    """Output from the Pose Estimation stage."""
    frame_number: int
    timestamp: float
    player_keypoints: Dict[int, np.ndarray] = field(default_factory=dict)
    player_scores: Dict[int, float] = field(default_factory=dict)


@dataclass
class HomographyData:
    """Output from the Homography stage."""
    frame_number: int
    timestamp: float
    homography_matrix: Optional[np.ndarray] = None
    camera_motion_vector: Optional[Tuple[float, float]] = None
    ball_world_position: Optional[Tuple[float, float]] = None
    player_world_positions: Dict[int, Tuple[float, float]] = field(default_factory=dict)
    is_valid: bool = False
    reprojection_error: float = 0.0


@dataclass
class KinematicsData:
    """Output from the Player Kinematics stage."""
    frame_number: int
    timestamp: float
    player_speeds: Dict[int, float] = field(default_factory=dict)
    player_accelerations: Dict[int, float] = field(default_factory=dict)
    player_distances: Dict[int, float] = field(default_factory=dict)
    player_headings: Dict[int, float] = field(default_factory=dict)


@dataclass
class BallAnalyticsData:
    """Output from the Ball Analytics stage."""
    frame_number: int
    timestamp: float
    ball_speed: float = 0.0
    ball_acceleration: float = 0.0
    possession_player: Optional[int] = None
    possession_team: Optional[str] = None
    is_pass: bool = False
    is_touch: bool = False


@dataclass
class BiomechanicsData:
    """Output from the Biomechanics stage."""
    frame_number: int
    timestamp: float
    player_metrics: Dict[int, Dict[str, float]] = field(default_factory=dict)


@dataclass
class StageResult:
    """
    Generic wrapper for any stage output.
    Contains the stage name, status, data, and optional error.
    """
    stage_name: str
    success: bool
    data: Optional[Any] = None
    error: Optional[str] = None
    execution_time_s: float = 0.0
    frames_processed: int = 0


@dataclass
class PipelineInput:
    """Input to the full pipeline."""
    video_path: Path
    output_dir: Path = Path("outputs")
    max_frames: Optional[int] = None
    config_override: Optional[Dict[str, Any]] = None


@dataclass
class PipelineOutput:
    """
    Final pipeline output containing all aggregated results,
    CSV paths, video path, and summary JSON.
    """
    video_metadata: Optional[VideoMetadata] = None
    annotated_video_path: Optional[Path] = None
    player_metrics_csv: Optional[Path] = None
    ball_metrics_csv: Optional[Path] = None
    biomechanics_csv: Optional[Path] = None
    summary_json_path: Optional[Path] = None
    player_kinematics_data: Dict[int, List[Dict]] = field(default_factory=dict)
    ball_analytics_data: Dict[int, List[Dict]] = field(default_factory=dict)
    stage_results: Dict[str, StageResult] = field(default_factory=dict)
    total_execution_time_s: float = 0.0
    total_frames_processed: int = 0
    success: bool = False
    error: Optional[str] = None