from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ClusteringAlgorithm(str, Enum):
    """Supported clustering algorithms for formation detection."""

    KMEANS = "kmeans"
    DBSCAN = "dbscan"
    HIERARCHICAL = "hierarchical"


@dataclass
class FormationConfig:
    """Centralized configuration for the Formation Detection & Tactical Shape Intelligence Engine.

    Attributes:
        analysis_window_seconds: Duration of the analysis window in seconds.
        frame_stride: Number of frames to skip between analyses.
        minimum_confidence: Minimum confidence threshold for valid detections (0.0-1.0).
        formation_change_threshold: Minimum score difference to register a formation change.
        minimum_tracked_players: Minimum number of tracked players required for analysis.
        ignore_goalkeeper: Whether to exclude the goalkeeper from formation analysis.
        interpolate_missing_tracks: Whether to interpolate missing player tracks.
        maximum_missing_frames: Maximum consecutive missing frames before dropping a track.
        clustering_algorithm: Clustering algorithm to use for player grouping.
        kmeans_random_state: Random state for K-Means reproducibility.
        dbscan_eps: Epsilon parameter for DBSCAN clustering.
        dbscan_min_samples: Minimum samples parameter for DBSCAN clustering.
        pitch_length: Length of the pitch in meters.
        pitch_width: Width of the pitch in meters.
        smoothing_window: Window size for temporal smoothing of detections.
        exponential_smoothing_alpha: Alpha parameter for exponential smoothing.
        minimum_player_visibility: Minimum visibility ratio for a player to be included.
        minimum_team_players: Minimum number of players required on a team for analysis.
        maximum_team_players: Maximum number of players allowed on a team.
        enable_parallel_processing: Whether to enable parallel processing for performance.
        cache_templates: Whether to cache formation templates in memory.
        profiling_enabled: Whether to enable performance profiling.
        enable_logging: Whether to enable detailed logging.
        save_intermediate_results: Whether to save intermediate analysis results to disk.
    """

    # Detection
    analysis_window_seconds: float = 10.0
    frame_stride: int = 5
    minimum_confidence: float = 0.7
    formation_change_threshold: float = 0.15

    # Tracking
    minimum_tracked_players: int = 6
    ignore_goalkeeper: bool = True
    interpolate_missing_tracks: bool = True
    maximum_missing_frames: int = 10

    # Clustering
    clustering_algorithm: ClusteringAlgorithm = ClusteringAlgorithm.KMEANS
    kmeans_random_state: int = 42
    dbscan_eps: float = 0.3
    dbscan_min_samples: int = 3

    # Pitch
    pitch_length: float = 105.0
    pitch_width: float = 68.0

    # Smoothing
    smoothing_window: int = 3
    exponential_smoothing_alpha: float = 0.3

    # Validation
    minimum_player_visibility: float = 0.5
    minimum_team_players: int = 6
    maximum_team_players: int = 11

    # Performance
    enable_parallel_processing: bool = False
    cache_templates: bool = True
    profiling_enabled: bool = False

    # Debug
    enable_logging: bool = False
    save_intermediate_results: bool = False

    def validate(self) -> None:
        """Validate all configuration parameters.

        Raises:
            ValueError: If any configuration parameter is invalid.
        """
        if self.pitch_length <= 0:
            raise ValueError(f"pitch_length must be positive, got {self.pitch_length}")
        if self.pitch_width <= 0:
            raise ValueError(f"pitch_width must be positive, got {self.pitch_width}")
        if not (0.0 <= self.minimum_confidence <= 1.0):
            raise ValueError(
                f"minimum_confidence must be in [0.0, 1.0], got {self.minimum_confidence}"
            )
        if self.analysis_window_seconds <= 0:
            raise ValueError(
                f"analysis_window_seconds must be positive, got {self.analysis_window_seconds}"
            )
        if self.frame_stride < 1:
            raise ValueError(f"frame_stride must be at least 1, got {self.frame_stride}")
        if self.formation_change_threshold < 0:
            raise ValueError(
                f"formation_change_threshold must be non-negative, got {self.formation_change_threshold}"
            )
        if self.minimum_tracked_players < 1:
            raise ValueError(
                f"minimum_tracked_players must be at least 1, got {self.minimum_tracked_players}"
            )
        if self.maximum_missing_frames < 0:
            raise ValueError(
                f"maximum_missing_frames must be non-negative, got {self.maximum_missing_frames}"
            )
        if self.dbscan_eps <= 0:
            raise ValueError(f"dbscan_eps must be positive, got {self.dbscan_eps}")
        if self.dbscan_min_samples < 1:
            raise ValueError(
                f"dbscan_min_samples must be at least 1, got {self.dbscan_min_samples}"
            )
        if self.smoothing_window < 1:
            raise ValueError(
                f"smoothing_window must be at least 1, got {self.smoothing_window}"
            )
        if not (0.0 <= self.exponential_smoothing_alpha <= 1.0):
            raise ValueError(
                f"exponential_smoothing_alpha must be in [0.0, 1.0], got {self.exponential_smoothing_alpha}"
            )
        if not (0.0 <= self.minimum_player_visibility <= 1.0):
            raise ValueError(
                f"minimum_player_visibility must be in [0.0, 1.0], got {self.minimum_player_visibility}"
            )
        if self.minimum_team_players < 1:
            raise ValueError(
                f"minimum_team_players must be at least 1, got {self.minimum_team_players}"
            )
        if self.maximum_team_players < self.minimum_team_players:
            raise ValueError(
                f"maximum_team_players ({self.maximum_team_players}) must be >= "
                f"minimum_team_players ({self.minimum_team_players})"
            )

    def pitch_dimensions(self) -> tuple[float, float]:
        """Return the pitch dimensions as (length, width).

        Returns:
            Tuple of (pitch_length, pitch_width) in meters.
        """
        return (self.pitch_length, self.pitch_width)

    def window_size_frames(self, fps: float) -> int:
        """Calculate the analysis window size in frames.

        Args:
            fps: Frames per second of the video.

        Returns:
            Number of frames in the analysis window.
        """
        if fps <= 0:
            raise ValueError(f"fps must be positive, got {fps}")
        return max(1, int(self.analysis_window_seconds * fps))

    def to_dict(self) -> dict[str, Any]:
        """Convert the configuration to a dictionary.

        Returns:
            Dictionary representation of the configuration.
        """
        config_dict: dict[str, Any] = {}
        for key, value in self.__dict__.items():
            if isinstance(value, ClusteringAlgorithm):
                config_dict[key] = value.value
            else:
                config_dict[key] = value
        return config_dict

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FormationConfig:
        """Create a FormationConfig from a dictionary.

        Args:
            data: Dictionary containing configuration values.

        Returns:
            FormationConfig instance initialized from the dictionary.
        """
        config_data = deepcopy(data)
        if "clustering_algorithm" in config_data and isinstance(
            config_data["clustering_algorithm"], str
        ):
            config_data["clustering_algorithm"] = ClusteringAlgorithm(
                config_data["clustering_algorithm"]
            )
        return cls(**config_data)

    def copy(self) -> FormationConfig:
        """Create a deep copy of the configuration.

        Returns:
            New FormationConfig instance with the same values.
        """
        return FormationConfig.from_dict(self.to_dict())