"""
Tracker Configuration Module

Central configuration for the ReID-enhanced tracking pipeline.
All parameters are configurable via config.yaml with sensible defaults.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class ReIDConfig:
    """Re-Identification configuration parameters."""
    enabled: bool = True
    model: str = "osnet_x0_25"
    device: str = "cuda"
    appearance_weight: float = 0.40
    motion_weight: float = 0.60
    similarity_threshold: float = 0.75
    max_embedding_history: int = 20
    max_lost_frames: int = 25
    reassign_ids: bool = True
    batch_size: int = 8
    input_size: tuple = (128, 256)  # (width, height) for OSNet


@dataclass
class ByteTrackConfig:
    """ByteTrack-specific configuration (mirrors config.yaml)."""
    track_high_thresh: float = 0.35
    track_low_thresh: float = 0.08
    new_track_thresh: float = 0.28
    track_buffer: int = 150
    match_thresh: float = 0.72
    fuse_score: bool = True
    min_track_frames: int = 2


@dataclass
class TrackerConfig:
    """Complete tracker configuration."""
    reid: ReIDConfig = field(default_factory=ReIDConfig)
    bytetrack: ByteTrackConfig = field(default_factory=ByteTrackConfig)
    tracker_type: str = "bytetrack"
    persist: bool = True

    @classmethod
    def from_dict(cls, cfg: Dict) -> "TrackerConfig":
        """Create config from a dictionary (typically from config.yaml)."""
        tracking_cfg = cfg.get("tracking", {})
        reid_cfg = tracking_cfg.get("reid", {})

        return cls(
            tracker_type=tracking_cfg.get("tracker_type", "bytetrack"),
            persist=tracking_cfg.get("persist", True),
            reid=ReIDConfig(
                enabled=reid_cfg.get("enabled", True),
                model=reid_cfg.get("model", "osnet_x0_25"),
                device=reid_cfg.get("device", "cuda"),
                appearance_weight=float(reid_cfg.get("appearance_weight", 0.40)),
                motion_weight=float(reid_cfg.get("motion_weight", 0.60)),
                similarity_threshold=float(reid_cfg.get("similarity_threshold", 0.75)),
                max_embedding_history=int(reid_cfg.get("max_embedding_history", 20)),
                max_lost_frames=int(reid_cfg.get("max_lost_frames", 25)),
                reassign_ids=bool(reid_cfg.get("reassign_ids", True)),
                batch_size=int(reid_cfg.get("batch_size", 8)),
            ),
            bytetrack=ByteTrackConfig(
                track_high_thresh=float(tracking_cfg.get("track_high_thresh", 0.35)),
                track_low_thresh=float(tracking_cfg.get("track_low_thresh", 0.08)),
                new_track_thresh=float(tracking_cfg.get("new_track_thresh", 0.28)),
                track_buffer=int(tracking_cfg.get("track_buffer", 150)),
                match_thresh=float(tracking_cfg.get("match_thresh", 0.72)),
                fuse_score=bool(tracking_cfg.get("fuse_score", True)),
                min_track_frames=int(tracking_cfg.get("min_track_frames", 2)),
            ),
        )