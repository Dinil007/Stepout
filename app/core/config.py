"""
Central Configuration Manager Module

Loads config.yaml and provides typed, structured configuration parameters
for all pipeline modules.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Tuple
import yaml

CONFIG_PATH = Path("config.yaml")


class ConfigManager:
    """Singleton Configuration Manager class."""

    _instance = None

    def __new__(cls, config_path: str = "config.yaml"):
        if cls._instance is None:
            cls._instance = super(ConfigManager, cls).__new__(cls)
            cls._instance._load_config(config_path)
        return cls._instance

    def _load_config(self, config_path: str) -> None:
        p = Path(config_path)
        if not p.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        with open(p, "r", encoding="utf-8") as f:
            self._config = yaml.safe_load(f)

    @property
    def raw(self) -> Dict[str, Any]:
        return self._config

    @property
    def device(self) -> str:
        return self._config.get("device", "cuda")

    @property
    def yolo_model_path(self) -> str:
        return self._config.get("models", {}).get("yolo_model_path", "yolov8x.pt")

    @property
    def confidence_threshold(self) -> float:
        return float(self._config.get("models", {}).get("confidence_threshold", 0.25))

    @property
    def iou_threshold(self) -> float:
        return float(self._config.get("models", {}).get("iou_threshold", 0.5))

    @property
    def image_size(self) -> int:
        return int(self._config.get("models", {}).get("image_size", 1280))

    @property
    def classes(self) -> List[int]:
        return self._config.get("models", {}).get("classes", [0, 32])

    @property
    def input_video_path(self) -> str:
        return self._config.get("video", {}).get("input_path", "videos/sample_video.mp4")

    @property
    def output_dir(self) -> Path:
        return Path(self._config.get("video", {}).get("output_dir", "outputs"))

    @property
    def max_frames(self) -> int:
        return int(self._config.get("video", {}).get("max_frames", 300))

    @property
    def fps(self) -> float:
        return float(self._config.get("video", {}).get("fps", 30.0))

    @property
    def pitch_length_m(self) -> float:
        return float(self._config.get("pitch", {}).get("length_m", 105.0))

    @property
    def pitch_width_m(self) -> float:
        return float(self._config.get("pitch", {}).get("width_m", 68.0))

    @property
    def pitch_canvas_size(self) -> Tuple[int, int]:
        w = self._config.get("pitch", {}).get("canvas_width", 1050)
        h = self._config.get("pitch", {}).get("canvas_height", 680)
        return (w, h)

    @property
    def tracker_config_path(self) -> str:
        return self._config.get("tracking", {}).get("tracker_config", "app/tracking/bytetrack_custom.yaml")

    @property
    def preprocessing_enabled(self) -> bool:
        return bool(self._config.get("preprocessing", {}).get("enabled", True))

    @property
    def pitch_roi_polygon(self) -> List[List[float]]:
        from app.utils.roi_loader import load_pitch_roi
        roi_polygon, _ = load_pitch_roi(project_root=Path.cwd(), verbose=False)
        return roi_polygon

    @property
    def detection_filter_config(self) -> Dict[str, Any]:
        return self._config.get("detection_filter", {})

    @property
    def team_classification_config(self) -> Dict[str, Any]:
        return self._config.get("team_classification", {})

    @property
    def tracking_config(self) -> Dict[str, Any]:
        return self._config.get("tracking", {})

    @property
    def ball_max_missing_frames(self) -> int:
        return int(self.tracking_config.get("ball_max_missing_frames", 45))

    @property
    def ball_max_match_dist(self) -> float:
        return float(self.tracking_config.get("ball_max_match_dist", 180.0))

    @property
    def min_track_frames(self) -> int:
        return int(self.tracking_config.get("min_track_frames", 2))


# Global config instance
get_config = ConfigManager
