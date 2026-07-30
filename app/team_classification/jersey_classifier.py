"""Jersey/team classification module."""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from sklearn.cluster import KMeans

from app.core.config import get_config

logger = logging.getLogger(__name__)


class JerseyClassifier:
    """Classify players into teams based on jersey colors."""

    def __init__(self, config: Optional[Dict] = None) -> None:
        cfg = config or get_config().raw
        team_cfg = cfg.get("team_classification", {})
        
        self.unknown_threshold = float(team_cfg.get("unknown_threshold", 0.22))
        self.sticky_threshold = float(team_cfg.get("sticky_threshold", 0.55))
        self.min_samples_per_track = int(team_cfg.get("min_samples_per_track", 2))
        self.min_tracks_to_fit = int(team_cfg.get("min_tracks_to_fit", 8))
        
        self.samples: Dict[int, List[np.ndarray]] = {}
        self.track_team: Dict[int, Optional[int]] = {}
        self.track_conf: Dict[int, float] = {}
        self.model: Optional[KMeans] = None
        self.centers: Optional[np.ndarray] = None

    @staticmethod
    def extract(frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
        """Extract jersey color feature from player bounding box."""
        x1, y1, x2, y2 = bbox
        w, h = x2 - x1, y2 - y1
        if w < 8 or h < 18:
            return None
        tx1 = max(0, x1 + int(0.22 * w))
        tx2 = min(frame.shape[1], x2 - int(0.22 * w))
        ty1 = max(0, y1 + int(0.18 * h))
        ty2 = min(frame.shape[0], y1 + int(0.55 * h))
        crop = frame[ty1:ty2, tx1:tx2]
        if crop.size == 0:
            return None
        lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        sat = hsv[:, :, 1]
        val = hsv[:, :, 2]
        hue = hsv[:, :, 0]
        green = (hue > 32) & (hue < 88) & (sat > 45)
        usable = (sat > 25) & (val > 35) & (val < 235) & (~green)
        if usable.mean() < 0.08:
            usable = (val > 35) & (val < 235)
        pixels = lab[usable]
        if len(pixels) < 8:
            return None
        return np.median(pixels[:, 1:3], axis=0).astype(np.float32)

    def update_sample(self, track_id: int, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> None:
        """Add a new color sample for a track."""
        color = self.extract(frame, bbox)
        if color is not None:
            if track_id not in self.samples:
                self.samples[track_id] = []
            self.samples[track_id].append(color)

    def fit_if_ready(self) -> None:
        """Fit KMeans if enough samples collected."""
        colors = []
        for vals in self.samples.values():
            if len(vals) >= self.min_samples_per_track:
                colors.append(np.median(np.array(vals), axis=0))
        if len(colors) >= self.min_tracks_to_fit:
            self.model = KMeans(n_clusters=2, random_state=7, n_init=10).fit(np.array(colors))
            self.centers = self.model.cluster_centers_

    def _compute_confidence(self, color: np.ndarray) -> float:
        """Compute confidence based on distance to cluster centers."""
        if self.centers is None or len(self.centers) < 2:
            return 0.0
        dists = np.linalg.norm(self.centers - color, axis=1)
        if len(dists) < 2:
            return 0.0
        return float(np.clip((dists.max() - dists.min()) / max(dists.max(), 1.0), 0, 1))

    def classify(self, track_id: int, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> Tuple[str, float]:
        """Classify a player into a team."""
        # Sticky assignment: preserve existing assignment if confidence high enough
        if track_id in self.track_team and self.track_conf.get(track_id, 0.0) >= self.sticky_threshold:
            team = self.track_team[track_id]
            return (f"Team {'A' if team == 0 else 'B'}" if team is not None else "Unknown", self.track_conf[track_id])
        
        # Extract color feature
        color = self.extract(frame, bbox)
        if color is None:
            self.track_team[track_id] = None
            self.track_conf[track_id] = 0.0
            return "Unknown", 0.0
        
        # Add to samples
        self.update_sample(track_id, frame, bbox)
        
        # Try to fit KMeans model if enough samples collected across tracks
        self.fit_if_ready()
        
        # Try to predict if model ready
        if self.model is not None and self.centers is not None and len(self.samples.get(track_id, [])) >= self.min_samples_per_track:
            color_median = np.median(np.array(self.samples[track_id]), axis=0)
            dists = np.linalg.norm(self.centers - color_median, axis=1)
            label = int(np.argmin(dists))
            conf = self._compute_confidence(color_median)
            
            if conf < self.unknown_threshold:
                self.track_team[track_id] = None
                self.track_conf[track_id] = conf
                return "Unknown", conf
            
            self.track_team[track_id] = label
            self.track_conf[track_id] = conf
            return f"Team {'A' if label == 0 else 'B'}", conf
        
        # Not enough data
        self.track_team[track_id] = None
        self.track_conf[track_id] = 0.0
        return "Unknown", 0.0