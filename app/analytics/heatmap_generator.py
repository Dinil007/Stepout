"""
Heatmap Generator Module

Generates Gaussian Kernel Density Estimation (KDE) based 2D player spatial
density heatmaps overlaid on top of the tactical pitch canvas.
Supports per-player, per-team, and full-squad heatmap generation.
"""

import logging
import os
from typing import Dict, List, Tuple, Optional
import cv2
import numpy as np

from app.homography.field_config import PITCH_IMAGE_WIDTH, PITCH_IMAGE_HEIGHT
from app.homography.visualize_pitch import PitchVisualizer

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)

# Default Gaussian smoothing kernel size (pixels, must be odd)
DEFAULT_KERNEL_SIZE: int = 51

# Heatmap overlay blend weight
PITCH_BLEND_WEIGHT: float = 0.5
HEATMAP_BLEND_WEIGHT: float = 0.5


class HeatmapGenerator:
    """
    Produces Gaussian KDE heatmaps representing player spatial density
    over a tactical 2D football pitch canvas.
    """

    def __init__(
        self,
        pitch_width: int = PITCH_IMAGE_WIDTH,
        pitch_height: int = PITCH_IMAGE_HEIGHT,
        kernel_size: int = DEFAULT_KERNEL_SIZE
    ):
        """
        Initializes the HeatmapGenerator.

        Args:
            pitch_width: Tactical pitch canvas width in pixels.
            pitch_height: Tactical pitch canvas height in pixels.
            kernel_size: Gaussian blur kernel size (must be positive and odd).
        """
        if kernel_size % 2 == 0 or kernel_size <= 0:
            raise ValueError(f"kernel_size must be a positive odd integer. Got: {kernel_size}")

        self.pitch_width = pitch_width
        self.pitch_height = pitch_height
        self.kernel_size = kernel_size

        self._visualizer = PitchVisualizer(width=pitch_width, height=pitch_height)

        # Accumulation canvas per entity: 'all' | team_id | track_id
        self._density_maps: Dict[str, np.ndarray] = {}

    def _get_or_create_map(self, key: str) -> np.ndarray:
        """Returns existing accumulation map or creates a new blank one."""
        if key not in self._density_maps:
            self._density_maps[key] = np.zeros((self.pitch_height, self.pitch_width), dtype=np.float32)
        return self._density_maps[key]

    def accumulate(
        self,
        positions: List[Tuple[float, float]],
        entity_key: str = "all"
    ) -> None:
        """
        Accumulates player positions onto the density canvas.

        Args:
            positions: List of (x, y) pixel positions on the 2D pitch canvas.
            entity_key: Identifier for the heatmap channel (e.g. 'all', 'team_0', 'player_7').
        """
        density_map = self._get_or_create_map(entity_key)
        for x, y in positions:
            px, py = int(round(x)), int(round(y))
            if 0 <= px < self.pitch_width and 0 <= py < self.pitch_height:
                density_map[py, px] += 1.0

    def accumulate_from_players(
        self,
        player_histories: Dict[int, List[Tuple[float, float]]],
        team_assignments: Optional[Dict[int, int]] = None
    ) -> None:
        """
        Bulk accumulates historical player trajectories onto density channels.

        Args:
            player_histories: Dict of track_id -> list of (x, y) canvas positions.
            team_assignments: Optional dict of track_id -> team_id for per-team heatmaps.
        """
        for track_id, positions in player_histories.items():
            # Full pitch heatmap
            self.accumulate(positions, entity_key="all")

            # Per-player heatmap
            self.accumulate(positions, entity_key=f"player_{track_id}")

            # Per-team heatmap
            if team_assignments and track_id in team_assignments:
                team_id = team_assignments[track_id]
                self.accumulate(positions, entity_key=f"team_{team_id}")

    def _render_heatmap(self, density_map: np.ndarray) -> np.ndarray:
        """
        Applies Gaussian smoothing, normalizes, and colorizes a density array.

        Returns:
            BGR heatmap image.
        """
        blurred = cv2.GaussianBlur(density_map, (self.kernel_size, self.kernel_size), 0)

        if np.max(blurred) > 0:
            normalized = (blurred / np.max(blurred) * 255).astype(np.uint8)
        else:
            normalized = blurred.astype(np.uint8)

        return cv2.applyColorMap(normalized, cv2.COLORMAP_JET)

    def generate(self, entity_key: str = "all") -> np.ndarray:
        """
        Renders a heatmap overlay on top of the tactical pitch for a given entity key.

        Args:
            entity_key: Heatmap channel identifier.

        Returns:
            BGR tactical pitch image with heatmap overlay.
        """
        base_pitch = self._visualizer.base_pitch_image.copy()
        density_map = self._get_or_create_map(entity_key)
        
        blurred = cv2.GaussianBlur(density_map, (self.kernel_size, self.kernel_size), 0)
        max_val = np.max(blurred)

        if max_val > 0:
            normalized = (blurred / max_val * 255).astype(np.uint8)
        else:
            normalized = blurred.astype(np.uint8)

        heatmap_color = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)

        # Alpha blend based on normalized density so zero-density areas remain clean pitch green
        alpha = (normalized.astype(np.float32) / 255.0) * HEATMAP_BLEND_WEIGHT
        alpha = np.expand_dims(alpha, axis=-1)

        overlay = (base_pitch.astype(np.float32) * (1.0 - alpha) + heatmap_color.astype(np.float32) * alpha).astype(np.uint8)
        return overlay

    def save(
        self,
        output_path: str,
        entity_key: str = "all"
    ) -> str:
        """
        Saves a rendered heatmap image to disk.

        Args:
            output_path: Destination file path.
            entity_key: Heatmap channel identifier.

        Returns:
            Saved absolute path.
        """
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
        heatmap = self.generate(entity_key=entity_key)
        cv2.imwrite(output_path, heatmap)
        logger.info(f"Heatmap saved: {output_path}")
        return output_path

    def save_all_team_heatmaps(self, output_dir: str) -> List[str]:
        """
        Saves individual per-team heatmaps for all registered team channels.

        Args:
            output_dir: Directory to save team heatmap images.

        Returns:
            List of saved file paths.
        """
        os.makedirs(output_dir, exist_ok=True)
        saved_paths = []
        for key in self._density_maps:
            if key.startswith("team_"):
                path = os.path.join(output_dir, f"heatmap_{key}.png")
                self.save(path, entity_key=key)
                saved_paths.append(path)
        return saved_paths

    def reset(self, entity_key: Optional[str] = None) -> None:
        """Resets density maps."""
        if entity_key is not None:
            self._density_maps.pop(entity_key, None)
        else:
            self._density_maps.clear()
