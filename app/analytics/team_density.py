"""
Team and Ball Density Analytics

Generates density maps for teams and ball using Kernel Density Estimation (KDE)
on transformed pitch coordinates.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import cv2

from app.homography.field_config import (
    PITCH_IMAGE_WIDTH,
    PITCH_IMAGE_HEIGHT,
    SCALE_X,
    SCALE_Y,
    FIELD_LENGTH_METERS,
    FIELD_WIDTH_METERS,
)

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)


class TeamDensityAnalytics:
    """
    Generates density maps for teams and ball using KDE.
    Uses transformed world coordinates (meters) from homography output.
    """

    def __init__(
        self,
        pitch_width_px: int = PITCH_IMAGE_WIDTH,
        pitch_height_px: int = PITCH_IMAGE_HEIGHT,
        pitch_length_m: float = FIELD_LENGTH_METERS,
        pitch_width_m: float = FIELD_WIDTH_METERS,
        bandwidth_m: float = 3.0,
    ):
        self.pitch_width_px = pitch_width_px
        self.pitch_height_px = pitch_height_px
        self.pitch_length_m = pitch_length_m
        self.pitch_width_m = pitch_width_m
        self.bandwidth_m = bandwidth_m
        self.scale_x = pitch_width_px / pitch_length_m
        self.scale_y = pitch_height_px / pitch_width_m

    def _meters_to_pixels(self, positions_m: List[Tuple[float, float]]) -> List[Tuple[int, int]]:
        """Convert world meter coordinates to pitch pixel coordinates."""
        px_coords = []
        for x_m, y_m in positions_m:
            px = int(round(x_m * self.scale_x))
            py = int(round(y_m * self.scale_y))
            px_coords.append((px, py))
        return px_coords

    def _compute_kde(
        self,
        positions_m: List[Tuple[float, float]],
        grid_shape: Tuple[int, int],
    ) -> np.ndarray:
        """
        Compute Kernel Density Estimation map.
        Uses Gaussian kernel with fixed bandwidth in meters.
        """
        if not positions_m:
            return np.zeros(grid_shape, dtype=np.float32)

        h, w = grid_shape
        density = np.zeros((h, w), dtype=np.float32)

        # Create coordinate grid in meters
        x_grid = np.arange(w) / self.scale_x
        y_grid = np.arange(h) / self.scale_y
        xx, yy = np.meshgrid(x_grid, y_grid)

        # Compute bandwidth in meters^2 (isotropic Gaussian)
        bw = self.bandwidth_m
        norm_factor = 1.0 / (2.0 * np.pi * bw * bw)

        # Accumulate density from each point
        for x_m, y_m in positions_m:
            dx = xx - x_m
            dy = yy - y_m
            d2 = dx * dx + dy * dy
            density += norm_factor * np.exp(-d2 / (2.0 * bw * bw))

        return density

    def generate_team_density_map(
        self,
        player_positions_by_frame: Dict[int, Dict[int, Tuple[float, float]]],
        team_assignments: Dict[int, str],
        target_team: str,
        base_pitch: Optional[np.ndarray] = None,
        alpha: float = 0.6,
    ) -> np.ndarray:
        """
        Generate density map for a specific team.

        Args:
            player_positions_by_frame: Dict mapping frame_number to {track_id: (x_m, y_m)}
            team_assignments: Dict mapping track_id to team label
            target_team: Team label to generate density map for
            base_pitch: Optional base pitch image to overlay on
            alpha: Transparency for overlay

        Returns:
            BGR density map image
        """
        # Collect all positions for target team
        positions = []
        for frame_players in player_positions_by_frame.values():
            for track_id, pos_m in frame_players.items():
                team = team_assignments.get(track_id)
                if team == target_team:
                    positions.append(pos_m)

        return self._render_density(positions, base_pitch, alpha, (255, 100, 0) if target_team == 0 else (0, 0, 230))

    def generate_match_density_map(
        self,
        player_positions_by_frame: Dict[int, Dict[int, Tuple[float, float]]],
        base_pitch: Optional[np.ndarray] = None,
        alpha: float = 0.6,
    ) -> np.ndarray:
        """
        Generate overall match density map using all players.

        Args:
            player_positions_by_frame: Dict mapping frame_number to {track_id: (x_m, y_m)}
            base_pitch: Optional base pitch image to overlay on
            alpha: Transparency for overlay

        Returns:
            BGR density map image
        """
        positions = []
        for frame_players in player_positions_by_frame.values():
            positions.extend(frame_players.values())

        return self._render_density(positions, base_pitch, alpha, (0, 255, 255))

    def generate_ball_density_map(
        self,
        ball_positions: List[Tuple[float, float]],
        base_pitch: Optional[np.ndarray] = None,
        alpha: float = 0.6,
    ) -> np.ndarray:
        """
        Generate ball density map.

        Args:
            ball_positions: List of (x_m, y_m) ball positions
            base_pitch: Optional base pitch image to overlay on
            alpha: Transparency for overlay

        Returns:
            BGR density map image
        """
        return self._render_density(ball_positions, base_pitch, alpha, (0, 255, 255))

    def _render_density(
        self,
        positions_m: List[Tuple[float, float]],
        base_pitch: Optional[np.ndarray],
        alpha: float,
        color_tint: Tuple[int, int, int],
    ) -> np.ndarray:
        """
        Render density map as heatmap overlay on pitch.
        """
        # Compute KDE
        density = self._compute_kde(positions_m, (self.pitch_height_px, self.pitch_width_px))

        # Normalize 0-255
        if density.max() > 0:
            density_norm = (density / density.max() * 255).astype(np.uint8)
        else:
            density_norm = density.astype(np.uint8)

        # Apply color map
        heatmap = cv2.applyColorMap(density_norm, cv2.COLORMAP_JET)

        # Create tinted version
        tinted = np.zeros_like(heatmap)
        for i in range(3):
            tinted[:, :, i] = (heatmap[:, :, i] * color_tint[i] / 255.0).astype(np.uint8)

        # Base pitch or green background
        if base_pitch is not None:
            base = base_pitch.copy()
            if base.shape[:2] != (self.pitch_height_px, self.pitch_width_px):
                base = cv2.resize(base, (self.pitch_width_px, self.pitch_height_px))
        else:
            base = np.full((self.pitch_height_px, self.pitch_width_px, 3), (20, 80, 20), dtype=np.uint8)

        overlay = cv2.addWeighted(base, 1.0 - alpha, tinted, alpha, 0)
        return overlay

    @staticmethod
    def save_image(image: np.ndarray, output_path: Path) -> Path:
        """Save density map image to disk."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), image)
        logger.info("Density map saved: %s", output_path)
        return output_path