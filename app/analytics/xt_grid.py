"""Configurable pitch grid and Expected Threat matrix for xT calculations."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List, Optional, Tuple

from app.homography.field_config import FIELD_LENGTH_METERS, FIELD_WIDTH_METERS

LOGGER = logging.getLogger(__name__)


class XTGrid:
    """Configurable pitch grid storing Expected Threat values per cell."""

    GRID_CONFIGS = {
        "12x8": (12, 8),
        "16x12": (16, 12),
        "24x16": (24, 16),
    }

    def __init__(
        self,
        grid_key: str = "12x8",
        pitch_length_m: float = FIELD_LENGTH_METERS,
        pitch_width_m: float = FIELD_WIDTH_METERS,
    ) -> None:
        if grid_key not in self.GRID_CONFIGS:
            raise ValueError(f"Unsupported grid: {grid_key}. Choose from {list(self.GRID_CONFIGS.keys())}")
        self.grid_key = grid_key
        self.cols, self.rows = self.GRID_CONFIGS[grid_key]
        self.pitch_length_m = pitch_length_m
        self.pitch_width_m = pitch_width_m
        self.cell_width_m = pitch_length_m / self.cols
        self.cell_height_m = pitch_width_m / self.rows
        self.matrix: List[List[float]] = self._default_matrix()

    def _default_matrix(self) -> List[List[float]]:
        """Return a reasonable default xT matrix."""
        matrix = [[0.01] * self.cols for _ in range(self.rows)]
        for r in range(self.rows):
            y_ratio = (r + 0.5) / self.rows
            y_centred = abs(y_ratio - 0.5) * 2.0
            for c in range(self.cols):
                # Distance to the nearest goal (left or right)
                x_min_dist = (c + 0.5) / self.cols
                x_max_dist = (self.cols - c - 0.5) / self.cols
                dist_from_goal = min(x_min_dist, x_max_dist)
                closeness = 1.0 / (1.0 + dist_from_goal * 3.0)
                angle_penalty = y_centred * 0.45
                value = 0.015 + closeness * 0.60 - angle_penalty
                value = max(value, 0.005)
                matrix[r][c] = round(min(value, 0.75), 4)
        return matrix

    def load_from_file(self, path: Path) -> None:
        """Load a pre-computed xT matrix from JSON."""
        data = json.loads(path.read_text(encoding="utf-8"))
        loaded = data.get("matrix", data)
        if len(loaded) != self.rows or any(len(row) != self.cols for row in loaded):
            raise ValueError(f"Matrix dimensions {len(loaded)}x{len(loaded[0])} don't match {self.rows}x{self.cols}")
        self.matrix = loaded
        LOGGER.info("Loaded xT matrix from %s", path)

    def save_to_file(self, path: Path) -> None:
        """Save current xT matrix to JSON."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"grid": self.grid_key, "rows": self.rows, "cols": self.cols, "matrix": self.matrix}, indent=2), encoding="utf-8")
        LOGGER.info("Saved xT matrix to %s", path)

    def cell_from_position(self, x_m: float, y_m: float) -> Tuple[int, int]:
        """Convert pitch coordinates (meters) to grid cell (col, row)."""
        col = min(self.cols - 1, max(0, int(x_m / self.cell_width_m)))
        row = min(self.rows - 1, max(0, int(y_m / self.cell_height_m)))
        return (col, row)

    def cell_centre(self, col: int, row: int) -> Tuple[float, float]:
        """Return the centre pitch coordinates of a grid cell."""
        x = (col + 0.5) * self.cell_width_m
        y = (row + 0.5) * self.cell_height_m
        return (x, y)

    def get_xt(self, col: int, row: int) -> float:
        """Return the xT value for a given cell."""
        return float(self.matrix[row][col])

    def get_xt_from_position(self, x_m: float, y_m: float) -> float:
        """Return the xT value for a pitch position."""
        col, row = self.cell_from_position(x_m, y_m)
        return self.get_xt(col, row)

    def compute_xt_added(self, start_x: float, start_y: float, end_x: float, end_y: float) -> float:
        """Compute xT added = xT(end) - xT(start)."""
        start_xt = self.get_xt_from_position(start_x, start_y)
        end_xt = self.get_xt_from_position(end_x, end_y)
        return round(end_xt - start_xt, 4)

    def is_positive_action(self, start_x: float, start_y: float, end_x: float, end_y: float) -> bool:
        return self.compute_xt_added(start_x, start_y, end_x, end_y) > 0.0