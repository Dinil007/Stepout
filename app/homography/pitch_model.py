"""
Pitch Model Module

Configurable football pitch model representing all standard markings
in world coordinates (meters).

All dimensions are loaded from config.yaml / pitch configuration,
allowing per-match customization without code changes.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class PitchDimensions:
    """Standard FIFA pitch dimensions in meters."""
    length: float = 105.0
    width: float = 68.0
    goal_width: float = 7.32
    goal_depth: float = 2.0
    penalty_area_length: float = 16.5
    penalty_area_width: float = 40.32
    goal_area_length: float = 5.5
    goal_area_width: float = 18.32
    center_circle_radius: float = 9.15
    corner_arc_radius: float = 1.0
    penalty_spot_distance: float = 11.0
    penalty_arc_radius: float = 9.15


@dataclass
class PitchModel:
    """
    Complete football pitch model in world coordinates.

    All coordinates in meters, with origin at bottom-left corner.
    X axis: 0 to length (touchlines / long sides)
    Y axis: 0 to width  (goal lines / short sides)

    Supports custom dimensions via config.
    """
    dimensions: PitchDimensions = field(default_factory=PitchDimensions)
    name: str = "FIFA Standard Pitch"

    # Cached geometry
    _corners: Optional[np.ndarray] = None
    _penalty_areas: Optional[List[np.ndarray]] = None
    _goal_areas: Optional[List[np.ndarray]] = None
    _goals: Optional[List[np.ndarray]] = None
    _halfway_line: Optional[np.ndarray] = None
    _center_circle: Optional[np.ndarray] = None
    _center_spot: Optional[np.ndarray] = None
    _touchlines: Optional[List[np.ndarray]] = None
    _goal_lines: Optional[List[np.ndarray]] = None

    def __post_init__(self) -> None:
        self._build_geometry()

    def _build_geometry(self) -> None:
        """Build all pitch geometry from dimensions."""
        L = self.dimensions.length
        W = self.dimensions.width
        pal = self.dimensions.penalty_area_length
        paw = self.dimensions.penalty_area_width
        gal = self.dimensions.goal_area_length
        gaw = self.dimensions.goal_area_width
        gw = self.dimensions.goal_width / 2.0
        gd = self.dimensions.goal_depth

        # Corners (4 corners, clockwise from bottom-left)
        self._corners = np.array([
            [0, 0],
            [L, 0],
            [L, W],
            [0, W],
        ], dtype=np.float64)

        # Penalty areas: [bottom, top] in world coords
        # Bottom: x in [0, pal], y in [(W-paw)/2, (W+paw)/2]
        # Top:    x in [L-pal, L], y in [(W-paw)/2, (W+paw)/2]
        pa_y0 = (W - paw) / 2.0
        pa_y1 = (W + paw) / 2.0
        self._penalty_areas = [
            np.array([
                [0, pa_y0],
                [pal, pa_y0],
                [pal, pa_y1],
                [0, pa_y1],
            ], dtype=np.float64),
            np.array([
                [L, pa_y0],
                [L - pal, pa_y0],
                [L - pal, pa_y1],
                [L, pa_y1],
            ], dtype=np.float64),
        ]

        # Goal areas
        ga_y0 = (W - gaw) / 2.0
        ga_y1 = (W + gaw) / 2.0
        self._goal_areas = [
            np.array([
                [0, ga_y0],
                [gal, ga_y0],
                [gal, ga_y1],
                [0, ga_y1],
            ], dtype=np.float64),
            np.array([
                [L, ga_y0],
                [L - gal, ga_y0],
                [L - gal, ga_y1],
                [L, ga_y1],
            ], dtype=np.float64),
        ]

        # Goals (as rectangles including depth behind goal line)
        self._goals = [
            np.array([
                [0, W / 2.0 - gw],
                [-gd, W / 2.0 - gw],
                [-gd, W / 2.0 + gw],
                [0, W / 2.0 + gw],
            ], dtype=np.float64),
            np.array([
                [L, W / 2.0 - gw],
                [L + gd, W / 2.0 - gw],
                [L + gd, W / 2.0 + gw],
                [L, W / 2.0 + gw],
            ], dtype=np.float64),
        ]

        # Halfway line
        self._halfway_line = np.array([
            [L / 2.0, 0],
            [L / 2.0, W],
        ], dtype=np.float64)

        # Center spot and circle
        self._center_spot = np.array([L / 2.0, W / 2.0], dtype=np.float64)
        theta = np.linspace(0, 2 * np.pi, 200)
        self._center_circle = np.column_stack([
            L / 2.0 + self.dimensions.center_circle_radius * np.cos(theta),
            W / 2.0 + self.dimensions.center_circle_radius * np.sin(theta),
        ]).astype(np.float64)

        # Touchlines and goal lines as polylines
        self._touchlines = [
            np.array([[0, 0], [L, 0]], dtype=np.float64),  # bottom touchline
            np.array([[0, W], [L, W]], dtype=np.float64),  # top touchline
            np.array([[0, 0], [0, W]], dtype=np.float64),  # left touchline
            np.array([[L, 0], [L, W]], dtype=np.float64),  # right touchline
        ]
        self._goal_lines = [
            np.array([[0, 0], [0, W]], dtype=np.float64),  # left goal line
            np.array([[L, 0], [L, W]], dtype=np.float64),  # right goal line
        ]

    @property
    def corners(self) -> np.ndarray:
        """Return 4x2 array of pitch corners in world coordinates."""
        return self._corners.copy()

    @property
    def penalty_areas(self) -> List[np.ndarray]:
        """Return two penalty area polygons."""
        return [pa.copy() for pa in self._penalty_areas]

    @property
    def goal_areas(self) -> List[np.ndarray]:
        """Return two goal area polygons."""
        return [ga.copy() for ga in self._goal_areas]

    @property
    def goals(self) -> List[np.ndarray]:
        """Return two goal polygons including depth."""
        return [g.copy() for g in self._goals]

    @property
    def halfway_line(self) -> np.ndarray:
        """Return halfway line endpoints."""
        return self._halfway_line.copy()

    @property
    def center_circle(self) -> np.ndarray:
        """Return center circle points."""
        return self._center_circle.copy()

    @property
    def center_spot(self) -> np.ndarray:
        """Return center spot coordinates."""
        return self._center_spot.copy()

    @property
    def touchlines(self) -> List[np.ndarray]:
        """Return all four touchlines."""
        return [tl.copy() for tl in self._touchlines]

    @property
    def goal_lines(self) -> List[np.ndarray]:
        """Return both goal lines."""
        return [gl.copy() for gl in self._goal_lines]

    def get_all_elements(self) -> Dict[str, np.ndarray]:
        """Return dict of all pitch elements for visualization."""
        return {
            "corners": self.corners,
            "penalty_areas": np.vstack(self.penalty_areas),
            "goal_areas": np.vstack(self.goal_areas),
            "goals": np.vstack(self.goals),
            "halfway_line": self.halfway_line,
            "center_circle": self.center_circle,
            "center_spot": self.center_spot,
            "touchlines": np.vstack(self.touchlines),
            "goal_lines": np.vstack(self.goal_lines),
        }

    def is_in_bounds(self, world_points: np.ndarray, margin: float = 0.0) -> np.ndarray:
        """
        Check which world points are inside pitch bounds.

        Args:
            world_points: (N, 2) array of world coordinates
            margin: margin in meters outside pitch considered valid

        Returns:
            (N,) boolean array
        """
        x = world_points[:, 0]
        y = world_points[:, 1]
        inside = (
            (-margin <= x) &
            (x <= self.dimensions.length + margin) &
            (-margin <= y) &
            (y <= self.dimensions.width + margin)
        )
        return inside

    def to_config(self) -> Dict:
        """Export pitch configuration to dict for JSON serialization."""
        return {
            "name": self.name,
            "length_m": self.dimensions.length,
            "width_m": self.dimensions.width,
            "goal_width_m": self.dimensions.goal_width,
            "penalty_area_length_m": self.dimensions.penalty_area_length,
            "penalty_area_width_m": self.dimensions.penalty_area_width,
            "goal_area_length_m": self.dimensions.goal_area_length,
            "goal_area_width_m": self.dimensions.goal_area_width,
            "center_circle_radius_m": self.dimensions.center_circle_radius,
        }

    @classmethod
    def from_config(cls, config: Dict) -> "PitchModel":
        """Create PitchModel from config dict."""
        dims = PitchDimensions(
            length=float(config.get("length_m", 105.0)),
            width=float(config.get("width_m", 68.0)),
            goal_width=float(config.get("goal_width_m", 7.32)),
            goal_depth=float(config.get("goal_depth_m", 2.0)),
            penalty_area_length=float(config.get("penalty_area_length_m", 16.5)),
            penalty_area_width=float(config.get("penalty_area_width_m", 40.32)),
            goal_area_length=float(config.get("goal_area_length_m", 5.5)),
            goal_area_width=float(config.get("goal_area_width_m", 18.32)),
            center_circle_radius=float(config.get("center_circle_radius_m", 9.15)),
            corner_arc_radius=float(config.get("corner_arc_radius_m", 1.0)),
            penalty_spot_distance=float(config.get("penalty_spot_distance_m", 11.0)),
        )
        return cls(dimensions=dims, name=config.get("name", "Custom Pitch"))