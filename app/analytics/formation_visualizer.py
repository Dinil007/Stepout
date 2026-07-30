from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from app.analytics.formation_engine import FormationAnalysisResult
from app.analytics.formation_types import (
    FormationMetrics,
    PlayerPosition,
)

logger = logging.getLogger(__name__)


@dataclass
class VisualizerConfig:
    """Configuration for visualization appearance.

    Attributes:
        image_width: Output image width in pixels.
        image_height: Output image height in pixels.
        pitch_color: Background pitch color as BGR tuple.
        line_color: Pitch line color as BGR tuple.
        home_color: Home team player color as BGR tuple.
        away_color: Away team player color as BGR tuple.
        goalkeeper_color: Goalkeeper color as BGR tuple.
        text_color: Text/label color as BGR tuple.
        line_thickness: Thickness for pitch lines.
        marker_radius: Radius for player markers.
        font_scale: Font scale for labels.
        font_thickness: Thickness for label text.
        alpha: Transparency for overlays (0-1).
        show_labels: Whether to draw labels/jersey numbers.
    """

    image_width: int = 1280
    image_height: int = 800
    pitch_color: tuple[int, int, int] = (34, 139, 34)
    line_color: tuple[int, int, int] = (255, 255, 255)
    home_color: tuple[int, int, int] = (255, 0, 0)
    away_color: tuple[int, int, int] = (0, 0, 255)
    goalkeeper_color: tuple[int, int, int] = (0, 255, 255)
    text_color: tuple[int, int, int] = (255, 255, 255)
    line_thickness: int = 2
    marker_radius: int = 8
    font_scale: float = 0.6
    font_thickness: int = 2
    alpha: float = 0.6
    show_labels: bool = True


class FormationVisualizer:
    """Converts formation analysis into visual artifacts.

    This module performs no detection, metrics computation, validation,
    or API exposure. It only renders tactical information onto images.

    Attributes:
        config: Visual appearance configuration.
    """

    def __init__(self, config: VisualizerConfig | None = None) -> None:
        self.config = config if config is not None else VisualizerConfig()

    def _create_canvas(self) -> np.ndarray:
        """Create a blank pitch-colored canvas.

        Returns:
            NumPy array image.
        """
        canvas = np.zeros(
            (self.config.image_height, self.config.image_width, 3), dtype=np.uint8
        )
        canvas[:] = self.config.pitch_color
        return canvas

    def _normalize_to_pixel(
        self, x: float, y: float, canvas: np.ndarray
    ) -> tuple[int, int]:
        """Convert normalized pitch coordinates to pixel coordinates.

        Args:
            x: Normalized x coordinate (0-1).
            y: Normalized y coordinate (0-1).
            canvas: Image array for size reference.

        Returns:
            Tuple of (pixel_x, pixel_y).
        """
        px = int(x * (self.config.image_width - 1))
        py = int(y * (self.config.image_height - 1))
        return px, py

    def draw_pitch(self, canvas: np.ndarray | None = None) -> np.ndarray:
        """Draw a FIFA-standard pitch outline.

        Args:
            canvas: Optional existing canvas. Creates new one if None.

        Returns:
            Image array with pitch drawn.
        """
        if canvas is None:
            canvas = self._create_canvas()
        h, w = canvas.shape[:2]
        # Boundary
        cv2 = self._import_opencv()
        cv2.rectangle(
            canvas,
            (0, 0),
            (w - 1, h - 1),
            self.config.line_color,
            self.config.line_thickness,
        )
        # Halfway line
        cv2.line(
            canvas,
            (0, h // 2),
            (w - 1, h // 2),
            self.config.line_color,
            self.config.line_thickness,
        )
        # Centre circle
        cv2.circle(
            canvas,
            (w // 2, h // 2),
            min(w, h) // 10,
            self.config.line_color,
            self.config.line_thickness,
        )
        logger.info("Pitch drawn.")
        return canvas

    def _import_opencv(self):
        """Lazily import OpenCV.

        Returns:
            cv2 module.
        """
        try:
            import cv2
        except ImportError as exc:
            raise ImportError("OpenCV is required for visualization.") from exc
        return cv2

    def draw_player_positions(
        self,
        players: Sequence[PlayerPosition],
        canvas: np.ndarray | None = None,
    ) -> np.ndarray:
        """Draw player markers on the canvas.

        Args:
            players: Sequence of PlayerPosition instances.
            canvas: Optional existing canvas.

        Returns:
            Image array with players drawn.
        """
        cv2 = self._import_opencv()
        if canvas is None:
            canvas = self._create_canvas()
        for player in players:
            px, py = self._normalize_to_pixel(player.x, player.y, canvas)
            color = self.config.goalkeeper_color if player.is_goalkeeper else (
                self.config.home_color if player.team_id == 1 else self.config.away_color
            )
            cv2.circle(
                canvas,
                (px, py),
                self.config.marker_radius,
                color,
                -1,
            )
            if self.config.show_labels:
                cv2.putText(
                    canvas,
                    str(player.jersey_number),
                    (px - 5, py + 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    self.config.font_scale,
                    self.config.text_color,
                    self.config.font_thickness,
                )
        return canvas

    def draw_detected_formation(
        self,
        result: FormationAnalysisResult,
        canvas: np.ndarray | None = None,
    ) -> np.ndarray:
        """Draw detected formation label on canvas.

        Args:
            result: FormationAnalysisResult to visualize.
            canvas: Optional existing canvas.

        Returns:
            Image array with formation label drawn.
        """
        cv2 = self._import_opencv()
        if canvas is None:
            canvas = self._create_canvas()
        text = f"{result.detected_formation} ({result.confidence:.2f})"
        cv2.putText(
            canvas,
            text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            self.config.font_scale * 1.2,
            self.config.text_color,
            self.config.font_thickness,
        )
        return canvas

    def draw_template_overlay(
        self,
        template_name: str,
        players: Sequence[PlayerPosition],
        canvas: np.ndarray | None = None,
    ) -> np.ndarray:
        """Overlay template positions with connecting lines.

        Args:
            template_name: Name of formation template.
            players: Sequence of PlayerPosition instances.
            canvas: Optional existing canvas.

        Returns:
            Image array with template overlay drawn.
        """
        cv2 = self._import_opencv()
        if canvas is None:
            canvas = self._create_canvas()
        # Draw thin connecting lines to nearest template positions would go here;
        # as a placeholder, draw faint circles at player positions.
        overlay = canvas.copy()
        for player in players:
            px, py = self._normalize_to_pixel(player.x, player.y, canvas)
            cv2.circle(
                overlay,
                (px, py),
                self.config.marker_radius + 2,
                (255, 255, 0),
                1,
            )
        cv2.addWeighted(
            overlay,
            self.config.alpha,
            canvas,
            1 - self.config.alpha,
            0,
            canvas,
        )
        return canvas

    def draw_centroid(
        self,
        metrics: FormationMetrics,
        canvas: np.ndarray | None = None,
    ) -> np.ndarray:
        """Draw team centroid marker.

        Args:
            metrics: FormationMetrics containing centroid.
            canvas: Optional existing canvas.

        Returns:
            Image array with centroid drawn.
        """
        cv2 = self._import_opencv()
        if canvas is None:
            canvas = self._create_canvas()
        px, py = self._normalize_to_pixel(metrics.centroid_x, metrics.centroid_y, canvas)
        cv2.drawMarker(
            canvas,
            (px, py),
            (0, 255, 0),
            cv2.MARKER_TILTED_CROSS,
            self.config.marker_radius * 2,
            self.config.line_thickness,
        )
        return canvas

    def draw_team_width(
        self,
        metrics: FormationMetrics,
        canvas: np.ndarray | None = None,
    ) -> np.ndarray:
        """Visualize team width as a horizontal line at centroid y.

        Args:
            metrics: FormationMetrics containing width and centroid.
            canvas: Optional existing canvas.

        Returns:
            Image array with width line drawn.
        """
        cv2 = self._import_opencv()
        if canvas is None:
            canvas = self._create_canvas()
        cx, cy = self._normalize_to_pixel(metrics.centroid_x, metrics.centroid_y, canvas)
        half = int(metrics.team_width * self.config.image_width / 2)
        pt1 = (max(0, cx - half), cy)
        pt2 = (min(self.config.image_width - 1, cx + half), cy)
        cv2.line(canvas, pt1, pt2, (255, 255, 0), self.config.line_thickness)
        return canvas

    def draw_team_length(
        self,
        metrics: FormationMetrics,
        canvas: np.ndarray | None = None,
    ) -> np.ndarray:
        """Visualize team length as a vertical line at centroid x.

        Args:
            metrics: FormationMetrics containing length and centroid.
            canvas: Optional existing canvas.

        Returns:
            Image array with length line drawn.
        """
        cv2 = self._import_opencv()
        if canvas is None:
            canvas = self._create_canvas()
        cx, cy = self._normalize_to_pixel(metrics.centroid_x, metrics.centroid_y, canvas)
        half = int(metrics.team_length * self.config.image_height / 2)
        pt1 = (cx, max(0, cy - half))
        pt2 = (cx, min(self.config.image_height - 1, cy + half))
        cv2.line(canvas, pt1, pt2, (0, 255, 255), self.config.line_thickness)
        return canvas

    def draw_compactness(
        self,
        metrics: FormationMetrics,
        canvas: np.ndarray | None = None,
    ) -> np.ndarray:
        """Draw compactness score as text.

        Args:
            metrics: FormationMetrics containing compactness.
            canvas: Optional existing canvas.

        Returns:
            Image array with compactness text drawn.
        """
        cv2 = self._import_opencv()
        if canvas is None:
            canvas = self._create_canvas()
        text = f"Compactness: {metrics.compactness:.2f}"
        cv2.putText(
            canvas,
            text,
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            self.config.font_scale,
            self.config.text_color,
            self.config.font_thickness,
        )
        return canvas

    def draw_convex_hull(
        self,
        players: Sequence[PlayerPosition],
        canvas: np.ndarray | None = None,
    ) -> np.ndarray:
        """Draw convex hull around player positions.

        Args:
            players: Sequence of PlayerPosition instances.
            canvas: Optional existing canvas.

        Returns:
            Image array with convex hull drawn.
        """
        cv2 = self._import_opencv()
        if canvas is None:
            canvas = self._create_canvas()
        pts = np.array(
            [self._normalize_to_pixel(p.x, p.y, canvas) for p in players], dtype=np.int32
        )
        if len(pts) >= 3:
            cv2.polylines(canvas, [pts], True, (0, 255, 0), self.config.line_thickness)
        return canvas

    def draw_defensive_line(
        self,
        metrics: FormationMetrics,
        canvas: np.ndarray | None = None,
    ) -> np.ndarray:
        """Draw defensive line across pitch width.

        Args:
            metrics: FormationMetrics with defensive_line.
            canvas: Optional existing canvas.

        Returns:
            Image array with defensive line drawn.
        """
        cv2 = self._import_opencv()
        if canvas is None:
            canvas = self._create_canvas()
        y = int(metrics.defensive_line * (self.config.image_height - 1))
        cv2.line(
            canvas,
            (0, y),
            (self.config.image_width - 1, y),
            self.config.line_color,
            self.config.line_thickness,
        )
        return canvas

    def draw_midfield_line(
        self,
        metrics: FormationMetrics,
        canvas: np.ndarray | None = None,
    ) -> np.ndarray:
        """Draw midfield line across pitch width.

        Args:
            metrics: FormationMetrics with midfield_line.
            canvas: Optional existing canvas.

        Returns:
            Image array with midfield line drawn.
        """
        cv2 = self._import_opencv()
        if canvas is None:
            canvas = self._create_canvas()
        y = int(metrics.midfield_line * (self.config.image_height - 1))
        cv2.line(
            canvas,
            (0, y),
            (self.config.image_width - 1, y),
            self.config.line_color,
            self.config.line_thickness,
        )
        return canvas

    def draw_forward_line(
        self,
        metrics: FormationMetrics,
        canvas: np.ndarray | None = None,
    ) -> np.ndarray:
        """Draw forward line across pitch width.

        Args:
            metrics: FormationMetrics with forward_line.
            canvas: Optional existing canvas.

        Returns:
            Image array with forward line drawn.
        """
        cv2 = self._import_opencv()
        if canvas is None:
            canvas = self._create_canvas()
        y = int(metrics.forward_line * (self.config.image_height - 1))
        cv2.line(
            canvas,
            (0, y),
            (self.config.image_width - 1, y),
            self.config.line_color,
            self.config.line_thickness,
        )
        return canvas

    def draw_labels(
        self,
        labels: Sequence[tuple[str, tuple[int, int]]],
        canvas: np.ndarray | None = None,
    ) -> np.ndarray:
        """Draw text labels at specified pixel coordinates.

        Args:
            labels: Sequence of (text, (x, y)) tuples.
            canvas: Optional existing canvas.

        Returns:
            Image array with labels drawn.
        """
        cv2 = self._import_opencv()
        if canvas is None:
            canvas = self._create_canvas()
        for text, (x, y) in labels:
            cv2.putText(
                canvas,
                str(text),
                (x, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                self.config.font_scale,
                self.config.text_color,
                self.config.font_thickness,
            )
        return canvas

    def draw_transition(
        self,
        previous_formation: str,
        new_formation: str,
        confidence: float,
        canvas: np.ndarray | None = None,
    ) -> np.ndarray:
        """Draw formation transition indicator.

        Args:
            previous_formation: Previous formation name.
            new_formation: New formation name.
            confidence: Current confidence score.
            canvas: Optional existing canvas.

        Returns:
            Image array with transition drawn.
        """
        cv2 = self._import_opencv()
        if canvas is None:
            canvas = self._create_canvas()
        text = f"{previous_formation} -> {new_formation} ({confidence:.2f})"
        cv2.putText(
            canvas,
            text,
            (10, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            self.config.font_scale,
            self.config.text_color,
            self.config.font_thickness,
        )
        return canvas

    def create_frame(
        self,
        players: Sequence[PlayerPosition],
        metrics: FormationMetrics,
        detection: FormationAnalysisResult | None = None,
        previous_formation: str = "",
    ) -> np.ndarray:
        """Create a complete frame rendering of the formation analysis.

        Args:
            players: Sequence of PlayerPosition instances.
            metrics: FormationMetrics for the team.
            detection: Optional FormationAnalysisResult for overlay info.
            previous_formation: Optional previous formation for transition view.

        Returns:
            Complete rendered image array.
        """
        canvas = self.draw_pitch()
        canvas = self.draw_player_positions(players, canvas)
        canvas = self.draw_convex_hull(players, canvas)
        canvas = self.draw_centroid(metrics, canvas)
        canvas = self.draw_team_width(metrics, canvas)
        canvas = self.draw_team_length(metrics, canvas)
        canvas = self.draw_defensive_line(metrics, canvas)
        canvas = self.draw_midfield_line(metrics, canvas)
        canvas = self.draw_forward_line(metrics, canvas)
        canvas = self.draw_compactness(metrics, canvas)
        if detection is not None:
            canvas = self.draw_detected_formation(detection, canvas)
            if previous_formation:
                canvas = self.draw_transition(
                    previous_formation,
                    detection.detected_formation,
                    detection.confidence,
                    canvas,
                )
        logger.info("Frame rendered.")
        return canvas

    def create_animation(
        self,
        frames: Sequence[Sequence[PlayerPosition]],
        metrics_list: Sequence[FormationMetrics],
        detections: Sequence[FormationAnalysisResult | None],
    ) -> list[np.ndarray]:
        """Create a sequence of frames for animation/video.

        Args:
            frames: Sequence of player lists.
            metrics_list: Sequence of FormationMetrics.
            detections: Sequence of optional FormationAnalysisResult.

        Returns:
            List of rendered image arrays.
        """
        if len(frames) != len(metrics_list) or len(frames) != len(detections):
            raise ValueError("frames, metrics_list, and detections must have the same length.")
        rendered = []
        for idx, (players, metrics, detection) in enumerate(zip(frames, metrics_list, detections)):
            previous = detections[idx - 1].detected_formation if idx > 0 and detections[idx - 1] else ""
            frame = self.create_frame(players, metrics, detection, previous_formation=previous)
            rendered.append(frame)
        logger.info("Animation created with %d frames.", len(rendered))
        return rendered