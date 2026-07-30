from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Sequence

import numpy as np

from app.analytics.pressing_config import PressingConfig
from app.analytics.pressing_types import (
    PPDAWindow,
    PressingDetection,
    PressingMetrics,
    PressingSequence,
    PressingZone,
    PressureEvent,
)

logger = logging.getLogger(__name__)


@dataclass
class VisualizerConfig:
    """Configuration for pressing visualization appearance.

    Attributes:
        image_width: Output image width in pixels.
        image_height: Output image height in pixels.
        pitch_color: Background pitch color as BGR tuple.
        line_color: Pitch line color as BGR tuple.
        home_color: Home team player color as BGR tuple.
        away_color: Away team player color as BGR tuple.
        ball_color: Ball marker color as BGR tuple.
        pressure_line_color: Pressure line color as BGR tuple.
        successful_pressure_color: Color for successful pressure events.
        failed_pressure_color: Color for failed pressure events.
        high_press_color: Color for high press zone fill.
        mid_block_color: Color for mid block zone fill.
        low_block_color: Color for low block zone fill.
        text_color: Text/label color as BGR tuple.
        line_thickness: Thickness for pitch and pressure lines.
        player_radius: Radius for player markers.
        font_scale: Font scale for labels.
        font_thickness: Thickness for label text.
        alpha: Transparency for overlays (0-1).
        show_pitch: Whether to draw the pitch background.
        show_players: Whether to draw player markers.
        show_ball: Whether to draw ball position.
        show_pressure_events: Whether to draw pressure event markers.
        show_pressing_zones: Whether to draw pressing zone regions.
        show_pressing_lines: Whether to draw defender-to-attacker pressure lines.
        show_labels: Whether to draw text labels.
        show_confidence: Whether to draw confidence labels.
        show_sequences: Whether to draw pressing sequence overlays.
        show_heat_overlay: Whether to draw heat map overlay.
    """

    image_width: int = 1280
    image_height: int = 800
    pitch_color: tuple[int, int, int] = (34, 139, 34)
    line_color: tuple[int, int, int] = (255, 255, 255)
    home_color: tuple[int, int, int] = (255, 0, 0)
    away_color: tuple[int, int, int] = (0, 0, 255)
    ball_color: tuple[int, int, int] = (255, 255, 255)
    pressure_line_color: tuple[int, int, int] = (0, 255, 255)
    successful_pressure_color: tuple[int, int, int] = (0, 255, 0)
    failed_pressure_color: tuple[int, int, int] = (0, 0, 255)
    high_press_color: tuple[int, int, int] = (0, 0, 255)
    mid_block_color: tuple[int, int, int] = (0, 165, 255)
    low_block_color: tuple[int, int, int] = (0, 255, 255)
    text_color: tuple[int, int, int] = (255, 255, 255)
    line_thickness: int = 2
    player_radius: int = 8
    font_scale: float = 0.6
    font_thickness: int = 2
    alpha: float = 0.4
    show_pitch: bool = True
    show_players: bool = True
    show_ball: bool = True
    show_pressure_events: bool = True
    show_pressing_zones: bool = True
    show_pressing_lines: bool = True
    show_labels: bool = True
    show_confidence: bool = True
    show_sequences: bool = True
    show_heat_overlay: bool = False


class PressingVisualizer:
    """Converts pressing analysis into visual artifacts.

    This module performs no detection, metrics computation, validation,
    or API exposure. It only renders pressing intelligence onto images.

    Attributes:
        config: Visual appearance configuration.
        pressing_config: Pressing analysis configuration for zone boundaries.
        _frame_count: Internal frame counter for animation tracking.
    """

    def __init__(
        self,
        config: VisualizerConfig | None = None,
        pressing_config: PressingConfig | None = None,
    ) -> None:
        """Initialise the visualizer.

        Args:
            config: Visual appearance configuration.
            pressing_config: Pressing analysis config for zone thresholds.
        """
        self.config = config if config is not None else VisualizerConfig()
        self.pressing_config = pressing_config if pressing_config is not None else PressingConfig()
        self._frame_count: int = 0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _import_opencv(self):
        """Lazily import OpenCV.

        Returns:
            cv2 module.

        Raises:
            ImportError: If OpenCV is not available.
        """
        try:
            import cv2  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError("OpenCV is required for visualization.") from exc
        return cv2

    def _create_canvas(self) -> np.ndarray:
        """Create a blank pitch-coloured canvas.

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
        """Convert normalised pitch coordinates to pixel coordinates.

        Coordinates are expected in the range [0, 1] where (0, 0) is the
        top-left corner of the pitch.

        Args:
            x: Normalised x coordinate (0-1).
            y: Normalised y coordinate (0-1).
            canvas: Image array for size reference.

        Returns:
            Tuple of (pixel_x, pixel_y).
        """
        px = int(x * (self.config.image_width - 1))
        py = int(y * (self.config.image_height - 1))
        return px, py

    def _clamp_pixel(self, x: int, y: int) -> tuple[int, int]:
        """Clamp pixel coordinates to image bounds.

        Args:
            x: Pixel x coordinate.
            y: Pixel y coordinate.

        Returns:
            Clamped (x, y) tuple.
        """
        return (
            max(0, min(x, self.config.image_width - 1)),
            max(0, min(y, self.config.image_height - 1)),
        )

    # ------------------------------------------------------------------
    # Public drawing methods
    # ------------------------------------------------------------------

    def draw_pitch(self, canvas: np.ndarray | None = None) -> np.ndarray:
        """Draw a standard pitch outline.

        Args:
            canvas: Optional existing canvas. Creates new one if None.

        Returns:
            Image array with pitch drawn.
        """
        try:
            cv2 = self._import_opencv()
            if canvas is None:
                canvas = self._create_canvas()
            h, w = canvas.shape[:2]

            # Outer boundary
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
            cx, cy = w // 2, h // 2
            radius = min(w, h) // 10
            cv2.circle(
                canvas,
                (cx, cy),
                radius,
                self.config.line_color,
                self.config.line_thickness,
            )
            # Centre spot
            cv2.circle(
                canvas,
                (cx, cy),
                3,
                self.config.line_color,
                -1,
            )
            # Penalty areas (proportional)
            pen_w = w // 6
            pen_h = h // 5
            # Top penalty area
            cv2.rectangle(
                canvas,
                (w // 2 - pen_w // 2, 0),
                (w // 2 + pen_w // 2, pen_h),
                self.config.line_color,
                self.config.line_thickness,
            )
            # Bottom penalty area
            cv2.rectangle(
                canvas,
                (w // 2 - pen_w // 2, h - pen_h),
                (w // 2 + pen_w // 2, h - 1),
                self.config.line_color,
                self.config.line_thickness,
            )
            # Goal areas (proportional)
            goal_w = w // 12
            goal_h = h // 12
            # Top goal area
            cv2.rectangle(
                canvas,
                (w // 2 - goal_w // 2, 0),
                (w // 2 + goal_w // 2, goal_h),
                self.config.line_color,
                self.config.line_thickness,
            )
            # Bottom goal area
            cv2.rectangle(
                canvas,
                (w // 2 - goal_w // 2, h - goal_h),
                (w // 2 + goal_w // 2, h - 1),
                self.config.line_color,
                self.config.line_thickness,
            )

            logger.debug("Pitch drawn (%d x %d).", w, h)
        except Exception:
            logger.exception("Failed to draw pitch.")
            if canvas is None:
                canvas = self._create_canvas()
        return canvas

    def draw_players(
        self,
        player_positions: Sequence[tuple[int, float, float, int]] | None = None,
        canvas: np.ndarray | None = None,
    ) -> np.ndarray:
        """Draw player markers on the canvas.

        Each player is expected as a tuple of
        ``(player_id, x, y, team_id)`` where x, y are normalised [0, 1].

        Args:
            player_positions: Sequence of (player_id, x, y, team_id) tuples.
            canvas: Optional existing canvas.

        Returns:
            Image array with players drawn.
        """
        if player_positions is None or len(player_positions) == 0:
            logger.warning("No player positions provided; skipping draw_players.")
            return canvas if canvas is not None else self._create_canvas()
        try:
            cv2 = self._import_opencv()
            if canvas is None:
                canvas = self._create_canvas()

            for player_id, x, y, team_id in player_positions:
                px, py = self._normalize_to_pixel(x, y, canvas)
                px, py = self._clamp_pixel(px, py)
                color = self.config.home_color if team_id == 1 else self.config.away_color

                cv2.circle(
                    canvas,
                    (px, py),
                    self.config.player_radius,
                    color,
                    -1,
                )
                if self.config.show_labels:
                    cv2.putText(
                        canvas,
                        str(player_id),
                        (px - 5, py + 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        self.config.font_scale,
                        self.config.text_color,
                        self.config.font_thickness,
                    )
        except Exception:
            logger.exception("Failed to draw players.")
        return canvas if canvas is not None else self._create_canvas()

    def draw_ball(
        self,
        ball_position: tuple[float, float] | None = None,
        canvas: np.ndarray | None = None,
    ) -> np.ndarray:
        """Draw the ball position marker on the canvas.

        Args:
            ball_position: Optional (x, y) normalised coordinates of the ball.
            canvas: Optional existing canvas.

        Returns:
            Image array with ball drawn.
        """
        if ball_position is None:
            logger.warning("No ball position provided; skipping draw_ball.")
            return canvas if canvas is not None else self._create_canvas()
        try:
            cv2 = self._import_opencv()
            if canvas is None:
                canvas = self._create_canvas()

            bx, by = ball_position
            px, py = self._normalize_to_pixel(bx, by, canvas)
            px, py = self._clamp_pixel(px, py)

            # Outer ring
            cv2.circle(
                canvas,
                (px, py),
                self.config.player_radius // 2 + 2,
                (0, 0, 0),
                1,
            )
            # Ball fill
            cv2.circle(
                canvas,
                (px, py),
                self.config.player_radius // 2,
                self.config.ball_color,
                -1,
            )
        except Exception:
            logger.exception("Failed to draw ball.")
        return canvas if canvas is not None else self._create_canvas()

    def draw_pressure_events(
        self,
        events: Sequence[PressureEvent] | None = None,
        canvas: np.ndarray | None = None,
        attacker_positions: dict[int, tuple[float, float]] | None = None,
        defender_positions: dict[int, tuple[float, float]] | None = None,
    ) -> np.ndarray:
        """Draw pressure event markers and pressure lines on the canvas.

        This method draws a line from the defender to the attacker for each
        pressure event, and marks the midpoint with a coloured circle
        indicating success or failure.

        Args:
            events: Sequence of PressureEvent instances.
            canvas: Optional existing canvas.
            attacker_positions: Mapping of attacker_id -> (x, y) normalised.
            defender_positions: Mapping of defender_id -> (x, y) normalised.

        Returns:
            Image array with pressure events drawn.
        """
        if events is None or len(events) == 0:
            logger.warning("No pressure events provided; skipping draw_pressure_events.")
            return canvas if canvas is not None else self._create_canvas()
        try:
            cv2 = self._import_opencv()
            if canvas is None:
                canvas = self._create_canvas()

            for event in events:
                # Resolve positions
                attacker_pos = (attacker_positions or {}).get(event.attacker_id)
                defender_pos = (defender_positions or {}).get(event.defender_id)

                if attacker_pos is None or defender_pos is None:
                    logger.debug(
                        "Skipping event %d -> %d: missing positions.",
                        event.attacker_id,
                        event.defender_id,
                    )
                    continue

                ax, ay = self._normalize_to_pixel(*attacker_pos, canvas)
                dx, dy = self._normalize_to_pixel(*defender_pos, canvas)
                ax, ay = self._clamp_pixel(ax, ay)
                dx, dy = self._clamp_pixel(dx, dy)

                # Pressure line (defender -> attacker)
                if self.config.show_pressing_lines:
                    cv2.line(
                        canvas,
                        (dx, dy),
                        (ax, ay),
                        self.config.pressure_line_color,
                        self.config.line_thickness,
                    )

                # Event marker at midpoint
                mx, my = (dx + ax) // 2, (dy + ay) // 2
                color = (
                    self.config.successful_pressure_color
                    if event.successful
                    else self.config.failed_pressure_color
                )
                cv2.circle(canvas, (mx, my), 5, color, -1)

                if self.config.show_labels:
                    label = f"P:{event.distance:.1f}m"
                    cv2.putText(
                        canvas,
                        label,
                        (mx + 6, my),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        self.config.font_scale * 0.8,
                        self.config.text_color,
                        self.config.font_thickness,
                    )
        except Exception:
            logger.exception("Failed to draw pressure events.")
        return canvas if canvas is not None else self._create_canvas()

    def draw_pressing_zones(
        self,
        canvas: np.ndarray | None = None,
    ) -> np.ndarray:
        """Draw High Press / Mid Block / Low Block region overlays.

        The vertical boundaries are read from the associated
        ``pressing_config`` instance.

        Args:
            canvas: Optional existing canvas.

        Returns:
            Image array with pressing zones drawn.
        """
        try:
            cv2 = self._import_opencv()
            if canvas is None:
                canvas = self._create_canvas()
            h, w = canvas.shape[:2]

            overlay = canvas.copy()

            # High press region (top third)
            hp_y = int(self.pressing_config.high_press_line_y * h)
            cv2.rectangle(
                overlay,
                (0, 0),
                (w - 1, hp_y),
                self.config.high_press_color,
                -1,
            )

            # Mid block region
            mb_y_top = hp_y
            mb_y_bot = int(self.pressing_config.mid_block_line_y * h)
            cv2.rectangle(
                overlay,
                (0, mb_y_top),
                (w - 1, mb_y_bot),
                self.config.mid_block_color,
                -1,
            )

            # Low block region
            lb_y = int(self.pressing_config.low_block_line_y * h)
            cv2.rectangle(
                overlay,
                (0, mb_y_bot),
                (w - 1, lb_y),
                self.config.low_block_color,
                -1,
            )

            # Blend overlay
            cv2.addWeighted(
                overlay,
                self.config.alpha,
                canvas,
                1.0 - self.config.alpha,
                0,
                canvas,
            )

            # Draw zone labels on the left side
            if self.config.show_labels:
                label_font = cv2.FONT_HERSHEY_SIMPLEX
                lf_scale = self.config.font_scale
                lf_thick = self.config.font_thickness
                margin = 10

                cv2.putText(
                    canvas,
                    "HIGH PRESS",
                    (margin, hp_y // 2),
                    label_font,
                    lf_scale,
                    self.config.text_color,
                    lf_thick,
                )
                cv2.putText(
                    canvas,
                    "MID BLOCK",
                    (margin, (mb_y_top + mb_y_bot) // 2),
                    label_font,
                    lf_scale,
                    self.config.text_color,
                    lf_thick,
                )
                cv2.putText(
                    canvas,
                    "LOW BLOCK",
                    (margin, (mb_y_bot + lb_y) // 2),
                    label_font,
                    lf_scale,
                    self.config.text_color,
                    lf_thick,
                )

            # Draw dividing lines between zones
            cv2.line(
                canvas,
                (0, hp_y),
                (w - 1, hp_y),
                self.config.line_color,
                self.config.line_thickness,
            )
            cv2.line(
                canvas,
                (0, mb_y_bot),
                (w - 1, mb_y_bot),
                self.config.line_color,
                self.config.line_thickness,
            )

            logger.debug("Pressing zones drawn.")
        except Exception:
            logger.exception("Failed to draw pressing zones.")
        return canvas if canvas is not None else self._create_canvas()

    def draw_pressing_sequences(
        self,
        sequences: Sequence[PressingSequence] | None = None,
        canvas: np.ndarray | None = None,
        attacker_positions: dict[int, tuple[float, float]] | None = None,
        defender_positions: dict[int, tuple[float, float]] | None = None,
    ) -> np.ndarray:
        """Draw pressing sequence visualisation on the canvas.

        Each sequence is drawn as a polyline connecting the event midpoints
        in chronological order, with an arrow indicating direction if
        positions are available.

        Args:
            sequences: Sequence of PressingSequence instances.
            canvas: Optional existing canvas.
            attacker_positions: Mapping of attacker_id -> (x, y) normalised.
            defender_positions: Mapping of defender_id -> (x, y) normalised.

        Returns:
            Image array with sequences drawn.
        """
        if sequences is None or len(sequences) == 0:
            logger.warning("No pressing sequences provided; skipping draw_pressing_sequences.")
            return canvas if canvas is not None else self._create_canvas()
        try:
            cv2 = self._import_opencv()
            if canvas is None:
                canvas = self._create_canvas()

            overlay = canvas.copy()

            for seq in sequences:
                if len(seq.pressure_events) < 2:
                    continue

                midpoints: list[tuple[int, int]] = []
                for event in seq.pressure_events:
                    apos = (attacker_positions or {}).get(event.attacker_id)
                    dpos = (defender_positions or {}).get(event.defender_id)
                    if apos is None or dpos is None:
                        continue
                    ax, ay = self._normalize_to_pixel(*apos, canvas)
                    dx, dy = self._normalize_to_pixel(*dpos, canvas)
                    mx, my = (dx + ax) // 2, (dy + ay) // 2
                    midpoints.append((mx, my))

                if len(midpoints) < 2:
                    continue

                # Draw polyline trail for the sequence
                pts = np.array(midpoints, dtype=np.int32)
                cv2.polylines(
                    overlay,
                    [pts],
                    False,
                    self.config.successful_pressure_color,
                    self.config.line_thickness,
                )

                # Draw small arrow at the last segment
                if len(midpoints) >= 2:
                    last_pt = midpoints[-1]
                    prev_pt = midpoints[-2]
                    dx_vec = last_pt[0] - prev_pt[0]
                    dy_vec = last_pt[1] - prev_pt[1]
                    arrow_len = int(np.sqrt(dx_vec**2 + dy_vec**2))
                    if arrow_len > 0:
                        tip_length = 0.3  # fraction of total arrow length
                        cv2.arrowedLine(
                            overlay,
                            prev_pt,
                            last_pt,
                            self.config.successful_pressure_color,
                            self.config.line_thickness,
                            tipLength=tip_length,
                        )

                # Label the sequence with its ID
                if self.config.show_labels and len(midpoints) > 0:
                    start = midpoints[0]
                    label = f"S{seq.sequence_id}"
                    cv2.putText(
                        canvas,
                        label,
                        (start[0] + 5, start[1] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        self.config.font_scale,
                        self.config.text_color,
                        self.config.font_thickness,
                    )

            # Blend overlay
            cv2.addWeighted(
                overlay,
                self.config.alpha,
                canvas,
                1.0 - self.config.alpha,
                0,
                canvas,
            )

            logger.debug("Pressing sequences drawn (%d sequences).", len(sequences))
        except Exception:
            logger.exception("Failed to draw pressing sequences.")
        return canvas if canvas is not None else self._create_canvas()

    def annotate_metrics(
        self,
        metrics: PressingMetrics | None = None,
        ppda_window: PPDAWindow | None = None,
        detection: PressingDetection | None = None,
        canvas: np.ndarray | None = None,
    ) -> np.ndarray:
        """Annotate the canvas with team pressing metrics overlay.

        Displays PPDA, success rate, total pressures, and pressing style
        with confidence in the top-left corner of the image.

        Args:
            metrics: Aggregate PressingMetrics instance.
            ppda_window: Optional PPDAWindow for detailed PPDA display.
            detection: Optional PressingDetection for style + confidence.
            canvas: Optional existing canvas.

        Returns:
            Image array with metrics annotation drawn.
        """
        try:
            cv2 = self._import_opencv()
            if canvas is None:
                canvas = self._create_canvas()

            lines: list[str] = []
            if metrics is not None:
                lines.append(f"Total Pressures: {metrics.total_pressures}")
                lines.append(
                    f"Success Rate: {metrics.pressure_success_rate * 100:.1f}%"
                )
                lines.append(f"Avg Closing Speed: {metrics.average_closing_speed:.2f} m/s")
                lines.append(f"PPDA: {metrics.ppda:.2f}")
                lines.append(
                    f"High/Mid/Low: {metrics.high_press_count}/{metrics.mid_block_count}/{metrics.low_block_count}"
                )

            if ppda_window is not None:
                lines.append(
                    f"PPDA Window: {ppda_window.ppda:.2f} "
                    f"(passes: {ppda_window.passes_allowed}, "
                    f"actions: {ppda_window.defensive_actions})"
                )

            if detection is not None:
                style_label = detection.pressing_style.value.replace("_", " ").title()
                confidence = detection.confidence
                if self.config.show_confidence:
                    lines.append(f"Style: {style_label} ({confidence:.2f})")
                else:
                    lines.append(f"Style: {style_label}")

            # Draw semi-transparent background box
            if lines:
                box_h = len(lines) * 22 + 10
                cv2.rectangle(
                    canvas,
                    (5, 5),
                    (380, 5 + box_h),
                    (0, 0, 0),
                    -1,
                )
                # Make the box semi-transparent
                overlay = canvas.copy()
                cv2.rectangle(
                    overlay,
                    (5, 5),
                    (380, 5 + box_h),
                    (0, 0, 0),
                    -1,
                )
                cv2.addWeighted(
                    overlay,
                    0.5,
                    canvas,
                    0.5,
                    0,
                    canvas,
                )

                for i, line in enumerate(lines):
                    y_pos = 25 + i * 22
                    cv2.putText(
                        canvas,
                        line,
                        (15, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        self.config.font_scale,
                        self.config.text_color,
                        self.config.font_thickness,
                    )

            logger.debug("Metrics annotated (%d lines).", len(lines))
        except Exception:
            logger.exception("Failed to annotate metrics.")
        return canvas if canvas is not None else self._create_canvas()

    # ------------------------------------------------------------------
    # Frame-level rendering
    # ------------------------------------------------------------------

    def render_frame(
        self,
        player_positions: Sequence[tuple[int, float, float, int]] | None = None,
        ball_position: tuple[float, float] | None = None,
        pressure_events: Sequence[PressureEvent] | None = None,
        sequences: Sequence[PressingSequence] | None = None,
        metrics: PressingMetrics | None = None,
        ppda_window: PPDAWindow | None = None,
        detection: PressingDetection | None = None,
        attacker_positions: dict[int, tuple[float, float]] | None = None,
        defender_positions: dict[int, tuple[float, float]] | None = None,
        canvas: np.ndarray | None = None,
    ) -> np.ndarray:
        """Render a complete frame combining all enabled visual elements.

        Args:
            player_positions: Sequence of (player_id, x, y, team_id).
            ball_position: Optional (x, y) for the ball.
            pressure_events: Sequence of PressureEvent instances.
            sequences: Sequence of PressingSequence instances.
            metrics: Aggregate PressingMetrics.
            ppda_window: Optional PPDAWindow.
            detection: Optional PressingDetection.
            attacker_positions: Mapping of attacker_id -> (x, y).
            defender_positions: Mapping of defender_id -> (x, y).
            canvas: Optional existing canvas to render onto.

        Returns:
            Fully rendered image array.
        """
        try:
            # Start with canvas or create one
            if canvas is not None:
                frame = canvas.copy()
            else:
                frame = self._create_canvas()

            # 1. Draw pitch
            if self.config.show_pitch:
                frame = self.draw_pitch(canvas=frame)

            # 2. Draw pressing zones (behind players)
            if self.config.show_pressing_zones:
                frame = self.draw_pressing_zones(canvas=frame)

            # 3. Draw players
            if self.config.show_players:
                frame = self.draw_players(
                    player_positions=player_positions,
                    canvas=frame,
                )

            # 4. Draw ball
            if self.config.show_ball:
                frame = self.draw_ball(
                    ball_position=ball_position,
                    canvas=frame,
                )

            # 5. Draw pressure events (including lines)
            if self.config.show_pressure_events:
                frame = self.draw_pressure_events(
                    events=pressure_events,
                    canvas=frame,
                    attacker_positions=attacker_positions,
                    defender_positions=defender_positions,
                )

            # 6. Draw pressing sequences
            if self.config.show_sequences:
                frame = self.draw_pressing_sequences(
                    sequences=sequences,
                    canvas=frame,
                    attacker_positions=attacker_positions,
                    defender_positions=defender_positions,
                )

            # 7. Annotate metrics
            if metrics is not None or detection is not None:
                frame = self.annotate_metrics(
                    metrics=metrics,
                    ppda_window=ppda_window,
                    detection=detection,
                    canvas=frame,
                )

            self._frame_count += 1
            logger.debug("Frame %d rendered.", self._frame_count)
            return frame

        except Exception:
            logger.exception("Failed to render frame; returning original canvas.")
            if canvas is not None:
                return canvas.copy()
            return self._create_canvas()

    def render_animation(
        self,
        frames: list[dict[str, Any]] | None = None,
    ) -> list[np.ndarray]:
        """Render a sequence of frames for animation or video output.

        Each element in ``frames`` is a dictionary with the same keyword
        arguments accepted by :meth:`render_frame`.

        Args:
            frames: List of keyword-argument dicts for each frame.

        Returns:
            List of rendered image arrays.

        Raises:
            ValueError: If frames is None or empty.
        """
        if frames is None or len(frames) == 0:
            raise ValueError("frames must be a non-empty list of frame data dictionaries.")

        rendered: list[np.ndarray] = []
        for idx, frame_data in enumerate(frames):
            try:
                # Determine canvas: use previous frame for smooth transitions if available
                prev_canvas = rendered[-1] if idx > 0 and len(rendered) > 0 else None
                frame = self.render_frame(
                    **frame_data,
                    canvas=prev_canvas,
                )
                rendered.append(frame)
            except Exception:
                logger.exception("Failed to render animation frame %d; falling back to blank.", idx)
                rendered.append(self._create_canvas())

        logger.info("Animation rendered with %d frames.", len(rendered))
        return rendered

    def reset(self) -> None:
        """Reset the visualizer internal state (frame counter)."""
        self._frame_count = 0
        logger.debug("Visualizer state reset.")