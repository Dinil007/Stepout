"""
Pitch Visualizer Module

Renders top-down 2D tactical football pitch graphics using OpenCV.
Visualizes player positions (color-coded by team), movement trajectories,
ball positions, and real-time match frame metadata.
"""

import os
import logging
from typing import List, Dict, Tuple, Optional, Union
import cv2
import numpy as np

from app.homography.field_config import (
    PITCH_IMAGE_WIDTH,
    PITCH_IMAGE_HEIGHT,
    SCALE_X,
    SCALE_Y,
    CENTER_CIRCLE_RADIUS_PIXELS,
    FIELD_LENGTH_METERS,
    FIELD_WIDTH_METERS,
    PENALTY_BOX_LENGTH_METERS,
    PENALTY_BOX_WIDTH_METERS,
    GOAL_AREA_LENGTH_METERS,
    GOAL_AREA_WIDTH_METERS,
    PENALTY_SPOT_DISTANCE_METERS,
    GOAL_WIDTH_METERS
)

def _meters_to_pixels(position_m: Tuple[float, float]) -> Tuple[int, int]:
    """Converts real-world meter coordinates to pitch canvas pixels."""
    x_px = int(round(position_m[0] * SCALE_X))
    y_px = int(round(position_m[1] * SCALE_Y))
    return x_px, y_px
from app.homography.pitch_mapper import PlayerMapping

# Configure module logger
logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)


class PitchVisualizer:
    """
    Renders 2D tactical football pitch diagrams and visualizes mapped player tracking telemetry.
    """

    def __init__(
        self,
        width: int = PITCH_IMAGE_WIDTH,
        height: int = PITCH_IMAGE_HEIGHT,
        background_color: Tuple[int, int, int] = (20, 80, 20),      # Dark Green
        line_color: Tuple[int, int, int] = (255, 255, 255),         # White
        line_thickness: int = 2,
        player_radius: int = 12
    ):
        """
        Initializes the PitchVisualizer graphics engine.

        Args:
            width: Pitch canvas width in pixels.
            height: Pitch canvas height in pixels.
            background_color: BGR tuple for grass field background.
            line_color: BGR tuple for pitch lines.
            line_thickness: Line drawing thickness.
            player_radius: Circle radius for player rendering.
        """
        self.width = width
        self.height = height
        self.background_color = background_color
        self.line_color = line_color
        self.line_thickness = line_thickness
        self.player_radius = player_radius

        # Color mappings in BGR format
        self.team_colors: Dict[Union[int, str], Tuple[int, int, int]] = {
            0: (255, 100, 0),         # Blue (Team A)
            "Team A": (255, 100, 0),
            1: (0, 0, 230),           # Red (Team B)
            "Team B": (0, 0, 230),
            "Unknown": (200, 200, 200) # Gray/White
        }

        self.ball_color: Tuple[int, int, int] = (0, 230, 255)       # Yellow
        self.history_color: Tuple[int, int, int] = (180, 180, 180)   # Light Gray

        # Cache clean empty pitch image for fast rendering
        self.base_pitch_image: np.ndarray = self.create_pitch()

    def create_pitch(self) -> np.ndarray:
        """
        Draws an empty FIFA standard football pitch using OpenCV primitives.

        Returns:
            Grass pitch OpenCV image (BGR).
        """
        pitch = np.full((self.height, self.width, 3), self.background_color, dtype=np.uint8)

        # 1. Outer Touchlines Boundary
        cv2.rectangle(
            pitch,
            (0, 0),
            (self.width - 1, self.height - 1),
            self.line_color,
            self.line_thickness
        )

        # 2. Halfway Line
        center_x = self.width // 2
        cv2.line(
            pitch,
            (center_x, 0),
            (center_x, self.height),
            self.line_color,
            self.line_thickness
        )

        # 3. Center Circle & Center Spot
        center_y = self.height // 2
        cv2.circle(
            pitch,
            (center_x, center_y),
            CENTER_CIRCLE_RADIUS_PIXELS,
            self.line_color,
            self.line_thickness
        )
        cv2.circle(
            pitch,
            (center_x, center_y),
            4,
            self.line_color,
            -1
        )

        # 4. Left & Right Penalty Boxes
        box_width_px = int(PENALTY_BOX_LENGTH_METERS * SCALE_X)
        box_height_px = int(PENALTY_BOX_WIDTH_METERS * SCALE_Y)
        box_top_y = (self.height - box_height_px) // 2

        # Left Penalty Box
        cv2.rectangle(
            pitch,
            (0, box_top_y),
            (box_width_px, box_top_y + box_height_px),
            self.line_color,
            self.line_thickness
        )

        # Right Penalty Box
        cv2.rectangle(
            pitch,
            (self.width - box_width_px, box_top_y),
            (self.width, box_top_y + box_height_px),
            self.line_color,
            self.line_thickness
        )

        # 5. Goal Areas
        goal_area_w_px = int(GOAL_AREA_LENGTH_METERS * SCALE_X)
        goal_area_h_px = int(GOAL_AREA_WIDTH_METERS * SCALE_Y)
        goal_area_top_y = (self.height - goal_area_h_px) // 2

        # Left Goal Area
        cv2.rectangle(
            pitch,
            (0, goal_area_top_y),
            (goal_area_w_px, goal_area_top_y + goal_area_h_px),
            self.line_color,
            self.line_thickness
        )

        # Right Goal Area
        cv2.rectangle(
            pitch,
            (self.width - goal_area_w_px, goal_area_top_y),
            (self.width, goal_area_top_y + goal_area_h_px),
            self.line_color,
            self.line_thickness
        )

        # 6. Penalty Spots
        pen_spot_x = int(PENALTY_SPOT_DISTANCE_METERS * SCALE_X)
        cv2.circle(pitch, (pen_spot_x, center_y), 3, self.line_color, -1)
        cv2.circle(pitch, (self.width - pen_spot_x, center_y), 3, self.line_color, -1)

        return pitch

    def draw_player_history(
        self,
        pitch_image: np.ndarray,
        player_histories: Dict[int, List[Tuple[float, float]]]
    ) -> np.ndarray:
        """
        Draws player trajectory history polylines on the tactical pitch.

        Args:
            pitch_image: Base BGR pitch image.
            player_histories: Dict mapping track_id to historical (x, y) coordinates in meters.

        Returns:
            Annotated BGR pitch image.
        """
        if pitch_image is None or not player_histories:
            return pitch_image

        annotated = pitch_image.copy()

        for track_id, history in player_histories.items():
            if len(history) < 2:
                continue

            # Convert history from meters to pixels for canvas drawing
            px_history = [_meters_to_pixels(pos) for pos in history]
            pts = np.array(px_history, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(annotated, [pts], isClosed=False, color=self.history_color, thickness=1, lineType=cv2.LINE_AA)

        return annotated

    def draw_players(
        self,
        pitch_image: np.ndarray,
        mapped_players: List[PlayerMapping],
        possessor_id: Optional[int] = None
    ) -> np.ndarray:
        """
        Draws mapped players as team color-coded circles with ID labels.
        Highlights possessing player with a prominent yellow halo.

        Args:
            pitch_image: Tactical pitch BGR frame.
            mapped_players: List of PlayerMapping objects.
            possessor_id: Optional player track ID currently in possession.

        Returns:
            Annotated image.
        """
        if pitch_image is None:
            raise ValueError("Input pitch_image cannot be None.")

        annotated = pitch_image.copy()

        for player in mapped_players:
            # field_position is now in real-world meters; convert to canvas pixels
            px, py = _meters_to_pixels(player.field_position)

            # Clamp coordinates to canvas bounds
            px = max(0, min(self.width - 1, px))
            py = max(0, min(self.height - 1, py))

            color = self.team_colors.get(player.team_id, self.team_colors["Unknown"])

            # Possessor Yellow Halo
            if possessor_id is not None and player.track_id == possessor_id:
                cv2.circle(annotated, (px, py), self.player_radius + 6, (0, 255, 255), 3, cv2.LINE_AA)

            # Outer white ring
            cv2.circle(annotated, (px, py), self.player_radius, (255, 255, 255), 2, cv2.LINE_AA)

            # Filled team circle
            cv2.circle(annotated, (px, py), self.player_radius - 2, color, -1, cv2.LINE_AA)

            # Player track ID label
            label = str(player.track_id)
            cv2.putText(
                annotated,
                label,
                (px - 5, py + 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255, 255, 255),
                1,
                cv2.LINE_AA
            )

        return annotated

    def draw_ball(
        self,
        pitch_image: np.ndarray,
        ball_position: Tuple[float, float]
    ) -> np.ndarray:
        """
        Draws ball marker on 2D tactical pitch canvas.

        Args:
            pitch_image: Tactical pitch frame.
            ball_position: (x, y) coordinates in real-world meters.

        Returns:
            Annotated frame copy.
        """
        if pitch_image is None or ball_position is None:
            return pitch_image

        annotated = pitch_image.copy()
        bx, by = _meters_to_pixels(ball_position)

        cv2.circle(annotated, (bx, by), 8, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.circle(annotated, (bx, by), 6, (0, 255, 255), -1, cv2.LINE_AA)

        return annotated

    def draw_frame_information(
        self,
        pitch_image: np.ndarray,
        frame_number: int,
        player_count: int,
        timestamp: Optional[float] = None
    ) -> np.ndarray:
        """
        Overlays frame metadata headers in the top-left corner.

        Args:
            pitch_image: Tactical pitch frame.
            frame_number: Frame index.
            player_count: Number of tracked players on field.
            timestamp: Video timestamp in seconds.

        Returns:
            Annotated image.
        """
        if pitch_image is None:
            return pitch_image

        annotated = pitch_image.copy()

        text = f"Frame: {frame_number} | Players: {player_count}"
        if timestamp is not None:
            text += f" | Time: {timestamp:.2f}s"

        cv2.rectangle(annotated, (10, 10), (320, 38), (0, 0, 0), -1)
        cv2.putText(
            annotated,
            text,
            (18, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )

        return annotated

    def render(
        self,
        mapped_players: List[PlayerMapping],
        ball_position: Optional[Tuple[float, float]] = None,
        player_histories: Optional[Dict[int, List[Tuple[float, float]]]] = None,
        frame_number: int = 0,
        possessor_id: Optional[int] = None,
        active_pass: Optional[Dict] = None,
        active_shot: Optional[Dict] = None
    ) -> np.ndarray:
        """
        Executes the complete top-down tactical visualization rendering pipeline.

        Args:
            mapped_players: Mapped player DTO objects.
            ball_position: Optional (x, y) ball coordinate.
            player_histories: Optional player trajectories dict.
            frame_number: Current frame index.
            possessor_id: Optional player track ID currently in possession.
            active_pass: Optional active completed pass dict to overlay.
            active_shot: Optional active shot dict to overlay.

        Returns:
            Rendered 2D tactical pitch image.
        """
        # 1. Base Pitch
        frame = self.base_pitch_image.copy()

        # 2. Player Trajectories
        if player_histories:
            frame = self.draw_player_history(frame, player_histories)

        # 3. Connection Line (Possessing Player -> Ball)
        if possessor_id is not None and ball_position is not None and mapped_players:
            possessor_p = next((p for p in mapped_players if p.track_id == possessor_id), None)
            if possessor_p is not None:
                px, py = _meters_to_pixels(possessor_p.field_position)
                bx, by = _meters_to_pixels(ball_position)
                cv2.line(frame, (px, py), (bx, by), (0, 255, 255), 2, cv2.LINE_AA)

        # 4. Tactical Pass Arrow & Highlights
        if active_pass is not None and mapped_players:
            passer_id = active_pass.get("passer")
            receiver_id = active_pass.get("receiver")
            dist_m = active_pass.get("distance_m", 0.0)

            p_obj = next((p for p in mapped_players if p.track_id == passer_id), None)
            r_obj = next((p for p in mapped_players if p.track_id == receiver_id), None)

            if p_obj and r_obj:
                px, py = _meters_to_pixels(p_obj.field_position)
                rx, ry = _meters_to_pixels(r_obj.field_position)

                # Green arrow from passer to receiver
                cv2.arrowedLine(frame, (px, py), (rx, ry), (0, 255, 0), 2, cv2.LINE_AA, tipLength=0.15)

                # Passer & Receiver highlights
                cv2.circle(frame, (px, py), self.player_radius + 5, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.circle(frame, (rx, ry), self.player_radius + 5, (0, 255, 0), 2, cv2.LINE_AA)

                # Pass distance text
                mid_x, mid_y = (px + rx) // 2, (py + ry) // 2
                cv2.putText(frame, f"{dist_m:.1f}m", (mid_x, max(15, mid_y - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)

        # 5. Tactical Shot Arrow & Highlights (Red)
        if active_shot is not None:
            shooter_id = active_shot.get("player_id")
            s_dist = active_shot.get("distance_m", 0.0)
            s_speed = active_shot.get("ball_speed_mps", 0.0)

            # Find shooter on canvas or use launch position
            s_obj = next((p for p in mapped_players if p.track_id == shooter_id), None) if mapped_players else None
            if s_obj:
                sx, sy = _meters_to_pixels(s_obj.field_position)
            elif active_shot.get("launch_position"):
                l_pos = active_shot["launch_position"]
                sx, sy = int(round(l_pos[0] * SCALE_X)), int(round(l_pos[1] * SCALE_Y))
            else:
                sx, sy = self.width // 2, self.height // 2

            # Target Goal Center on 2D canvas (right goal: x=PITCH_IMAGE_WIDTH, y=PITCH_IMAGE_HEIGHT/2)
            gx = self.width if sx < self.width / 2 else 0
            gy = self.height // 2

            # Bright Red arrow to goal
            cv2.arrowedLine(frame, (sx, sy), (gx, gy), (0, 0, 255), 3, cv2.LINE_AA, tipLength=0.08)
            cv2.circle(frame, (sx, sy), self.player_radius + 7, (0, 0, 255), 3, cv2.LINE_AA)

            # Shot text overlay
            lbl = f"SHOT {s_dist:.1f}m | {s_speed:.1f}m/s"
            cv2.putText(frame, lbl, (sx - 40, max(20, sy - 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 255), 2, cv2.LINE_AA)

        # 6. Mapped Players (with Yellow possessor halo)
        if mapped_players:
            frame = self.draw_players(frame, mapped_players, possessor_id=possessor_id)

        # 7. Ball Position
        if ball_position:
            frame = self.draw_ball(frame, ball_position)

        # 8. Metadata HUD
        frame = self.draw_frame_information(
            frame,
            frame_number=frame_number,
            player_count=len(mapped_players) if mapped_players else 0
        )

        return frame

    def save_image(self, image: np.ndarray, output_path: str) -> str:
        """
        Saves a rendered tactical frame image to disk.

        Args:
            image: Rendered BGR frame image.
            output_path: Destination file path.

        Returns:
            Saved absolute path string.
        """
        if image is None:
            raise ValueError("Cannot save None image.")

        directory = os.path.dirname(output_path)
        if directory:
            os.makedirs(directory, exist_ok=True)

        cv2.imwrite(output_path, image)
        logger.info(f"Tactical pitch image saved to: {output_path}")
        return output_path

    @staticmethod
    def save_video_frame(writer: cv2.VideoWriter, image: np.ndarray) -> None:
        """Appends a rendered tactical frame to a VideoWriter instance."""
        if writer is None or not writer.isOpened():
            raise RuntimeError("VideoWriter is closed or uninitialized.")
        writer.write(image)

    @staticmethod
    def release(writer: cv2.VideoWriter) -> None:
        """Safely releases an active VideoWriter instance."""
        if writer is not None and writer.isOpened():
            writer.release()
            logger.info("VideoWriter released successfully.")
