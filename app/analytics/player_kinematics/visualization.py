"""
Player Kinematics Visualization Module

Generates debug videos with overlays showing:
- Player ID
- Current Speed
- Distance Covered
- Sprint Indicator
- Movement Arrow
"""

import logging
from typing import Dict, List, Optional
import numpy as np
import cv2

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)


class KinematicsVisualizer:
    """
    Renders kinematics debug overlays on video frames.
    """

    def __init__(
        self,
        output_path: str,
        canvas_size: tuple = (1050, 680),
        pitch_size_m: tuple = (105.0, 68.0)
    ):
        """
        Initializes KinematicsVisualizer.

        Args:
            output_path: Path to save debug video.
            canvas_size: Output canvas size (width, height) in pixels.
            pitch_size_m: Real pitch dimensions (length, width) in meters.
        """
        self.output_path = output_path
        self.canvas_size = canvas_size
        self.pitch_size_m = pitch_size_m
        self.scale_x = canvas_size[0] / pitch_size_m[0]
        self.scale_y = canvas_size[1] / pitch_size_m[1]

    def world_to_canvas(self, pos_m: tuple) -> tuple:
        """Converts world coordinates to canvas pixel coordinates."""
        x = int(pos_m[0] * self.scale_x)
        y = int(pos_m[1] * self.scale_y)
        return (x, y)

    def draw_overlay(
        self,
        frame: np.ndarray,
        points: List[Dict],
        frame_number: int,
        fps: float
    ) -> np.ndarray:
        """
        Draws kinematics overlay on a video frame.

        Args:
            frame: Input video frame.
            points: List of all player trajectory points for current frame.
            frame_number: Current frame number.
            fps: Video frame rate.

        Returns:
            Annotated frame.
        """
        canvas = np.zeros((self.canvas_size[1], self.canvas_size[0], 3), dtype=np.uint8)

        # Draw pitch outline
        cv2.rectangle(canvas, (0, 0), (self.canvas_size[0]-1, self.canvas_size[1]-1), (255, 255, 255), 2)

        for p in points:
            if p.get('frame_number') != frame_number:
                continue

            pos = p.get('smoothed_world_position') or p.get('clean_world_position')
            if pos is None:
                continue

            cx, cy = self.world_to_canvas(pos)
            track_id = p.get('track_id', 0)
            speed_kmh = p.get('speed_kmh', 0)
            dist = p.get('distance_m', 0)
            is_sprinting = p.get('is_sprinting', False)
            vx = p.get('vx', 0)
            vy = p.get('vy', 0)

            color = (0, 255, 0)
            if is_sprinting:
                color = (0, 0, 255)

            # Draw movement arrow
            if vx != 0 or vy != 0:
                end_x = cx + int(vx * 2)
                end_y = cy + int(vy * 2)
                cv2.arrowedLine(canvas, (cx, cy), (end_x, end_y), color, 2)

            # Draw player marker
            cv2.circle(canvas, (cx, cy), 6, color, -1)
            cv2.circle(canvas, (cx, cy), 8, color, 2)

            # Text labels
            label = f"ID:{track_id}"
            speed_label = f"{speed_kmh:.1f}km/h"
            dist_label = f"{dist:.1f}m"

            cv2.putText(canvas, label, (cx + 12, cy - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            cv2.putText(canvas, label, (cx + 12, cy - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            cv2.putText(canvas, speed_label, (cx + 12, cy - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            cv2.putText(canvas, speed_label, (cx + 12, cy - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            cv2.putText(canvas, dist_label, (cx + 12, cy + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            cv2.putText(canvas, dist_label, (cx + 12, cy + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

            # Sprint indicator
            if is_sprinting:
                cv2.putText(canvas, "SPRINT", (cx + 12, cy + 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # Frame info
        info = f"Frame: {frame_number} | Players: {len(points)}"
        cv2.putText(canvas, info, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(canvas, info, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)

        return canvas

    def generate_debug_video(
        self,
        all_points_by_frame: Dict[int, List[Dict]],
        fps: float
    ) -> bool:
        """
        Generates debug video from frame-by-frame player data.

        Args:
            all_points_by_frame: Dict mapping frame_number to list of player dicts.
            fps: Output video frame rate.

        Returns:
            True if video was generated successfully.
        """
        if not all_points_by_frame:
            logger.warning("No data provided for visualization")
            return False

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(self.output_path, fourcc, fps, self.canvas_size)

        max_frame = max(all_points_by_frame.keys())
        for frame_num in range(max_frame + 1):
            points = all_points_by_frame.get(frame_num, [])
            frame = np.zeros((self.canvas_size[1], self.canvas_size[0], 3), dtype=np.uint8)
            annotated = self.draw_overlay(frame, points, frame_num, fps)
            out.write(annotated)

        out.release()
        logger.info("Debug video saved to %s", self.output_path)
        return True