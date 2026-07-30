"""
Ball Analytics Visualization Module

Generates debug videos with overlays showing:
- Ball Speed
- Current Possession
- Current Pass
- Pass Distance
- Touch Count
"""

import logging
from typing import Dict, List, Optional
import numpy as np
import cv2

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)


class BallAnalyticsVisualizer:
    """
    Renders ball analytics debug overlays on video frames.
    """

    def __init__(
        self,
        output_path: str,
        canvas_size: tuple = (1050, 680),
        pitch_size_m: tuple = (105.0, 68.0)
    ):
        self.output_path = output_path
        self.canvas_size = canvas_size
        self.pitch_size_m = pitch_size_m
        self.scale_x = canvas_size[0] / pitch_size_m[0]
        self.scale_y = canvas_size[1] / pitch_size_m[1]

    def world_to_canvas(self, pos_m: tuple) -> tuple:
        x = int(pos_m[0] * self.scale_x)
        y = int(pos_m[1] * self.scale_y)
        return (x, y)

    def draw_overlay(
        self,
        frame: np.ndarray,
        points: List[Dict],
        frame_number: int,
        fps: float,
        possession_pct: Optional[Dict[str, float]] = None
    ) -> np.ndarray:
        canvas = np.zeros((self.canvas_size[1], self.canvas_size[0], 3), dtype=np.uint8)
        cv2.rectangle(canvas, (0, 0), (self.canvas_size[0]-1, self.canvas_size[1]-1), (255, 255, 255), 2)

        for p in points:
            if p.get('frame_number') != frame_number:
                continue

            pos = p.get('smoothed_world_position') or p.get('clean_world_position')
            if pos is None:
                continue

            cx, cy = self.world_to_canvas(pos)
            speed_kmh = p.get('ball_speed_kmh', 0)
            possession = p.get('possession')
            pass_info = p.get('pass_info')
            touch_count = p.get('touch_count', 0)

            # Ball marker
            cv2.circle(canvas, (cx, cy), 8, (0, 255, 255), -1)
            cv2.circle(canvas, (cx, cy), 10, (0, 255, 255), 2)

            # Speed label
            speed_label = f"{speed_kmh:.1f} km/h"
            cv2.putText(canvas, speed_label, (cx + 15, cy - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            cv2.putText(canvas, speed_label, (cx + 15, cy - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

            # Possession info
            if possession:
                poss_label = f"Poss: {possession.get('track_id', 'None')}"
                cv2.putText(canvas, poss_label, (cx + 15, cy + 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                cv2.putText(canvas, poss_label, (cx + 15, cy + 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

            # Pass info
            if pass_info:
                pass_label = f"Pass: {pass_info.get('distance_m', 0):.1f}m"
                cv2.putText(canvas, pass_label, (cx + 15, cy + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                cv2.putText(canvas, pass_label, (cx + 15, cy + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

            # Touch count
            touch_label = f"Touches: {touch_count}"
            cv2.putText(canvas, touch_label, (cx + 15, cy + 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            cv2.putText(canvas, touch_label, (cx + 15, cy + 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        # Possession percentages panel (top-left)
        if possession_pct:
            y_offset = 30
            cv2.rectangle(canvas, (10, 10), (200, 10 + y_offset * max(len(possession_pct) + 1, 3)), (0, 0, 0), -1)
            cv2.rectangle(canvas, (10, 10), (200, 10 + y_offset * max(len(possession_pct) + 1, 3)), (255, 255, 255), 2)
            for team, pct in possession_pct.items():
                if team == 'Free_Ball':
                    label = f"Free Ball: {pct}%"
                    color = (128, 128, 128)
                else:
                    label = f"{team}: {pct}%"
                    color = (0, 0, 255) if 'Red' in team else (255, 0, 0)
                cv2.putText(canvas, label, (20, 10 + y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                cv2.putText(canvas, label, (20, 10 + y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
                y_offset += 25

        # Frame info
        info = f"Frame: {frame_number} | Ball"
        cv2.putText(canvas, info, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(canvas, info, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)

        return canvas

    def generate_debug_video(
        self,
        all_points_by_frame: Dict[int, List[Dict]],
        fps: float,
        possession_pct: Optional[Dict[str, float]] = None
    ) -> bool:
        if not all_points_by_frame:
            logger.warning("No data provided for ball visualization")
            return False

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(self.output_path, fourcc, fps, self.canvas_size)

        max_frame = max(all_points_by_frame.keys())
        for frame_num in range(max_frame + 1):
            points = all_points_by_frame.get(frame_num, [])
            frame = np.zeros((self.canvas_size[1], self.canvas_size[0], 3), dtype=np.uint8)
            annotated = self.draw_overlay(frame, points, frame_num, fps, possession_pct=possession_pct)
            out.write(annotated)

        out.release()
        logger.info("Ball debug video saved to %s", self.output_path)
        return True
