"""
Ball Path Renderer Module

Renders professional-quality ball trajectory visualizations.
Uses team possession semantics instead of tracker IDs.
"""

import logging
from typing import Dict, List, Optional, Tuple
from collections import deque
import numpy as np
import cv2

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)


class BallPathRenderer:
    """
    Production-quality ball path renderer with team-colour semantics,
    temporal stability, and configurable styling.
    """

    def __init__(self, config: Dict):
        bpr_cfg = config.get('visualization', {}).get('ball_path_renderer', {})
        pitch = config.get('pitch', {})

        self.enabled = bpr_cfg.get('enabled', True)
        self.history_length = int(bpr_cfg.get('history_length', 100))
        self.line_thickness = int(bpr_cfg.get('line_thickness', 3))
        self.fade_enabled = bool(bpr_cfg.get('fade_enabled', True))
        self.fade_alpha = float(bpr_cfg.get('fade_alpha', 0.6))
        self.arrow_size = int(bpr_cfg.get('arrow_size', 8))
        self.min_possession_frames = int(bpr_cfg.get('min_possession_frames', 5))
        self.speed_thickness_threshold_kmh = float(bpr_cfg.get('speed_thickness_threshold_kmh', 20.0))
        self.speed_max_thickness_kmh = float(bpr_cfg.get('speed_max_thickness_kmh', 40.0))

        self.team_colors = {}
        for key, val in bpr_cfg.get('team_colors', {}).items():
            self.team_colors[key] = tuple(val)
        self.default_team_color = tuple(bpr_cfg.get('team_colors', {}).get('unknown', [128, 128, 128]))
        self.loose_ball_color = tuple(bpr_cfg.get('team_colors', {}).get('loose_ball', [0, 255, 0]))
        self.pass_color = tuple(bpr_cfg.get('pass_color', [0, 255, 255]))
        self.touch_color = tuple(bpr_cfg.get('touch_color', [0, 200, 200]))
        self.low_confidence_color = tuple(bpr_cfg.get('team_colors', {}).get('unknown', [128, 128, 128]))

        self.canvas_size = (
            int(pitch.get('canvas_width', 1050)),
            int(pitch.get('canvas_height', 680))
        )
        self.pitch_size = (
            float(pitch.get('length_m', 105.0)),
            float(pitch.get('width_m', 68.0))
        )
        self.scale_x = self.canvas_size[0] / self.pitch_size[0]
        self.scale_y = self.canvas_size[1] / self.pitch_size[1]

        self.history: deque = deque(maxlen=self.history_length)
        self.current_color = self.default_team_color
        self.color_switch_counter = 0
        self.prev_team_id = None

    def reset(self) -> None:
        self.history.clear()
        self.current_color = self.default_team_color
        self.color_switch_counter = 0
        self.prev_team_id = None

    def world_to_canvas(self, pos_m: Tuple[float, float]) -> Tuple[int, int]:
        x = int(pos_m[0] * self.scale_x)
        y = int(pos_m[1] * self.scale_y)
        return (x, y)

    def _get_color_for_state(
        self,
        team_id: Optional[str],
        has_possession: bool,
        possession_confidence: float,
        confidence_threshold: float = 0.5
    ) -> Tuple[int, int, int]:
        if not has_possession or possession_confidence < confidence_threshold:
            return self.loose_ball_color if has_possession else self.low_confidence_color

        if team_id in self.team_colors:
            return self.team_colors[team_id]

        return self.default_team_color

    def update(
        self,
        frame_number: int,
        timestamp: float,
        pixel_position: Optional[Tuple[int, int]],
        world_position: Optional[Tuple[float, float]],
        team_id: Optional[str],
        player_id: Optional[int],
        has_possession: bool,
        possession_confidence: float,
        ball_speed_kmh: float = 0.0,
        is_pass: bool = False,
        is_touch: bool = False
    ) -> None:
        if not self.enabled or world_position is None or pixel_position is None:
            return

        target_color = self._get_color_for_state(team_id, has_possession, possession_confidence)

        if target_color != self.current_color:
            self.color_switch_counter += 1
            if self.color_switch_counter >= self.min_possession_frames:
                self.current_color = target_color
                self.color_switch_counter = 0
        else:
            self.color_switch_counter = 0

        entry = {
            'frame_number': frame_number,
            'timestamp': timestamp,
            'pixel_position': pixel_position,
            'world_position': world_position,
            'team_id': team_id,
            'player_id': player_id,
            'has_possession': has_possession,
            'possession_confidence': possession_confidence,
            'ball_speed_kmh': ball_speed_kmh,
            'is_pass': is_pass,
            'is_touch': is_touch,
            'color': self.current_color
        }
        self.history.append(entry)

    def render(self, frame: Optional[np.ndarray] = None) -> np.ndarray:
        if frame is None:
            frame = np.zeros((self.canvas_size[1], self.canvas_size[0], 3), dtype=np.uint8)
        else:
            frame = frame.copy()

        if len(self.history) < 2:
            return frame

        pts = [entry['pixel_position'] for entry in self.history]
        colors = [entry['color'] for entry in self.history]
        speeds = [entry.get('ball_speed_kmh', 0.0) for entry in self.history]
        passes = [entry.get('is_pass', False) for entry in self.history]
        touches = [entry.get('is_touch', False) for entry in self.history]

        overlay = frame.copy()

        n = len(pts)
        for i in range(1, n):
            pt1 = pts[i-1]
            pt2 = pts[i]

            if pt1[0] is None or pt1[1] is None or pt2[0] is None or pt2[1] is None:
                continue

            age_frac = i / (n - 1)
            base_thickness = self.line_thickness
            speed = speeds[i]
            if speed >= self.speed_thickness_threshold_kmh:
                speed_factor = min(1.0, (speed - self.speed_thickness_threshold_kmh) /
                                   max(1.0, self.speed_max_thickness_kmh - self.speed_thickness_threshold_kmh))
                base_thickness += int(2.0 * speed_factor)

            if passes[i]:
                thickness = base_thickness + 2
                color = self.pass_color
            else:
                thickness = base_thickness
                color = colors[i]

            if self.fade_enabled:
                alpha = 1.0 - age_frac * self.fade_alpha
                alpha = max(0.1, alpha)
                cv2.line(overlay, pt1, pt2, color, thickness, cv2.LINE_AA)
                cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0, frame)
                overlay = frame.copy()
            else:
                cv2.line(frame, pt1, pt2, color, thickness, cv2.LINE_AA)

            if passes[i]:
                self._draw_arrow(frame, pt1, pt2, self.pass_color, thickness)

        for i, touch in enumerate(touches):
            if touch and pts[i][0] is not None:
                cv2.circle(frame, pts[i], 6, self.touch_color, 2, cv2.LINE_AA)

        if len(pts) > 0:
            last = pts[-1]
            if last[0] is not None:
                cv2.circle(frame, last, max(4, self.line_thickness), colors[-1], -1, cv2.LINE_AA)
                cv2.circle(frame, last, max(6, self.line_thickness + 2), colors[-1], 1, cv2.LINE_AA)

        return frame

    def _draw_arrow(self, frame: np.ndarray, pt1: Tuple[int, int], pt2: Tuple[int, int], color: Tuple[int, int, int], thickness: int) -> None:
        dx = pt2[0] - pt1[0]
        dy = pt2[1] - pt1[1]
        norm = np.hypot(dx, dy)
        if norm == 0:
            return
        ux, uy = dx / norm, dy / norm
        arrow = int(self.arrow_size)
        p1 = (int(pt2[0] - ux * arrow - uy * arrow / 2), int(pt2[1] - uy * arrow + ux * arrow / 2))
        p2 = (int(pt2[0] - ux * arrow + uy * arrow / 2), int(pt2[1] - uy * arrow - ux * arrow / 2))
        cv2.line(frame, p1, pt2, color, thickness, cv2.LINE_AA)
        cv2.line(frame, p2, pt2, color, thickness, cv2.LINE_AA)

    def draw_debug_overlay(self, frame: np.ndarray, config: Dict) -> np.ndarray:
        debug_cfg = config.get('visualization', {}).get('ball_path_renderer', {})
        if not debug_cfg.get('debug_overlay', False) or not self.history:
            return frame

        latest = self.history[-1]
        y0 = 30
        dy = 20
        lines = [
            f"Frame: {latest['frame_number']}",
            f"Team: {latest.get('team_id', 'None')}",
            f"Player: {latest.get('player_id', 'None')}",
            f"Confidence: {latest.get('possession_confidence', 0.0):.2f}",
            f"Speed: {latest.get('ball_speed_kmh', 0.0):.1f} km/h",
            f"History: {len(self.history)}/{self.history_length}"
        ]

        for i, line in enumerate(lines):
            y = y0 + i * dy
            cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        return frame

    def generate_debug_video(self, output_path: str, fps: float = 25.0) -> bool:
        if not self.history:
            return False

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, self.canvas_size)

        max_frame = max(entry['frame_number'] for entry in self.history)
        temp_renderer = BallPathRenderer({
            'visualization': {
                'ball_path_renderer': {
                    'enabled': self.enabled,
                    'history_length': self.history_length,
                    'line_thickness': self.line_thickness,
                    'fade_enabled': self.fade_enabled,
                    'fade_alpha': self.fade_alpha,
                    'arrow_size': self.arrow_size,
                    'min_possession_frames': self.min_possession_frames,
                    'speed_thickness_threshold_kmh': self.speed_thickness_threshold_kmh,
                    'speed_max_thickness_kmh': self.speed_max_thickness_kmh,
                    'team_colors': {k: list(v) for k, v in self.team_colors.items()},
                    'pass_color': list(self.pass_color),
                    'touch_color': list(self.touch_color),
                    'debug_overlay': True
                }
            },
            'pitch': {
                'canvas_width': self.canvas_size[0],
                'canvas_height': self.canvas_size[1],
                'length_m': self.pitch_size[0],
                'width_m': self.pitch_size[1]
            }
        })
        temp_renderer.reset()

        current_entries = []
        for entry in self.history:
            current_entries.append(entry)
            temp_renderer.history = deque(current_entries, maxlen=self.history_length)
            temp_renderer.current_color = entry['color']
            temp_renderer.prev_team_id = entry.get('team_id')
            temp_renderer.color_switch_counter = 0

            frame = np.zeros((self.canvas_size[1], self.canvas_size[0], 3), dtype=np.uint8)
            rendered = temp_renderer.render(frame)
            rendered = temp_renderer.draw_debug_overlay(rendered, {
                'visualization': {
                    'ball_path_renderer': {
                        'debug_overlay': True
                    }
                }
            })
            out.write(rendered)

        out.release()
        return True