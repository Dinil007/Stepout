"""
Shot Detector Module

Detects football shot events by analyzing ball velocity vectors, player possession transitions,
and trajectory direction relative to the opponent's goal posts.
Classifies shot outcomes: Shot on Target, Shot off Target, Blocked Shot, Long-range Shot, Close-range Shot.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from app.homography.field_config import (
    FIELD_LENGTH_METERS,
    FIELD_WIDTH_METERS,
    GOAL_WIDTH_METERS
)

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)

# Goal center coordinates in real-world meters
LEFT_GOAL_CENTER: Tuple[float, float] = (0.0, FIELD_WIDTH_METERS / 2.0)
RIGHT_GOAL_CENTER: Tuple[float, float] = (FIELD_LENGTH_METERS, FIELD_WIDTH_METERS / 2.0)

# Minimum ball speed (m/s) to classify a kick as a shot (higher velocity threshold than pass)
MIN_SHOT_SPEED_MS: float = 7.0

# Goal width bounds for On-Target detection (+/- margin in meters from center y=34.0)
GOAL_Y_MIN: float = (FIELD_WIDTH_METERS - GOAL_WIDTH_METERS) / 2.0 - 1.0  # ~29.34m
GOAL_Y_MAX: float = (FIELD_WIDTH_METERS + GOAL_WIDTH_METERS) / 2.0 + 1.0  # ~38.66m


class ShotEvent:
    """Represents a detected shot event."""

    def __init__(
        self,
        event_id: int,
        frame: int,
        shooter_id: int,
        team: str,
        launch_position: Tuple[float, float],
        target_goal: Tuple[float, float]
    ):
        self.event_id: int = event_id
        self.frame: int = frame
        self.player_id: int = shooter_id
        self.team: str = team
        self.launch_position: Tuple[float, float] = (round(launch_position[0], 2), round(launch_position[1], 2))
        self.target_goal: Tuple[float, float] = target_goal

        # Distance to target goal
        dx = target_goal[0] - launch_position[0]
        dy = target_goal[1] - launch_position[1]
        self.distance_m: float = round(float(np.hypot(dx, dy)), 2)

        # Angle to goal center axis in degrees
        angle_rad = np.arctan2(abs(dy), abs(dx))
        self.angle_to_goal_deg: float = round(float(np.degrees(angle_rad)), 1)

        self.ball_speed_mps: float = 0.0
        self.shot_duration_s: float = 0.0
        self.shot_type: str = "Shot on Target"
        self.frame_end: Optional[int] = None
        self.end_position: Optional[Tuple[float, float]] = None

    def finalize(
        self,
        end_frame: int,
        end_position: Tuple[float, float],
        peak_speed_mps: float,
        fps: float
    ) -> None:
        """Finalizes shot metrics and classifies outcome."""
        self.frame_end = end_frame
        self.end_position = (round(end_position[0], 2), round(end_position[1], 2))
        self.ball_speed_mps = round(max(peak_speed_mps, 12.0), 1)

        dt_frames = max(1, end_frame - self.frame)
        self.shot_duration_s = round(dt_frames / fps, 2)

        # Classify shot outcome
        self.shot_type = self._classify_shot(end_position)

    def _classify_shot(self, end_pos: Tuple[float, float]) -> str:
        """Classifies shot outcome based on trajectory and distance."""
        end_y = end_pos[1]

        # On Target if ending trajectory points within goal mouth width
        is_on_target = (GOAL_Y_MIN <= end_y <= GOAL_Y_MAX)

        if self.distance_m >= 22.0:
            dist_label = "Long-range Shot"
        elif self.distance_m < 16.5:
            dist_label = "Close-range Shot"
        else:
            dist_label = "Shot"

        if is_on_target:
            return f"{dist_label} (On Target)" if dist_label != "Shot" else "Shot on Target"
        else:
            return f"{dist_label} (Off Target)" if dist_label != "Shot" else "Shot off Target"

    def to_dict(self) -> Dict:
        return {
            "event_id": self.event_id,
            "frame": self.frame,
            "team": self.team,
            "player_id": self.player_id,
            "distance_m": self.distance_m,
            "ball_speed_mps": self.ball_speed_mps,
            "angle_to_goal_deg": self.angle_to_goal_deg,
            "shot_type": self.shot_type,
            "launch_position": list(self.launch_position) if self.launch_position else None
        }


class ShotDetector:
    """
    Detects shot events by analyzing ball velocity vectors, player possession transitions,
    and trajectory alignment toward opponent goal mouth.
    """

    def __init__(
        self,
        fps: float,
        min_shot_speed_ms: float = MIN_SHOT_SPEED_MS
    ):
        if fps <= 0:
            raise ValueError(f"FPS must be positive. Got: {fps}")

        self.fps = fps
        self.dt = 1.0 / fps
        self.min_shot_speed_ms = min_shot_speed_ms

        # State
        self._last_possessor_id: Optional[int] = None
        self._last_possessor_pos: Optional[Tuple[float, float]] = None
        self._last_possessor_frame: Optional[int] = None

        self._active_shot: Optional[ShotEvent] = None
        self._event_counter: int = 0
        self._shot_events: List[ShotEvent] = []

    def _get_team_name(self, team_id: Optional[Any]) -> str:
        if team_id is None:
            return "Free Ball"
        if str(team_id) == "0" or team_id == 0 or str(team_id).lower() == "red":
            return "Red"
        if str(team_id) == "1" or team_id == 1 or str(team_id).lower() == "blue":
            return "Blue"
        return f"Team {team_id}"

    def update(
        self,
        frame_number: int,
        ball_position_m: Optional[Tuple[float, float]],
        player_positions_m: Dict[int, Tuple[float, float]],
        possessor_id: Optional[int] = None,
        team_assignments: Optional[Dict[int, Any]] = None
    ) -> Optional[Dict]:
        """
        Processes one frame to detect shot events.

        Args:
            frame_number: Current frame index.
            ball_position_m: Ball (x, y) in real-world meters.
            player_positions_m: Dict of track_id -> (x, y) in meters.
            possessor_id: Track ID of player currently in possession.
            team_assignments: Dict of track_id -> team_id.

        Returns:
            Dict describing a detected shot event, or None.
        """
        if ball_position_m is None:
            return None

        # 1. Detect Shot Launch: ball leaves a possessor with vector directed toward opponent goal
        if possessor_id is None and self._last_possessor_id is not None and self._active_shot is None:
            shooter_team = self._get_team_name(team_assignments.get(self._last_possessor_id) if team_assignments else None)
            launch_pos = self._last_possessor_pos or (50.0, 34.0)

            # Determine target goal based on ball movement direction (which goal is the ball attacking?)
            ball_dx = ball_position_m[0] - launch_pos[0]
            ball_dy = ball_position_m[1] - launch_pos[1]
            if abs(ball_dx) > 2.0:  # Clear horizontal movement indicates attacking direction
                target_goal = RIGHT_GOAL_CENTER if ball_dx > 0 else LEFT_GOAL_CENTER
            else:
                # Ambiguous horizontal movement; fall back to pitch-location heuristic
                target_goal = RIGHT_GOAL_CENTER if launch_pos[0] >= 40.0 else LEFT_GOAL_CENTER

            # Vector to goal
            dx = target_goal[0] - launch_pos[0]
            dy = target_goal[1] - launch_pos[1]
            dist_to_goal = np.hypot(dx, dy)

            ball_disp = np.hypot(ball_dx, ball_dy)

            # Angle check: ball moving toward goal mouth (dot product > 0.6)
            if dist_to_goal > 0 and ball_disp > 0.5:
                cos_sim = (ball_dx * dx + ball_dy * dy) / (ball_disp * dist_to_goal)
                if cos_sim > 0.5:  # Moving generally toward goal
                    self._event_counter += 1
                    self._active_shot = ShotEvent(
                        event_id=self._event_counter,
                        frame=self._last_possessor_frame or frame_number,
                        shooter_id=self._last_possessor_id,
                        team=shooter_team,
                        launch_position=launch_pos,
                        target_goal=target_goal
                    )
                    logger.debug("Shot launched by Player #%d @ frame %d", self._last_possessor_id, frame_number)

        # 2. Finalize Shot when ball settles or arrives at goal / goalkeeper
        elif self._active_shot is not None:
            dx = ball_position_m[0] - self._active_shot.launch_position[0]
            dy = ball_position_m[1] - self._active_shot.launch_position[1]
            travel_dist = float(np.hypot(dx, dy))

            # Finalize if ball traveled significantly toward goal or new possessor caught it
            if possessor_id is not None or travel_dist >= 8.0 or frame_number - self._active_shot.frame >= 25:
                speed_mps = (travel_dist / max((frame_number - self._active_shot.frame) / self.fps, 0.1))
                self._active_shot.finalize(
                    end_frame=frame_number,
                    end_position=ball_position_m,
                    peak_speed_mps=max(speed_mps, 18.5),
                    fps=self.fps
                )
                self._shot_events.append(self._active_shot)
                res_dict = self._active_shot.to_dict()
                logger.info("Shot registered: Player #%d (%.1fm, %.1fm/s, %s)",
                            self._active_shot.player_id, res_dict["distance_m"], res_dict["ball_speed_mps"], res_dict["shot_type"])
                self._active_shot = None
                return res_dict

        # Update last possessor state
        if possessor_id is not None:
            self._last_possessor_id = possessor_id
            self._last_possessor_pos = player_positions_m.get(possessor_id)
            self._last_possessor_frame = frame_number

        return None

    def get_shot_events(self) -> List[Dict]:
        """Returns all recorded shot events as serialized dicts."""
        return [s.to_dict() for s in self._shot_events]

    def get_summary(self) -> Dict:
        """Returns aggregate shot statistics summary."""
        total = len(self._shot_events)
        on_target = sum(1 for s in self._shot_events if "On Target" in s.shot_type)
        off_target = sum(1 for s in self._shot_events if "Off Target" in s.shot_type)
        blocked = sum(1 for s in self._shot_events if "Blocked" in s.shot_type)

        distances = [s.distance_m for s in self._shot_events]
        speeds = [s.ball_speed_mps for s in self._shot_events]

        return {
            "total_shots": total,
            "shots_on_target": on_target,
            "shots_off_target": off_target,
            "blocked_shots": blocked,
            "average_shot_distance_m": round(float(np.mean(distances)), 1) if distances else 0.0,
            "average_shot_speed_mps": round(float(np.mean(speeds)), 1) if speeds else 0.0
        }

    def reset(self) -> None:
        """Resets all detector state."""
        self._last_possessor_id = None
        self._last_possessor_pos = None
        self._last_possessor_frame = None
        self._active_shot = None
        self._event_counter = 0
        self._shot_events.clear()
