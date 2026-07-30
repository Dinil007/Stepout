"""
Ball Possession Analyzer Module

Determines ball possession by computing proximity between the ball's
real-world 2D pitch position and each player's feet position.
Tracks possession duration per player and per team across frames.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)

# Possession proximity threshold (meters): player within this distance controls the ball
DEFAULT_POSSESSION_RADIUS_M: float = 2.5

# Minimum consecutive frames to confirm possession
POSSESSION_CONFIRMATION_FRAMES: int = 3


class BallPossessionAnalyzer:
    """
    Determines which player and team has ball possession based on proximity
    to the ball position in real-world pitch coordinates.
    """

    def __init__(
        self,
        possession_radius_m: float = DEFAULT_POSSESSION_RADIUS_M,
        confirmation_frames: int = POSSESSION_CONFIRMATION_FRAMES,
        fps: float = 30.0
    ):
        """
        Initializes the BallPossessionAnalyzer.

        Args:
            possession_radius_m: Maximum proximity distance to claim possession (meters).
            confirmation_frames: Minimum consecutive frames in proximity before confirming possession.
            fps: Video frames per second for duration calculation.
        """
        self.possession_radius_m = possession_radius_m
        self.confirmation_frames = confirmation_frames
        self.fps = max(fps, 1.0)

        # Frame counters & metrics
        self._possession_frames: Dict[int, int] = {}          # track_id -> frames in possession
        self._team_possession_frames: Dict[Any, int] = {}     # team_id -> total frames in possession
        self._contested_frames: int = 0
        self._no_possession_frames: int = 0
        self._total_frames: int = 0

        # Current state & spell duration
        self._current_possessor: Optional[int] = None         # track_id
        self._current_team: Optional[Any] = None              # team_id / label
        self._possession_start_frame: Optional[int] = None

        # Candidate streak tracking (anti-flicker filter)
        self._candidate_id: Optional[int] = None
        self._candidate_streak: int = 0

        # History log for JSON exports
        self.history_log: List[Dict] = []

    @staticmethod
    def _euclidean_distance(a: Tuple[float, float], b: Tuple[float, float]) -> float:
        """Returns Euclidean distance between two 2D points."""
        return float(np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2))

    def _get_team_name(self, team_id: Optional[Any]) -> str:
        """Maps internal team_id to human readable team string."""
        if team_id is None:
            return "Free Ball"
        if str(team_id) == "0" or team_id == 0 or str(team_id).lower() == "red":
            return "Red"
        if str(team_id) == "1" or team_id == 1 or str(team_id).lower() == "blue":
            return "Blue"
        return f"Team {team_id}"

    def update(
        self,
        ball_position_m: Optional[Tuple[float, float]],
        player_positions_m: Dict[int, Tuple[float, float]],
        team_assignments: Optional[Dict[int, Any]] = None,
        frame_number: Optional[int] = None,
        ball_image_position: Optional[Tuple[int, int]] = None
    ) -> Dict:
        """
        Determines which player is in possession for the current frame.

        Args:
            ball_position_m: Ball (x, y) in real-world meters. None if undetected.
            player_positions_m: Dict of track_id -> (x, y) position in meters.
            team_assignments: Optional dict of track_id -> team_id.
            frame_number: Current frame index.
            ball_image_position: Optional (cx, cy) pixel coordinate of ball in frame.

        Returns:
            Dict with possession state for this frame.
        """
        self._total_frames += 1
        curr_frame = frame_number if frame_number is not None else self._total_frames

        if ball_position_m is None or not player_positions_m:
            self._no_possession_frames += 1
            self._candidate_id = None
            self._candidate_streak = 0
            if self._current_possessor is not None:
                self._current_possessor = None
                self._current_team = None
                self._possession_start_frame = None

            state_dict = {
                "frame": curr_frame,
                "possessor_id": None,
                "possessor_team": None,
                "team_name": "Free Ball",
                "duration_seconds": 0.0,
                "state": "Free Ball",
                "ball_position": list(ball_image_position) if ball_image_position else None
            }
            self.history_log.append(state_dict)
            return state_dict

        # Find the closest player to the ball in 2D pitch meters
        distances = {
            tid: self._euclidean_distance(ball_position_m, pos)
            for tid, pos in player_positions_m.items()
        }

        closest_id = min(distances, key=distances.get)
        closest_dist = distances[closest_id]

        if closest_dist > self.possession_radius_m:
            self._no_possession_frames += 1
            self._candidate_id = None
            self._candidate_streak = 0
            if self._current_possessor is not None:
                self._current_possessor = None
                self._current_team = None
                self._possession_start_frame = None

            state_dict = {
                "frame": curr_frame,
                "possessor_id": None,
                "possessor_team": None,
                "team_name": "Free Ball",
                "duration_seconds": 0.0,
                "state": "Free Ball",
                "ball_position": list(ball_image_position) if ball_image_position else None
            }
            self.history_log.append(state_dict)
            return state_dict

        # Candidate streak logic to suppress flickering
        if self._candidate_id == closest_id:
            self._candidate_streak += 1
        else:
            self._candidate_id = closest_id
            self._candidate_streak = 1

        # Only confirm possession after consecutive frames threshold
        if self._candidate_streak >= self.confirmation_frames:
            if self._current_possessor != closest_id:
                self._current_possessor = closest_id
                self._possession_start_frame = curr_frame

            self._possession_frames[closest_id] = self._possession_frames.get(closest_id, 0) + 1

            team_id = team_assignments.get(closest_id) if team_assignments else None
            team_name = self._get_team_name(team_id)
            self._current_team = team_name

            if team_name != "Free Ball":
                self._team_possession_frames[team_name] = self._team_possession_frames.get(team_name, 0) + 1

            spell_frames = (curr_frame - self._possession_start_frame + 1) if self._possession_start_frame else 1
            duration_sec = round(spell_frames / self.fps, 2)

            state_dict = {
                "frame": curr_frame,
                "player_id": closest_id,
                "possessor_id": closest_id,
                "possessor_team": team_id,
                "team": team_name,
                "team_name": team_name,
                "duration": duration_sec,
                "duration_seconds": duration_sec,
                "state": "In Possession",
                "distance_to_ball_m": round(closest_dist, 2),
                "ball_position": list(ball_image_position) if ball_image_position else None
            }
            self.history_log.append(state_dict)
            return state_dict

        current_team_name = self._get_team_name(self._current_team) if self._current_possessor else "Free Ball"
        spell_frames = (curr_frame - self._possession_start_frame + 1) if self._possession_start_frame else 1
        duration_sec = round(spell_frames / self.fps, 2) if self._current_possessor else 0.0

        state_dict = {
            "frame": curr_frame,
            "player_id": self._current_possessor,
            "possessor_id": self._current_possessor,
            "possessor_team": self._current_team,
            "team": current_team_name,
            "team_name": current_team_name,
            "duration": duration_sec,
            "duration_seconds": duration_sec,
            "state": "Contested / Transitioning",
            "ball_position": list(ball_image_position) if ball_image_position else None
        }
        self.history_log.append(state_dict)
        return state_dict

    def get_possession_percentage(self) -> Dict:
        """
        Returns team-level possession split as percentages.

        Returns:
            Dict with team possession percentages.
        """
        effective_frames = max(self._total_frames, 1)
        result = {}

        for team_name, frames in self._team_possession_frames.items():
            result[f"{team_name}_pct"] = round((frames / effective_frames) * 100.0, 1)

        result["Free_Ball_pct"] = round((self._no_possession_frames / effective_frames) * 100.0, 1)
        return result

    def get_team_possession_summary(self) -> Dict:
        """
        Export dictionary for outputs/team_possession_summary.json
        """
        effective_frames = max(self._total_frames, 1)
        red_frames = self._team_possession_frames.get("Red", 0)
        blue_frames = self._team_possession_frames.get("Blue", 0)
        free_frames = self._no_possession_frames

        return {
            "total_frames": self._total_frames,
            "total_duration_seconds": round(self._total_frames / self.fps, 2),
            "team_possession_pct": {
                "Red": round((red_frames / effective_frames) * 100.0, 1),
                "Blue": round((blue_frames / effective_frames) * 100.0, 1),
                "Free Ball": round((free_frames / effective_frames) * 100.0, 1)
            },
            "total_possession_time_seconds": {
                "Red": round(red_frames / self.fps, 2),
                "Blue": round(blue_frames / self.fps, 2),
                "Free Ball": round(free_frames / self.fps, 2)
            }
        }

    def get_player_possession_frames(self, track_id: int) -> int:
        """Returns total frames a player had possession."""
        return self._possession_frames.get(track_id, 0)

    def get_current_possessor(self) -> Optional[int]:
        """Returns the current confirmed possessing player track ID."""
        return self._current_possessor

    def get_summary(self) -> Dict:
        """Returns full possession analytics summary."""
        return {
            "total_frames": self._total_frames,
            "no_possession_frames": self._no_possession_frames,
            "team_possession": self.get_possession_percentage(),
            "team_summary": self.get_team_possession_summary(),
            "player_possession_frames": dict(self._possession_frames)
        }

    def reset(self) -> None:
        """Resets all possession state."""
        self._possession_frames.clear()
        self._team_possession_frames.clear()
        self._contested_frames = 0
        self._no_possession_frames = 0
        self._total_frames = 0
        self._current_possessor = None
        self._current_team = None
        self._possession_start_frame = None
        self._candidate_id = None
        self._candidate_streak = 0
        self.history_log.clear()
