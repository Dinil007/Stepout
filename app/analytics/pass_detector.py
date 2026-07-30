"""
Pass Detector Module

Detects football passing events by tracking ball trajectory and player possession transitions.
Classifies passes by length (Short, Medium, Long), direction (Forward, Back, Side),
and special tactical types (Through Ball).
"""

import logging
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from app.homography.field_config import FIELD_WIDTH_METERS

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)

# Minimum ball displacement distance (meters) to register a pass (not just tight foot control)
MIN_PASS_DISTANCE_M: float = 1.5

# Pass classification thresholds
CROSS_START_Y_DIST_M: float = 20.0  # Start within this distance of sideline
CROSS_END_Y_DIST_M: float = 15.0    # End within this distance of center
CROSS_MIN_DISTANCE_M: float = 10.0  # Minimum travel distance for cross

# Minimum ball speed (m/s) to be considered an active pass in flight
MIN_PASS_SPEED_MS: float = 2.0

# Minimum frames ball must be out of possession before resetting active pass state
PASS_RESET_FRAMES: int = 15


class PassEvent:
    """Represents a detected passing event."""

    def __init__(
        self,
        event_id: int,
        frame_start: int,
        passer_id: int,
        passer_team: str,
        start_position: Tuple[float, float]
    ):
        self.event_id: int = event_id
        self.frame_start: int = frame_start
        self.frame_end: Optional[int] = None
        self.passer: int = passer_id
        self.team: str = passer_team
        self.receiver: Optional[int] = None
        self.start_position: Tuple[float, float] = (round(start_position[0], 2), round(start_position[1], 2))
        self.end_position: Optional[Tuple[float, float]] = None
        self.distance_m: float = 0.0
        self.ball_travel_time_s: float = 0.0
        self.ball_speed_mps: float = 0.0
        self.successful: bool = False
        self.pass_type: str = "Short Pass"

    def complete(
        self,
        receiver_id: int,
        receiver_team: str,
        end_frame: int,
        end_position: Tuple[float, float],
        fps: float
    ) -> None:
        """Marks this pass sequence completed."""
        self.frame_end = end_frame
        self.receiver = receiver_id
        self.end_position = (round(end_position[0], 2), round(end_position[1], 2))

        # Trajectory metrics
        dx = self.end_position[0] - self.start_position[0]
        dy = self.end_position[1] - self.start_position[1]
        self.distance_m = round(float(np.hypot(dx, dy)), 2)

        dt_frames = max(1, self.frame_end - self.frame_start)
        self.ball_travel_time_s = round(dt_frames / fps, 2)
        self.ball_speed_mps = round(self.distance_m / max(self.ball_travel_time_s, 0.05), 2)

        # Success flag: same team pass completion
        self.successful = (receiver_team == self.team and receiver_id != self.passer)

        # Tactical classification
        self.pass_type = self._classify_pass(dx, self.distance_m, self.ball_speed_mps)

    def _classify_pass(self, dx: float, dist_m: float, speed_mps: float) -> str:
        """Classifies pass into tactical category."""
        # Cross detection: pass from flank toward goal area
        start_y = self.start_position[1]
        end_y = self.end_position[1]
        center_y = FIELD_WIDTH_METERS / 2.0
        start_dist_from_center = abs(start_y - center_y)
        end_dist_from_center = abs(end_y - center_y)

        if (start_dist_from_center > CROSS_START_Y_DIST_M and
            end_dist_from_center < CROSS_END_Y_DIST_M and
            dist_m >= CROSS_MIN_DISTANCE_M):
            return "Cross"

        # Through ball: long forward pass with high speed
        if dx > 15.0 and speed_mps > 10.0:
            return "Through Ball"

        # Direction classification
        if dx > 3.0:
            dir_str = "Forward Pass"
        elif dx < -3.0:
            dir_str = "Back Pass"
        else:
            dir_str = "Side Pass"

        # Distance classification
        if dist_m < 12.0:
            dist_str = "Short Pass"
        elif dist_m < 25.0:
            dist_str = "Medium Pass"
        else:
            dist_str = "Long Pass"

        return f"{dist_str} ({dir_str})" if dist_str != dir_str else dir_str

    def to_dict(self) -> Dict:
        return {
            "event_id": self.event_id,
            "frame_start": self.frame_start,
            "frame_end": self.frame_end,
            "team": self.team,
            "passer": self.passer,
            "receiver": self.receiver,
            "start_position": list(self.start_position) if self.start_position else None,
            "end_position": list(self.end_position) if self.end_position else None,
            "distance_m": self.distance_m,
            "ball_travel_time_s": self.ball_travel_time_s,
            "ball_speed_mps": self.ball_speed_mps,
            "successful": self.successful,
            "pass_type": self.pass_type
        }


class PassDetector:
    """
    Detects pass events by analyzing player-ball possession transitions and
    ball travel distance in 2D pitch space.
    """

    def __init__(
        self,
        fps: float,
        min_pass_distance_m: float = MIN_PASS_DISTANCE_M,
        reset_frames: int = PASS_RESET_FRAMES
    ):
        if fps <= 0:
            raise ValueError(f"FPS must be positive. Got: {fps}")

        self.fps = fps
        self.min_pass_distance_m = min_pass_distance_m
        self.reset_frames = reset_frames

        # State
        self._last_possessor_id: Optional[int] = None
        self._last_possessor_pos: Optional[Tuple[float, float]] = None
        self._last_possessor_frame: Optional[int] = None

        self._active_pass: Optional[PassEvent] = None
        self._event_counter: int = 0
        self._pass_events: List[PassEvent] = []

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
        Processes one frame to detect pass events.

        Args:
            frame_number: Current frame index.
            ball_position_m: Ball (x, y) in real-world meters.
            player_positions_m: Dict of track_id -> (x, y) in meters.
            possessor_id: Track ID of player currently in possession.
            team_assignments: Dict of track_id -> team_id.

        Returns:
            Dict describing a completed event, or None.
        """
        curr_team = self._get_team_name(team_assignments.get(possessor_id) if (possessor_id and team_assignments) else None)

        # 1. Detect Pass Launch: ball leaves a player who had possession
        if possessor_id is None and self._last_possessor_id is not None and self._active_pass is None:
            passer_team = self._get_team_name(team_assignments.get(self._last_possessor_id) if team_assignments else None)
            passer_pos = self._last_possessor_pos or (0.0, 0.0)

            self._event_counter += 1
            self._active_pass = PassEvent(
                event_id=self._event_counter,
                frame_start=self._last_possessor_frame or frame_number,
                passer_id=self._last_possessor_id,
                passer_team=passer_team,
                start_position=passer_pos
            )
            logger.debug("Pass launched by Player #%d @ frame %d", self._last_possessor_id, frame_number)

        # 2. Detect Pass Reception: ball arrives at a player
        elif possessor_id is not None and self._active_pass is not None:
            # Different player or same player after movement
            rec_pos = player_positions_m.get(possessor_id, (0.0, 0.0))
            dist = float(np.hypot(rec_pos[0] - self._active_pass.start_position[0],
                                  rec_pos[1] - self._active_pass.start_position[1]))

            if possessor_id != self._active_pass.passer and dist >= self.min_pass_distance_m:
                self._active_pass.complete(
                    receiver_id=possessor_id,
                    receiver_team=curr_team,
                    end_frame=frame_number,
                    end_position=rec_pos,
                    fps=self.fps
                )
                self._pass_events.append(self._active_pass)
                res_dict = self._active_pass.to_dict()
                logger.info("Pass completed: Passer #%d -> Receiver #%d (%.1fm, %s)",
                            self._active_pass.passer, possessor_id, res_dict["distance_m"], res_dict["pass_type"])
                self._active_pass = None
                self._last_possessor_id = possessor_id
                self._last_possessor_pos = rec_pos
                self._last_possessor_frame = frame_number
                return res_dict

            elif possessor_id == self._active_pass.passer:
                # Same player reclaimed ball (dribble / touch) -> cancel pass launch
                self._active_pass = None
            else:
                # Different player but too close to start: likely interception/deflection
                # Cancel active pass to prevent erroneous completion later
                logger.debug("Pass canceled: Player #%d intercepted near start position", possessor_id)
                self._active_pass = None

        # Update last known possessor state
        if possessor_id is not None:
            self._last_possessor_id = possessor_id
            self._last_possessor_pos = player_positions_m.get(possessor_id)
            self._last_possessor_frame = frame_number

        return None

    def get_pass_events(self) -> List[Dict]:
        """Returns all completed pass events as serialized dicts."""
        return [p.to_dict() for p in self._pass_events]

    def get_summary(self) -> Dict:
        """Returns aggregate player and team pass statistics summary."""
        total = len(self._pass_events)
        completed = sum(1 for p in self._pass_events if p.successful)
        unsuccessful = total - completed

        # Team stats
        team_stats = {}
        for p in self._pass_events:
            t = p.team
            if t not in team_stats:
                team_stats[t] = {"total_passes": 0, "completed_passes": 0, "total_distance_m": 0.0}
            team_stats[t]["total_passes"] += 1
            if p.successful:
                team_stats[t]["completed_passes"] += 1
            team_stats[t]["total_distance_m"] += p.distance_m

        for t, s in team_stats.items():
            tot = s["total_passes"]
            comp = s["completed_passes"]
            s["accuracy_pct"] = round((comp / tot) * 100.0, 1) if tot > 0 else 0.0
            s["avg_distance_m"] = round(s["total_distance_m"] / max(tot, 1), 2)

        # Player stats
        player_stats = {}
        for p in self._pass_events:
            pid = p.passer
            if pid not in player_stats:
                player_stats[pid] = {"attempted": 0, "completed": 0, "total_distance_m": 0.0}
            player_stats[pid]["attempted"] += 1
            if p.successful:
                player_stats[pid]["completed"] += 1
            player_stats[pid]["total_distance_m"] += p.distance_m

        for pid, ps in player_stats.items():
            att = ps["attempted"]
            cmp = ps["completed"]
            ps["accuracy_pct"] = round((cmp / att) * 100.0, 1) if att > 0 else 0.0
            ps["avg_distance_m"] = round(ps["total_distance_m"] / max(att, 1), 2)

        return {
            "total_passes": total,
            "completed_passes": completed,
            "unsuccessful_passes": unsuccessful,
            "overall_accuracy_pct": round((completed / total) * 100.0, 1) if total > 0 else 0.0,
            "team_pass_summary": team_stats,
            "player_pass_summary": player_stats
        }

    def reset(self) -> None:
        """Resets all detector state."""
        self._last_possessor_id = None
        self._last_possessor_pos = None
        self._last_possessor_frame = None
        self._active_pass = None
        self._event_counter = 0
        self._pass_events.clear()

