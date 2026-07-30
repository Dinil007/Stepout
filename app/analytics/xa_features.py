"""Feature engineering for Expected Assists (xA) calculations."""
from __future__ import annotations

import logging
import math
from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

try:
    from app.homography.field_config import FIELD_LENGTH_METERS, FIELD_WIDTH_METERS, GOAL_WIDTH_METERS
except Exception:
    FIELD_LENGTH_METERS = 105.0
    FIELD_WIDTH_METERS = 68.0
    GOAL_WIDTH_METERS = 7.32

LOGGER = logging.getLogger(__name__)

JsonDict = Dict[str, Any]


@dataclass(frozen=True)
class XAFeatures:
    """Single-assist candidate feature vector."""

    pass_id: int
    shot_id: int
    passer_id: Optional[int]
    receiver_id: Optional[int]
    team: str
    pass_start_x: float
    pass_start_y: float
    pass_end_x: float
    pass_end_y: float
    pass_length_m: float
    forward_distance_m: float
    pass_angle_deg: float
    receiver_distance_to_goal_m: float
    receiver_angle_to_goal_deg: float
    time_before_shot_s: float
    frame_difference: int
    touches_before_shot: int
    defensive_pressure: float
    defenders_near_receiver: int
    attackers_in_final_third: int
    ball_progression_m: float
    pass_type: str
    shot_xg: float
    shot_distance_m: float
    shot_angle_deg: float
    shot_type: str

    def to_dict(self) -> JsonDict:
        return asdict(self)


class XAFeatureExtractor:
    """Extracts production-ready xA features from linked pass-to-shot events."""

    FEATURE_COLUMNS = [
        "pass_length_m",
        "forward_distance_m",
        "pass_angle_deg",
        "receiver_distance_to_goal_m",
        "receiver_angle_to_goal_deg",
        "time_before_shot_s",
        "frame_difference",
        "touches_before_shot",
        "defensive_pressure",
        "defenders_near_receiver",
        "attackers_in_final_third",
        "ball_progression_m",
        "shot_xg",
        "shot_distance_m",
        "shot_angle_deg",
    ]

    def __init__(
        self,
        pitch_length_m: float = FIELD_LENGTH_METERS,
        pitch_width_m: float = FIELD_WIDTH_METERS,
        goal_width_m: float = GOAL_WIDTH_METERS,
    ) -> None:
        self.pitch_length_m = pitch_length_m
        self.pitch_width_m = pitch_width_m
        self.goal_width_m = goal_width_m

    def extract(
        self,
        pass_event: JsonDict,
        shot_event: JsonDict,
        average_positions: Optional[JsonDict] = None,
        fps: Optional[float] = None,
    ) -> XAFeatures:
        pass_id = int(pass_event.get("event_id") or pass_event.get("pass_id") or 0)
        shot_id = int(shot_event.get("event_id") or shot_event.get("shot_id") or 0)
        passer_id = pass_event.get("passer")
        receiver_id = pass_event.get("receiver")
        team = str(pass_event.get("team") or "Unknown")

        start_pos = self._position(pass_event, "start_position")
        end_pos = self._position(pass_event, "end_position")
        shot_pos = self._position(shot_event, "launch_position")

        pass_length = float(pass_event.get("distance_m") or self._distance(start_pos, end_pos))
        forward_dist = end_pos[0] - start_pos[0]
        pass_angle = self._pass_angle(start_pos, end_pos)

        target_goal_x = self.pitch_length_m if shot_pos[0] <= self.pitch_length_m / 2 else 0.0
        target_goal = (target_goal_x, self.pitch_width_m / 2.0)
        receiver_dist_to_goal = self._distance(end_pos, target_goal)
        receiver_angle_to_goal = self._goal_angle(end_pos, target_goal_x)

        pass_frame = int(pass_event.get("frame_end") or pass_event.get("frame_start") or 0)
        shot_frame = int(shot_event.get("frame") or shot_event.get("frame_start") or 0)
        frame_diff = max(0, shot_frame - pass_frame)
        time_before_shot = round(frame_diff / fps, 2) if fps else 0.0

        touches = self._estimate_touches(frame_diff, fps or 30.0)
        pressure = self._defensive_pressure(end_pos, average_positions or {}, team)
        defenders = self._count_defenders_near(end_pos, average_positions or {}, team, radius=5.0)
        attackers_final_third = self._count_attackers_final_third(average_positions or {}, team)

        ball_prog = max(0.0, forward_dist)
        shot_xg = float(shot_event.get("xg") or shot_event.get("distance_m", 0) * 0.02)
        shot_dist = float(shot_event.get("distance_m") or self._distance(shot_pos, target_goal))
        shot_angle = float(
            shot_event.get("angle_to_goal_deg")
            or shot_event.get("angle_deg")
            or self._goal_angle(shot_pos, target_goal_x)
        )
        shot_type = str(shot_event.get("shot_type") or "unknown")

        return XAFeatures(
            pass_id=pass_id,
            shot_id=shot_id,
            passer_id=passer_id,
            receiver_id=receiver_id,
            team=team,
            pass_start_x=round(start_pos[0], 2),
            pass_start_y=round(start_pos[1], 2),
            pass_end_x=round(end_pos[0], 2),
            pass_end_y=round(end_pos[1], 2),
            pass_length_m=round(pass_length, 2),
            forward_distance_m=round(forward_dist, 2),
            pass_angle_deg=round(pass_angle, 2),
            receiver_distance_to_goal_m=round(receiver_dist_to_goal, 2),
            receiver_angle_to_goal_deg=round(receiver_angle_to_goal, 2),
            time_before_shot_s=time_before_shot,
            frame_difference=frame_diff,
            touches_before_shot=touches,
            defensive_pressure=pressure,
            defenders_near_receiver=defenders,
            attackers_in_final_third=attackers_final_third,
            ball_progression_m=round(ball_prog, 2),
            pass_type=str(pass_event.get("pass_type") or "unknown"),
            shot_xg=round(shot_xg, 3),
            shot_distance_m=round(shot_dist, 2),
            shot_angle_deg=round(shot_angle, 2),
            shot_type=shot_type,
        )

    def to_model_vector(self, features: XAFeatures) -> List[float]:
        payload = features.to_dict()
        return [float(payload.get(column) or 0.0) for column in self.FEATURE_COLUMNS]

    def _position(self, event: JsonDict, key: str) -> Tuple[float, float]:
        raw = event.get(key) or [self.pitch_length_m / 2.0, self.pitch_width_m / 2.0]
        if len(raw) < 2:
            return self.pitch_length_m / 2.0, self.pitch_width_m / 2.0
        return float(raw[0]), float(raw[1])

    def _distance(self, a: Tuple[float, float], b: Tuple[float, float]) -> float:
        return math.hypot(b[0] - a[0], b[1] - a[1])

    def _pass_angle(self, start: Tuple[float, float], end: Tuple[float, float]) -> float:
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        if abs(dx) < 1e-6:
            return 90.0 if dy > 0 else -90.0
        return math.degrees(math.atan2(dy, dx))

    def _goal_angle(self, source: Tuple[float, float], goal_x: float) -> float:
        left_post = (goal_x, (self.pitch_width_m - self.goal_width_m) / 2.0)
        right_post = (goal_x, (self.pitch_width_m + self.goal_width_m) / 2.0)
        a = self._distance(source, left_post)
        b = self._distance(source, right_post)
        c = self.goal_width_m
        denominator = max(2 * a * b, 1e-9)
        cosine = max(-1.0, min(1.0, (a * a + b * b - c * c) / denominator))
        return math.degrees(math.acos(cosine))

    def _estimate_touches(self, frame_diff: int, fps: float) -> int:
        time_s = frame_diff / max(fps, 1.0)
        if time_s < 0.5:
            return 0
        if time_s < 1.0:
            return 1
        if time_s < 2.0:
            return 2
        return min(int(time_s * 1.5), 5)

    def _defensive_pressure(
        self, position: Tuple[float, float], avg_positions: JsonDict, team: str, radius: float = 8.0
    ) -> float:
        defenders = 0
        for player in avg_positions.values():
            pos = player.get("average_position") if isinstance(player, dict) else None
            if not pos or len(pos) < 2:
                continue
            if str(player.get("team")) in {team, "Unknown", "Free Ball", ""}:
                continue
            dist = self._distance(position, (float(pos[0]), float(pos[1])))
            if dist <= radius:
                defenders += 1
        return round(min(defenders / 5.0, 1.0), 3)

    def _count_defenders_near(
        self, position: Tuple[float, float], avg_positions: JsonDict, team: str, radius: float = 5.0
    ) -> int:
        count = 0
        for player in avg_positions.values():
            pos = player.get("average_position") if isinstance(player, dict) else None
            if not pos or len(pos) < 2:
                continue
            if str(player.get("team")) in {team, "Unknown", "Free Ball", ""}:
                continue
            dist = self._distance(position, (float(pos[0]), float(pos[1])))
            if dist <= radius:
                count += 1
        return count

    def _count_attackers_final_third(self, avg_positions: JsonDict, team: str) -> int:
        count = 0
        for player in avg_positions.values():
            pos = player.get("average_position") if isinstance(player, dict) else None
            if not pos or len(pos) < 2:
                continue
            if str(player.get("team")) != team:
                continue
            if float(pos[0]) >= self.pitch_length_m * 0.67:
                count += 1
        return count