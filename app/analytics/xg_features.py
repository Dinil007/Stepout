"""Feature engineering for expected goals calculations."""

from __future__ import annotations

import logging
import math
from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

try:
    from app.homography.field_config import FIELD_LENGTH_METERS, FIELD_WIDTH_METERS, GOAL_WIDTH_METERS
except Exception:  # pragma: no cover - import fallback for isolated tooling
    FIELD_LENGTH_METERS = 105.0
    FIELD_WIDTH_METERS = 68.0
    GOAL_WIDTH_METERS = 7.32

LOGGER = logging.getLogger(__name__)


JsonDict = Dict[str, Any]


@dataclass(frozen=True)
class XGFeatures:
    """Single-shot xG feature vector."""

    shot_id: int
    player_id: Optional[int]
    team: str
    frame: int
    match_time_s: Optional[float]
    shot_x: float
    shot_y: float
    distance_m: float
    angle_deg: float
    distance_from_centre_m: float
    goal_mouth_visibility: float
    ball_speed_mps: float
    shooter_speed_mps: Optional[float]
    defenders_nearby: int
    attackers_nearby: int
    pressure_score: float
    goalkeeper_distance_m: Optional[float]
    pass_sequence_length: int
    possession_duration_s: Optional[float]
    open_play: bool
    set_piece: bool
    counter_attack: bool
    body_part: str
    shot_type: str

    def to_dict(self) -> JsonDict:
        return asdict(self)


class XGFeatureExtractor:
    """Extracts production-ready xG features from structured shot events."""

    FEATURE_COLUMNS = [
        "distance_m",
        "angle_deg",
        "distance_from_centre_m",
        "goal_mouth_visibility",
        "ball_speed_mps",
        "defenders_nearby",
        "attackers_nearby",
        "pressure_score",
        "pass_sequence_length",
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
        shot: JsonDict,
        fps: Optional[float] = None,
        pass_events: Optional[Iterable[JsonDict]] = None,
        average_positions: Optional[JsonDict] = None,
    ) -> XGFeatures:
        shot_id = int(shot.get("event_id") or shot.get("shot_id") or shot.get("id") or 0)
        frame = int(shot.get("frame") or shot.get("frame_start") or 0)
        player_id = shot.get("player_id") if shot.get("player_id") is not None else shot.get("player")
        player_id = int(player_id) if player_id is not None else None
        team = str(shot.get("team") or "Unknown")
        launch_position = self._position(shot)
        target_goal_x = self.pitch_length_m if launch_position[0] <= self.pitch_length_m / 2 else 0.0
        target_goal = (target_goal_x, self.pitch_width_m / 2.0)
        distance = float(shot.get("distance_m") or self._distance(launch_position, target_goal))
        angle = float(shot.get("angle_to_goal_deg") or shot.get("angle_deg") or self._goal_angle(launch_position, target_goal_x))
        y_centre_distance = abs(launch_position[1] - self.pitch_width_m / 2.0)
        defenders, attackers = self._nearby_players(
            launch_position=launch_position,
            shooter_team=team,
            average_positions=average_positions or {},
        )
        pressure = self._pressure_score(distance, defenders, attackers)
        sequence_length = self._pass_sequence_length(frame, team, pass_events or [])

        return XGFeatures(
            shot_id=shot_id,
            player_id=player_id,
            team=team,
            frame=frame,
            match_time_s=round(frame / fps, 2) if fps else None,
            shot_x=round(launch_position[0], 2),
            shot_y=round(launch_position[1], 2),
            distance_m=round(distance, 2),
            angle_deg=round(angle, 2),
            distance_from_centre_m=round(y_centre_distance, 2),
            goal_mouth_visibility=self._goal_visibility(angle, distance, defenders),
            ball_speed_mps=float(shot.get("ball_speed_mps") or shot.get("ball_speed") or 0.0),
            shooter_speed_mps=shot.get("shooter_speed_mps"),
            defenders_nearby=defenders,
            attackers_nearby=attackers,
            pressure_score=pressure,
            goalkeeper_distance_m=shot.get("goalkeeper_distance_m"),
            pass_sequence_length=sequence_length,
            possession_duration_s=shot.get("possession_duration_s"),
            open_play=bool(shot.get("open_play", True)),
            set_piece=bool(shot.get("set_piece", False)),
            counter_attack=bool(shot.get("counter_attack", False)),
            body_part=str(shot.get("body_part") or "unknown"),
            shot_type=str(shot.get("shot_type") or "unknown"),
        )

    def to_model_vector(self, features: XGFeatures) -> List[float]:
        payload = features.to_dict()
        return [float(payload.get(column) or 0.0) for column in self.FEATURE_COLUMNS]

    def _position(self, shot: JsonDict) -> Tuple[float, float]:
        raw = (
            shot.get("launch_position")
            or shot.get("start_position")
            or shot.get("position")
            or [self.pitch_length_m / 2.0, self.pitch_width_m / 2.0]
        )
        if len(raw) < 2:
            return self.pitch_length_m / 2.0, self.pitch_width_m / 2.0
        return float(raw[0]), float(raw[1])

    def _distance(self, source: Tuple[float, float], target: Tuple[float, float]) -> float:
        return math.hypot(target[0] - source[0], target[1] - source[1])

    def _goal_angle(self, source: Tuple[float, float], goal_x: float) -> float:
        left_post = (goal_x, (self.pitch_width_m - self.goal_width_m) / 2.0)
        right_post = (goal_x, (self.pitch_width_m + self.goal_width_m) / 2.0)
        a = self._distance(source, left_post)
        b = self._distance(source, right_post)
        c = self.goal_width_m
        denominator = max(2 * a * b, 1e-9)
        cosine = max(-1.0, min(1.0, (a * a + b * b - c * c) / denominator))
        return math.degrees(math.acos(cosine))

    def _goal_visibility(self, angle_deg: float, distance_m: float, defenders: int) -> float:
        angle_component = min(angle_deg / 60.0, 1.0)
        distance_component = max(0.2, 1.0 - min(distance_m, 35.0) / 45.0)
        pressure_penalty = max(0.35, 1.0 - defenders * 0.12)
        return round(max(0.0, min(1.0, angle_component * distance_component * pressure_penalty)), 3)

    def _nearby_players(
        self,
        launch_position: Tuple[float, float],
        shooter_team: str,
        average_positions: JsonDict,
        radius_m: float = 8.0,
    ) -> Tuple[int, int]:
        defenders = 0
        attackers = 0
        for player in average_positions.values():
            pos = player.get("average_position") if isinstance(player, dict) else None
            if not pos or len(pos) < 2:
                continue
            distance = self._distance(launch_position, (float(pos[0]), float(pos[1])))
            if distance > radius_m:
                continue
            if str(player.get("team")) == shooter_team:
                attackers += 1
            elif str(player.get("team")) not in {"Unknown", "Free Ball", ""}:
                defenders += 1
        return defenders, max(attackers - 1, 0)

    def _pressure_score(self, distance_m: float, defenders: int, attackers: int) -> float:
        defender_component = min(defenders / 5.0, 1.0)
        distance_component = max(0.0, 1.0 - distance_m / 35.0) * 0.25
        support_relief = min(attackers / 5.0, 1.0) * 0.2
        return round(max(0.0, min(1.0, defender_component + distance_component - support_relief)), 3)

    def _pass_sequence_length(
        self,
        frame: int,
        team: str,
        pass_events: Iterable[JsonDict],
        lookback_frames: int = 300,
    ) -> int:
        count = 0
        for event in pass_events:
            event_frame = int(event.get("frame_end") or event.get("frame_start") or 0)
            if event.get("team") == team and 0 <= frame - event_frame <= lookback_frames:
                count += 1
        return count
