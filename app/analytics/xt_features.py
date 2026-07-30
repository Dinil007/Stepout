"""Feature engineering for Expected Threat (xT) calculations."""
from __future__ import annotations

import logging
import math
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

from app.homography.field_config import FIELD_LENGTH_METERS, FIELD_WIDTH_METERS

LOGGER = logging.getLogger(__name__)

JsonDict = Dict[str, Any]


@dataclass(frozen=True)
class XTFeatures:
    """Single-action xT feature vector."""
    event_id: int
    player_id: Optional[int]
    team: str
    action: str  # "pass" or "carry"
    start_x: float
    start_y: float
    end_x: float
    end_y: float
    distance_m: float
    direction_deg: float
    forward_distance_m: float
    lateral_distance_m: float
    ball_speed_mps: float
    progressive: bool
    pressure_score: float
    defenders_nearby: int
    attackers_nearby: int
    start_cell_col: int
    start_cell_row: int
    end_cell_col: int
    end_cell_row: int
    xt_start: float
    xt_end: float
    xt_added: float

    def to_dict(self) -> JsonDict:
        return asdict(self)


class XTFeatureExtractor:
    """Extracts features for xT computation from passes and carries."""

    def __init__(
        self,
        pitch_length_m: float = FIELD_LENGTH_METERS,
        pitch_width_m: float = FIELD_WIDTH_METERS,
    ) -> None:
        self.pitch_length_m = pitch_length_m
        self.pitch_width_m = pitch_width_m

    def extract_pass(
        self,
        pass_event: JsonDict,
        average_positions: Optional[JsonDict] = None,
        xt_start: float = 0.0,
        xt_end: float = 0.0,
        start_cell: Tuple[int, int] = (0, 0),
        end_cell: Tuple[int, int] = (0, 0),
    ) -> XTFeatures:
        return self._extract(
            event=pass_event,
            action="pass",
            start_key="start_position",
            end_key="end_position",
            speed_key="ball_speed_mps",
            distance_key="distance_m",
            average_positions=average_positions,
            xt_start=xt_start,
            xt_end=xt_end,
            start_cell=start_cell,
            end_cell=end_cell,
        )

    def extract_carry(
        self,
        carry_event: JsonDict,
        average_positions: Optional[JsonDict] = None,
        xt_start: float = 0.0,
        xt_end: float = 0.0,
        start_cell: Tuple[int, int] = (0, 0),
        end_cell: Tuple[int, int] = (0, 0),
    ) -> XTFeatures:
        return self._extract(
            event=carry_event,
            action="carry",
            start_key="start_position",
            end_key="end_position",
            speed_key="carry_speed_mps",
            distance_key="distance_m",
            average_positions=average_positions,
            xt_start=xt_start,
            xt_end=xt_end,
            start_cell=start_cell,
            end_cell=end_cell,
        )

    def _extract(
        self,
        event: JsonDict,
        action: str,
        start_key: str,
        end_key: str,
        speed_key: str,
        distance_key: str,
        average_positions: Optional[JsonDict],
        xt_start: float,
        xt_end: float,
        start_cell: Tuple[int, int],
        end_cell: Tuple[int, int],
    ) -> XTFeatures:
        event_id = int(event.get("event_id") or event.get("id") or 0)
        player_id = event.get("passer") if action == "pass" else event.get("player_id") or event.get("player")
        player_id = int(player_id) if player_id is not None else None
        team = str(event.get("team") or "Unknown")

        start_pos = self._position(event, start_key)
        end_pos = self._position(event, end_key)

        distance = float(event.get(distance_key) or self._distance(start_pos, end_pos))
        dx = end_pos[0] - start_pos[0]
        dy = end_pos[1] - start_pos[1]
        direction = math.degrees(math.atan2(dy, dx)) if abs(dx) > 1e-6 else 90.0
        forward_dist = max(0.0, dx)
        lateral_dist = abs(dy)
        speed = float(event.get(speed_key) or 0.0)
        progressive = forward_dist > 5.0

        pressure = self._pressure_score(end_pos, average_positions or {}, team)
        defenders = self._count_nearby(end_pos, average_positions or {}, team, own_team=False)
        attackers = self._count_nearby(end_pos, average_positions or {}, team, own_team=True)

        return XTFeatures(
            event_id=event_id,
            player_id=player_id,
            team=team,
            action=action,
            start_x=round(start_pos[0], 2),
            start_y=round(start_pos[1], 2),
            end_x=round(end_pos[0], 2),
            end_y=round(end_pos[1], 2),
            distance_m=round(distance, 2),
            direction_deg=round(direction, 2),
            forward_distance_m=round(forward_dist, 2),
            lateral_distance_m=round(lateral_dist, 2),
            ball_speed_mps=round(speed, 2),
            progressive=progressive,
            pressure_score=pressure,
            defenders_nearby=defenders,
            attackers_nearby=attackers,
            start_cell_col=start_cell[0],
            start_cell_row=start_cell[1],
            end_cell_col=end_cell[0],
            end_cell_row=end_cell[1],
            xt_start=round(xt_start, 4),
            xt_end=round(xt_end, 4),
            xt_added=round(xt_end - xt_start, 4),
        )

    def _position(self, event: JsonDict, key: str) -> Tuple[float, float]:
        raw = event.get(key) or [self.pitch_length_m / 2.0, self.pitch_width_m / 2.0]
        if len(raw) < 2:
            return self.pitch_length_m / 2.0, self.pitch_width_m / 2.0
        return float(raw[0]), float(raw[1])

    def _distance(self, a: Tuple[float, float], b: Tuple[float, float]) -> float:
        return math.hypot(b[0] - a[0], b[1] - a[1])

    def _pressure_score(
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

    def _count_nearby(
        self, position: Tuple[float, float], avg_positions: JsonDict, team: str, own_team: bool, radius: float = 8.0
    ) -> int:
        count = 0
        for player in avg_positions.values():
            pos = player.get("average_position") if isinstance(player, dict) else None
            if not pos or len(pos) < 2:
                continue
            pteam = str(player.get("team") or "")
            if own_team and pteam != team:
                continue
            if not own_team and pteam in {team, "Unknown", "Free Ball", ""}:
                continue
            dist = self._distance(position, (float(pos[0]), float(pos[1])))
            if dist <= radius:
                count += 1
        return count