"""
Ball Possession Module

Determines which player has possession of the ball using distance to feet,
player confidence, ball confidence, and temporal consistency to avoid flickering.
"""

import logging
from typing import Dict, List, Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)


class PossessionDetector:
    """
    Detects ball possession using distance and confidence gating.
    Tracks team-level possession percentages and frame-by-frame history.
    """

    def __init__(
        self,
        possession_radius_m: float = 1.5,
        min_possession_duration_frames: int = 3,
        confidence_weight: float = 0.7,
        fps: float = 25.0
    ):
        self.possession_radius_m = possession_radius_m
        self.min_possession_duration_frames = min_possession_duration_frames
        self.confidence_weight = confidence_weight
        self.fps = fps

        # Total frames processed
        self._total_frames: int = 0
        # Frames with no possession (ball not near any player)
        self._no_possession_frames: int = 0
        # Counters per team: team_name -> frames
        self._team_possession_frames: Dict[str, int] = {}
        # History for visualization: list of dicts per frame
        self.history: List[Dict] = []

        # Current state
        self.current_possessor: Optional[int] = None
        self.current_team: Optional[str] = None
        self.possession_frames: int = 0
        self.candidate_id: Optional[int] = None
        self.candidate_streak: int = 0
        self.possession_events: List[Dict] = []

    def update(
        self,
        ball_position_m: Optional[Tuple[float, float]],
        player_positions_m: Dict[int, Tuple[float, float]],
        player_teams: Dict[int, str],
        frame_number: int,
        ball_confidence: float = 1.0,
        player_confidences: Optional[Dict[int, float]] = None
    ) -> Optional[Dict]:
        """
        Updates possession state for current frame.

        Args:
            ball_position_m: Ball position in meters.
            player_positions_m: Dict of track_id to position.
            player_teams: Dict of track_id to team id.
            frame_number: Current frame number.
            ball_confidence: Ball detection confidence.
            player_confidences: Optional dict of player confidences.

        Returns:
            Possession dict or None.
        """
        self._total_frames += 1

        if ball_position_m is None or not player_positions_m:
            self._no_possession_frames += 1
            self.candidate_streak = 0
            self.history.append({
                "frame": frame_number,
                "possessor": None,
                "team": None,
                "state": "Free Ball",
                "distance_m": None
            })
            return None

        ball_pos = np.array(ball_position_m)
        candidates = []

        for track_id, player_pos in player_positions_m.items():
            dist = np.linalg.norm(ball_pos - np.array(player_pos))
            if dist <= self.possession_radius_m:
                p_conf = player_confidences.get(track_id, 1.0) if player_confidences else 1.0
                score = (1.0 - dist / self.possession_radius_m) * (1.0 - self.confidence_weight) + p_conf * self.confidence_weight
                candidates.append((track_id, dist, score))

        if not candidates:
            self._no_possession_frames += 1
            self.candidate_streak = 0
            self.history.append({
                "frame": frame_number,
                "possessor": None,
                "team": None,
                "state": "Free Ball",
                "distance_m": None
            })
            return None

        best_track_id, best_dist, best_score = max(candidates, key=lambda c: c[2])

        # Anti-flicker filter with candidate streak
        if self.candidate_id == best_track_id:
            self.candidate_streak += 1
        else:
            self.candidate_id = best_track_id
            self.candidate_streak = 1

        if self.current_possessor == best_track_id:
            self.possession_frames += 1
        else:
            if self.current_possessor is not None and self.possession_frames >= self.min_possession_duration_frames:
                self.possession_events.append({
                    'frame_start': frame_number - self.possession_frames,
                    'frame_end': frame_number - 1,
                    'track_id': self.current_possessor,
                    'team_id': player_teams.get(self.current_possessor, 'Unknown')
                })
            self.current_possessor = best_track_id
            self.possession_frames = 1

        team = player_teams.get(best_track_id, 'Unknown')
        self.current_team = team

        # Update team possession frames
        if team != 'Unknown':
            self._team_possession_frames[team] = self._team_possession_frames.get(team, 0) + 1

        result = {
            'track_id': best_track_id,
            'team_id': team,
            'distance_m': round(best_dist, 3),
            'frame_number': frame_number,
            'is_current': True
        }

        self.history.append({
            "frame": frame_number,
            "possessor": best_track_id,
            "team": team,
            "state": "In Possession",
            "distance_m": round(best_dist, 3)
        })

        return result

    def get_possession_percentage(self) -> Dict[str, float]:
        """
        Returns team-level possession split as percentages.

        Formula:
            Team Possession % = (Team Frames / Total Frames) × 100
            Free Ball % = (No Possession Frames / Total Frames) × 100
        """
        effective_frames = max(self._total_frames, 1)
        result = {}

        for team_name, frames in self._team_possession_frames.items():
            result[f"{team_name}_pct"] = round((frames / effective_frames) * 100.0, 1)

        result["Free_Ball_pct"] = round((self._no_possession_frames / effective_frames) * 100.0, 1)
        return result

    def get_team_possession_summary(self) -> Dict:
        """
        Returns detailed possession summary including time conversions.
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

    def get_possession_summary(self) -> Dict:
        total_possession_frames = sum(e['frame_end'] - e['frame_start'] + 1 for e in self.possession_events)
        return {
            'total_possession_events': len(self.possession_events),
            'total_possession_frames': total_possession_frames,
            'total_frames_processed': self._total_frames,
            'no_possession_frames': self._no_possession_frames,
            'team_possession_pct': self.get_possession_percentage(),
            'team_possession_summary': self.get_team_possession_summary()
        }
