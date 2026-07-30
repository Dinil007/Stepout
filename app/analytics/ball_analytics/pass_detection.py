"""
Pass Detection Module

Detects passes using rule-based logic:
- Player A possesses the ball
- Ball leaves Player A
- Ball travels
- Player B gains possession
"""

import logging
from typing import Dict, List, Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)


class PassDetector:
    """
    Detects pass events between players based on possession changes and ball movement.
    """

    def __init__(
        self,
        min_pass_distance_m: float = 1.0,
        max_pass_duration_s: float = 5.0,
        min_pass_speed_kmh: float = 5.0
    ):
        self.min_pass_distance_m = min_pass_distance_m
        self.max_pass_duration_s = max_pass_duration_s
        self.min_pass_speed_kmh = min_pass_speed_kmh
        self.passes: List[Dict] = []
        self.current_pass: Optional[Dict] = None

    def update(
        self,
        possession: Optional[Dict],
        ball_position: Optional[Tuple[float, float]],
        ball_speed_kmh: float,
        frame_number: int,
        timestamp: float,
        player_positions_m: Dict[int, Tuple[float, float]]
    ) -> Optional[Dict]:
        """
        Updates pass detection for current frame.

        Args:
            possession: Current possession dict.
            ball_position: Ball position in meters.
            ball_speed_kmh: Ball speed in km/h.
            frame_number: Current frame number.
            timestamp: Current timestamp.
            player_positions_m: All player positions.

        Returns:
            Completed pass dict if a pass just finished, else None.
        """
        if possession is None or ball_position is None:
            if self.current_pass is not None:
                self._end_pass(frame_number, timestamp)
            return None

        track_id = possession['track_id']
        ball_pos = np.array(ball_position)

        # Start new pass if possession changed and we have a previous possessor
        if self.current_pass is None:
            # Start tracking a potential pass
            self.current_pass = {
                'passer_id': track_id,
                'passer_team_id': possession['team_id'],
                'start_frame': frame_number,
                'end_frame': frame_number,
                'start_timestamp': timestamp,
                'end_timestamp': timestamp,
                'start_position': ball_pos,
                'end_position': ball_pos,
                'speeds': [ball_speed_kmh]
            }
        else:
            # Continue or end current pass
            self.current_pass['end_frame'] = frame_number
            self.current_pass['end_timestamp'] = timestamp
            self.current_pass['end_position'] = ball_pos
            self.current_pass['speeds'].append(ball_speed_kmh)

            # Check if pass completed (possession changed to different player)
            if self.current_pass['passer_id'] != track_id:
                return self._end_pass(frame_number, timestamp)

        return None

    def _end_pass(self, frame_number: int, timestamp: float) -> Optional[Dict]:
        if self.current_pass is None:
            return None

        duration = timestamp - self.current_pass['start_timestamp']
        distance = float(np.linalg.norm(self.current_pass['end_position'] - self.current_pass['start_position']))
        avg_speed = float(np.mean(self.current_pass['speeds'])) if self.current_pass['speeds'] else 0.0

        # Validate pass
        if distance < self.min_pass_distance_m or duration > self.max_pass_duration_s or avg_speed < self.min_pass_speed_kmh:
            self.current_pass = None
            return None

        pass_event = {
            'passer_id': self.current_pass['passer_id'],
            'passer_team_id': self.current_pass['passer_team_id'],
            'receiver_id': frame_number,  # placeholder; updated by caller
            'receiver_team_id': '',
            'start_frame': self.current_pass['start_frame'],
            'end_frame': self.current_pass['end_frame'],
            'start_timestamp': self.current_pass['start_timestamp'],
            'end_timestamp': self.current_pass['end_timestamp'],
            'duration_s': round(duration, 3),
            'distance_m': round(distance, 2),
            'average_speed_kmh': round(avg_speed, 2),
            'max_speed_kmh': round(max(self.current_pass['speeds']), 2),
            'successful': True
        }
        self.passes.append(pass_event)
        self.current_pass = None
        return pass_event

    def finalize(self) -> List[Dict]:
        return self.passes

    def get_summary(self) -> Dict:
        total_passes = len(self.passes)
        total_distance = sum(p['distance_m'] for p in self.passes)
        return {
            'total_passes': total_passes,
            'total_pass_distance_m': round(total_distance, 2),
            'average_pass_distance_m': round(total_distance / total_passes, 2) if total_passes else 0.0,
            'successful_passes': total_passes
        }