"""
Ball Touch Detection Module

Detects individual ball touches including start, end, duration, count, and location.
"""

import logging
from typing import Dict, List, Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)


class TouchDetector:
    """
    Detects ball touch events based on possession changes and ball movement.
    """

    def __init__(
        self,
        min_touch_duration_frames: int = 1,
        max_touch_gap_frames: int = 5
    ):
        self.min_touch_duration_frames = min_touch_duration_frames
        self.max_touch_gap_frames = max_touch_gap_frames
        self.touches: List[Dict] = []
        self.current_touch: Optional[Dict] = None

    def update(
        self,
        possession: Optional[Dict],
        ball_position: Optional[Tuple[float, float]],
        frame_number: int,
        timestamp: float
    ) -> Optional[Dict]:
        """
        Updates touch detection for current frame.

        Args:
            possession: Current possession dict or None.
            ball_position: Ball position in meters.
            frame_number: Current frame number.
            timestamp: Current timestamp.

        Returns:
            Current touch dict or None.
        """
        if possession is None or ball_position is None:
            if self.current_touch is not None:
                self._end_touch(frame_number, timestamp)
            return None

        track_id = possession['track_id']

        if self.current_touch is None:
            self.current_touch = {
                'track_id': track_id,
                'team_id': possession['team_id'],
                'start_frame': frame_number,
                'end_frame': frame_number,
                'start_timestamp': timestamp,
                'end_timestamp': timestamp,
                'positions': [ball_position]
            }
        elif self.current_touch['track_id'] != track_id:
            self._end_touch(frame_number, timestamp)
            self.current_touch = {
                'track_id': track_id,
                'team_id': possession['team_id'],
                'start_frame': frame_number,
                'end_frame': frame_number,
                'start_timestamp': timestamp,
                'end_timestamp': timestamp,
                'positions': [ball_position]
            }
        else:
            self.current_touch['end_frame'] = frame_number
            self.current_touch['end_timestamp'] = timestamp
            self.current_touch['positions'].append(ball_position)

        return self.current_touch

    def _end_touch(self, frame_number: int, timestamp: float) -> None:
        if self.current_touch is None:
            return
        duration_frames = self.current_touch['end_frame'] - self.current_touch['start_frame'] + 1
        if duration_frames >= self.min_touch_duration_frames:
            touch = {
                'track_id': self.current_touch['track_id'],
                'team_id': self.current_touch['team_id'],
                'start_frame': self.current_touch['start_frame'],
                'end_frame': self.current_touch['end_frame'],
                'start_timestamp': self.current_touch['start_timestamp'],
                'end_timestamp': self.current_touch['end_timestamp'],
                'duration_frames': duration_frames,
                'duration_s': round(self.current_touch['end_timestamp'] - self.current_touch['start_timestamp'], 3),
                'start_position': self.current_touch['positions'][0],
                'end_position': self.current_touch['positions'][-1]
            }
            self.touches.append(touch)
        self.current_touch = None

    def finalize(self) -> List[Dict]:
        """Returns completed touches and resets state."""
        # End any ongoing touch
        # Note: caller should provide final frame/timestamp if needed
        return self.touches

    def get_summary(self) -> Dict:
        return {
            'total_touches': len(self.touches),
            'average_touch_duration_frames': round(np.mean([t['duration_frames'] for t in self.touches]), 2) if self.touches else 0.0
        }