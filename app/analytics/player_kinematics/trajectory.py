"""
Trajectory Cleaning Module

Cleans raw player tracking data by removing duplicates, missing coordinates,
large jumps, and short tracks. Interpolates short gaps and stores clean
world positions.
"""

import logging
from typing import Dict, List, Tuple, Optional
import numpy as np

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)


class TrajectoryCleaner:
    """
    Cleans raw player trajectory data by removing artifacts and interpolating gaps.
    """

    def __init__(
        self,
        max_jump_m: float = 5.0,
        min_track_frames: int = 5,
        max_gap_frames: int = 3,
        duplicate_threshold_m: float = 0.1
    ):
        """
        Initializes TrajectoryCleaner.

        Args:
            max_jump_m: Maximum allowed displacement between consecutive frames (meters).
            min_track_frames: Minimum number of frames for a track to be valid.
            max_gap_frames: Maximum consecutive missing frames to interpolate.
            duplicate_threshold_m: Distance threshold below which positions are duplicates.
        """
        self.max_jump_m = max_jump_m
        self.min_track_frames = min_track_frames
        self.max_gap_frames = max_gap_frames
        self.duplicate_threshold_m = duplicate_threshold_m

    def clean_trajectory(
        self,
        track_data: List[Dict]
    ) -> Optional[List[Dict]]:
        """
        Cleans a single player's trajectory data.

        Args:
            track_data: List of dicts with keys: track_id, frame_number, timestamp,
                       world_position, confidence.

        Returns:
            Cleaned list of dicts with added 'clean_world_position', or None if
            track is too short/invalid.
        """
        if not track_data or len(track_data) < 2:
            return None

        track_id = track_data[0]['track_id']
        cleaned = []
        last_valid = None

        for i, point in enumerate(track_data):
            if 'world_position' not in point or point['world_position'] is None:
                continue

            pos = point['world_position']
            frame = point['frame_number']
            ts = point['timestamp']
            conf = point.get('confidence', 0.0)

            # Skip None coordinates
            if pos[0] is None or pos[1] is None or np.isnan(pos[0]) or np.isnan(pos[1]):
                continue

            # Handle first valid point
            if last_valid is None:
                point['clean_world_position'] = pos
                cleaned.append(point)
                last_valid = {'pos': pos, 'frame': frame, 'ts': ts}
                continue

            dist = np.sqrt((pos[0] - last_valid['pos'][0])**2 +
                          (pos[1] - last_valid['pos'][1])**2)

            # Skip duplicate
            if dist < self.duplicate_threshold_m:
                continue

            # Detect large jump (tracking artifact)
            if dist > self.max_jump_m:
                gap = frame - last_valid['frame']
                if gap <= self.max_gap_frames:
                    # Interpolate between last_valid and this point
                    interp = self._interpolate(
                        last_valid['pos'], last_valid['ts'],
                        pos, ts, gap
                    )
                    for j, ip in enumerate(interp):
                        interp_point = {
                            'track_id': track_id,
                            'frame_number': last_valid['frame'] + j + 1,
                            'timestamp': last_valid['ts'] + (ts - last_valid['ts']) * (j + 1) / (gap + 1),
                            'world_position': ip,
                            'clean_world_position': ip,
                            'confidence': conf,
                            'interpolated': True
                        }
                        cleaned.append(interp_point)
                    last_valid = {'pos': pos, 'frame': frame, 'ts': ts}
                    # Include the current point
                    point['clean_world_position'] = pos
                    cleaned.append(point)
                    continue
                else:
                    # Gap too large, skip this point but keep last_valid
                    continue

            # Valid position
            point['clean_world_position'] = pos
            cleaned.append(point)
            last_valid = {'pos': pos, 'frame': frame, 'ts': ts}

        if len(cleaned) < self.min_track_frames:
            return None

        return cleaned

    @staticmethod
    def _interpolate(
        start_pos: Tuple[float, float],
        start_ts: float,
        end_pos: Tuple[float, float],
        end_ts: float,
        gap: int
    ) -> List[Tuple[float, float]]:
        """Linearly interpolates positions between two known points."""
        points = []
        for i in range(1, gap + 1):
            t = i / (gap + 1)
            x = start_pos[0] + t * (end_pos[0] - start_pos[0])
            y = start_pos[1] + t * (end_pos[1] - start_pos[1])
            points.append((x, y))
        return points

    def clean_tracks_batch(
        self,
        all_tracks: Dict[int, List[Dict]]
    ) -> Tuple[Dict[int, List[Dict]], Dict]:
        """
        Cleans trajectories for all players.

        Args:
            all_tracks: Dict mapping track_id to list of raw track dicts.

        Returns:
            Tuple of (cleaned_tracks, validation_issues).
        """
        cleaned_tracks = {}
        issues = {
            'rejected_too_short': 0,
            'total_frames_processed': 0,
            'total_valid_frames': 0,
            'total_interpolated_frames': 0
        }

        for track_id, track_data in all_tracks.items():
            issues['total_frames_processed'] += len(track_data)
            result = self.clean_trajectory(track_data)
            if result is None:
                issues['rejected_too_short'] += 1
                logger.info("Track %d rejected: too short after cleaning", track_id)
            else:
                cleaned_tracks[track_id] = result
                issues['total_valid_frames'] += len(result)
                issues['total_interpolated_frames'] += sum(
                    1 for p in result if p.get('interpolated', False)
                )

        issues['rejection_rate'] = (
            issues['rejected_too_short'] / len(all_tracks) if all_tracks else 0.0
        )

        return cleaned_tracks, issues