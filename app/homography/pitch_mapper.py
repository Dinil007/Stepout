"""
Pitch Mapper Engine Module

Main homography coordinate mapping engine. Converts 2D image bounding boxes 
(bottom-center feet positions) into real-world 2D pitch coordinates (in meters/pixels),
maintains trajectory histories for track IDs, and supports batch processing.
"""

from dataclasses import dataclass, asdict
import logging
from typing import List, Tuple, Dict, Union, Optional
import numpy as np

from app.homography.homography_utils import (
    compute_homography,
    transform_point,
    transform_points
)

# Configure module logger
logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)


@dataclass
class PlayerMapping:
    """
    Data Transfer Object representing mapped spatial position of a tracked player.
    """
    track_id: int
    team_id: Union[int, str]
    frame_number: int
    pixel_position: Tuple[float, float]
    field_position: Tuple[float, float]

    def to_dict(self) -> Dict:
        """Serializes the data structure to a Python dictionary."""
        return asdict(self)


class PitchMapper:
    """
    High-performance engine for transforming tracked player bounding boxes
    into real-world 2D football pitch coordinates using homography perspective matrices.
    """

    def __init__(
        self,
        homography_matrix: Optional[np.ndarray] = None,
        max_history_length: int = 300
    ):
        """
        Initializes the PitchMapper engine.

        Args:
            homography_matrix: Optional 3x3 Homography matrix.
            max_history_length: Maximum trajectory history frames to buffer per track ID.
        """
        self.homography_matrix: Optional[np.ndarray] = homography_matrix
        self.max_history_length: int = max_history_length
        self.player_histories: Dict[int, List[Tuple[float, float]]] = {}

    def load_homography(
        self,
        source_points: Union[List[Tuple[float, float]], np.ndarray],
        destination_points: Union[List[Tuple[float, float]], np.ndarray]
    ) -> np.ndarray:
        """
        Computes and loads the 3x3 Homography matrix from source and destination point sets.

        Args:
            source_points: Source image coordinates (N x 2).
            destination_points: Target pitch coordinates (N x 2).

        Returns:
            The computed 3x3 Homography matrix.
        """
        matrix, _ = compute_homography(source_points, destination_points)
        self.homography_matrix = matrix
        logger.info("PitchMapper loaded new Homography matrix.")
        return self.homography_matrix

    @staticmethod
    def extract_player_position(bbox: Tuple[float, float, float, float]) -> Tuple[float, float]:
        """
        Extracts the player's ground contact point (bottom-center of bounding box).

        Args:
            bbox: Bounding box tuple (x1, y1, x2, y2).

        Returns:
            Bottom-center point (center_x, y2).
        """
        x1, y1, x2, y2 = bbox
        center_x = (float(x1) + float(x2)) / 2.0
        bottom_y = float(y2)
        return center_x, bottom_y

    def update_history(self, track_id: int, field_position: Tuple[float, float]) -> None:
        """
        Appends the updated pitch position to the player's historical trajectory.

        Args:
            track_id: Unique tracking ID.
            field_position: Transformed 2D pitch coordinate (x, y).
        """
        if track_id not in self.player_histories:
            self.player_histories[track_id] = []

        self.player_histories[track_id].append(field_position)

        # Truncate history if buffer exceeds limit
        if len(self.player_histories[track_id]) > self.max_history_length:
            self.player_histories[track_id].pop(0)

    def get_player_history(self, track_id: int) -> List[Tuple[float, float]]:
        """
        Retrieves the trajectory position history for a given player track ID.

        Args:
            track_id: Player track ID.

        Returns:
            List of 2D pitch position tuples.
        """
        return self.player_histories.get(track_id, [])

    def clear_history(self, track_id: Optional[int] = None) -> None:
        """
        Clears trajectory history.

        Args:
            track_id: Optional track ID to clear. If None, resets all player histories.
        """
        if track_id is not None:
            self.player_histories.pop(track_id, None)
        else:
            self.player_histories.clear()

    def map_player(
        self,
        track_id: int,
        bbox: Tuple[float, float, float, float],
        team_id: Union[int, str],
        frame_number: int
    ) -> PlayerMapping:
        """
        Maps a single tracked player detection to pitch coordinates.

        Args:
            track_id: Player unique track ID.
            bbox: Bounding box (x1, y1, x2, y2).
            team_id: Team identifier.
            frame_number: Current frame index.

        Returns:
            PlayerMapping instance.
        """
        if self.homography_matrix is None:
            raise RuntimeError("Homography matrix is not loaded. Call load_homography() first.")

        pixel_pos = self.extract_player_position(bbox)
        field_pos = transform_point(pixel_pos, self.homography_matrix)

        if track_id != -1:
            self.update_history(track_id, field_pos)

        return PlayerMapping(
            track_id=track_id,
            team_id=team_id,
            frame_number=frame_number,
            pixel_position=pixel_pos,
            field_position=field_pos
        )

    def map_players(
        self,
        tracked_players: List[Dict],
        frame_number: int
    ) -> List[PlayerMapping]:
        """
        Vectorized batch transformation of multiple tracked players in a frame.

        Args:
            tracked_players: List of dicts containing 'track_id', 'bbox', 'team_id'.
            frame_number: Current frame index.

        Returns:
            List of PlayerMapping DTO instances.
        """
        if self.homography_matrix is None:
            raise RuntimeError("Homography matrix is not loaded. Call load_homography() first.")

        if not tracked_players:
            return []

        pixel_positions = []
        player_metadata = []

        for p in tracked_players:
            bbox = p.get("bbox", (0, 0, 0, 0))
            track_id = p.get("track_id", -1)
            team_id = p.get("team_id", "Unknown")

            pixel_pos = self.extract_player_position(bbox)
            pixel_positions.append(pixel_pos)
            player_metadata.append((track_id, team_id))

        # Perform high-performance batch perspective transformation
        field_positions = transform_points(pixel_positions, self.homography_matrix)

        mapped_players = []
        for idx, (field_pos, (track_id, team_id)) in enumerate(zip(field_positions, player_metadata)):
            pixel_pos = pixel_positions[idx]

            if track_id != -1:
                self.update_history(track_id, field_pos)

            mapped_players.append(
                PlayerMapping(
                    track_id=track_id,
                    team_id=team_id,
                    frame_number=frame_number,
                    pixel_position=pixel_pos,
                    field_position=field_pos
                )
            )

        return mapped_players

    def process_frame(
        self,
        tracked_players: List[Dict],
        frame_number: int
    ) -> List[PlayerMapping]:
        """
        Processes single-frame detections and returns mapped player coordinates.

        Args:
            tracked_players: Detections for current frame.
            frame_number: Frame index.

        Returns:
            List of PlayerMapping objects.
        """
        return self.map_players(tracked_players, frame_number)

    def run(
        self,
        tracking_results_by_frame: List[List[Dict]]
    ) -> List[List[PlayerMapping]]:
        """
        Executes pitch coordinate mapping across a full sequence of frames.

        Args:
            tracking_results_by_frame: Outer list per frame, inner list per player detection.

        Returns:
            Outer list per frame, inner list of PlayerMapping DTOs.
        """
        logger.info(f"Processing batch of {len(tracking_results_by_frame)} frames...")
        results = []
        for frame_idx, frame_detections in enumerate(tracking_results_by_frame):
            mapped_frame = self.process_frame(frame_detections, frame_number=frame_idx + 1)
            results.append(mapped_frame)

        logger.info("Batch pitch mapping execution complete.")
        return results
