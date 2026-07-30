"""
Track Manager - Track state management with ID recovery.

Manages the lifecycle of player tracks:
- Track creation, update, and deletion
- ID recovery for re-appearing players after occlusion
- Track state transitions (active → lost → recovered → active)
- ID switch detection and logging

This module works WITH ByteTrack, not instead of it.
ByteTrack handles motion-based tracking; this adds appearance-based ID recovery.
"""

import logging
import math
from typing import Dict, List, Optional, Tuple

import numpy as np

from app.tracking.association import MotionAppearanceAssociator
from app.tracking.embedding_cache import EmbeddingCacheManager
from app.tracking.tracker_config import ReIDConfig

logger = logging.getLogger(__name__)


class TrackState:
    """Track state constants."""
    ACTIVE = "active"
    LOST = "lost"
    RECOVERED = "recovered"


class TrackInfo:
    """
    Stores comprehensive state for a single track.

    Tracks:
    - Track Length
    - Frames Visible
    - Frames Lost
    - Frames Recovered
    - Average Confidence
    - Embedding Age
    - Trajectory Length
    """

    def __init__(self, track_id: int, frame: int, center: Tuple[float, float]) -> None:
        self.track_id: int = track_id
        self.state: str = TrackState.ACTIVE
        self.first_frame: int = frame
        self.last_frame: int = frame
        self.frames_visible: int = 1
        self.frames_lost: int = 0
        self.frames_recovered: int = 0
        self.confidence_sum: float = 0.0
        self.confidence_count: int = 0
        self.centers: List[Tuple[float, float]] = [center]
        self.lost_frame: int = 0

    def update(self, frame: int, center: Tuple[float, float], confidence: float) -> None:
        """Update track with new detection."""
        self.last_frame = frame
        self.frames_visible += 1
        self.confidence_sum += confidence
        self.confidence_count += 1
        self.centers.append(center)

    def mark_lost(self, frame: int) -> None:
        """Mark track as lost."""
        self.state = TrackState.LOST
        self.lost_frame = frame
        self.frames_lost += 1

    def mark_recovered(self, frame: int) -> None:
        """Mark track as recovered from lost state."""
        self.state = TrackState.RECOVERED
        self.frames_recovered += 1
        self.last_frame = frame

    def get_avg_confidence(self) -> float:
        """Get average detection confidence."""
        if self.confidence_count == 0:
            return 0.0
        return self.confidence_sum / self.confidence_count

    def get_track_length(self) -> int:
        """Get total track length in frames."""
        return self.last_frame - self.first_frame + 1

    def get_trajectory_length(self) -> int:
        """Get number of trajectory points."""
        return len(self.centers)

    def get_last_center(self) -> Optional[Tuple[float, float]]:
        """Get the last known center position."""
        if self.centers:
            return self.centers[-1]
        return None

    def to_dict(self) -> Dict:
        """Serialize track info for reporting."""
        return {
            "track_id": self.track_id,
            "state": self.state,
            "first_frame": self.first_frame,
            "last_frame": self.last_frame,
            "frames_visible": self.frames_visible,
            "frames_lost": self.frames_lost,
            "frames_recovered": self.frames_recovered,
            "track_length": self.get_track_length(),
            "avg_confidence": round(self.get_avg_confidence(), 4),
            "trajectory_length": self.get_trajectory_length(),
        }


class TrackManager:
    """
    Manages track lifecycle with appearance-based ID recovery.

    Key responsibilities:
    1. Track creation and state management
    2. ID recovery: When ByteTrack creates a new track, check if it matches
       a recently lost track via appearance similarity
    3. ID switch detection and logging
    4. Stale track cleanup
    """

    def __init__(
        self,
        config: ReIDConfig,
        associator: MotionAppearanceAssociator,
        cache_manager: EmbeddingCacheManager,
    ) -> None:
        """
        Args:
            config: ReID configuration
            associator: Motion + Appearance associator
            cache_manager: Embedding cache manager
        """
        self.config = config
        self.associator = associator
        self.cache_manager = cache_manager
        self._tracks: Dict[int, TrackInfo] = {}
        self._lost_tracks: Dict[int, TrackInfo] = {}
        self._id_switches: List[Dict] = []
        self._recovery_attempts: int = 0
        self._successful_recoveries: int = 0

    def register_track(
        self,
        track_id: int,
        frame: int,
        center: Tuple[float, float],
        confidence: float,
        embedding: Optional[np.ndarray] = None,
    ) -> Optional[int]:
        """
        Register a track (new or recovered).

        If ReID is enabled and the track is new, checks if it matches a
        recently lost track via appearance similarity. If a match is found,
        the previous track ID is recovered instead of creating a new one.

        Args:
            track_id: ByteTrack-assigned track ID
            frame: Current frame number
            center: (x, y) center position
            confidence: Detection confidence
            embedding: L2-normalized appearance embedding (optional)

        Returns:
            The track ID to use (may be a recovered previous ID)
        """
        # If track already exists, just update it
        if track_id in self._tracks:
            self._tracks[track_id].update(frame, center, confidence)
            return track_id

        # Check if this is a new track that might match a lost one
        if (
            self.config.reassign_ids
            and embedding is not None
            and self._lost_tracks
        ):
            recovered_id = self._attempt_id_recovery(
                track_id, frame, center, embedding
            )
            if recovered_id is not None:
                return recovered_id

        # Create new track
        track_info = TrackInfo(track_id, frame, center)
        track_info.confidence_sum = confidence
        track_info.confidence_count = 1
        self._tracks[track_id] = track_info
        return track_id

    def _attempt_id_recovery(
        self,
        new_track_id: int,
        frame: int,
        center: Tuple[float, float],
        embedding: np.ndarray,
    ) -> Optional[int]:
        """
        Attempt to recover a previous track ID for a new detection.

        Steps:
        1. Find lost tracks with similar appearance
        2. Check motion distance (spatial proximity)
        3. Check time gap (not too long ago)
        4. If all conditions met, recover the previous ID

        Args:
            new_track_id: ByteTrack's new track ID
            frame: Current frame number
            center: (x, y) center of new detection
            embedding: Appearance embedding of new detection

        Returns:
            Recovered track ID, or None if no good match
        """
        self._recovery_attempts += 1

        # Find similar lost tracks by appearance
        similar_lost = self.cache_manager.find_similar_lost(
            query_embedding=embedding,
            similarity_threshold=self.config.similarity_threshold,
            current_frame=frame,
        )

        for lost_id, similarity in similar_lost:
            lost_info = self._lost_tracks.get(lost_id)
            if lost_info is None:
                continue

            # Check motion distance
            last_center = lost_info.get_last_center()
            if last_center is None:
                continue

            motion_distance = math.hypot(
                center[0] - last_center[0],
                center[1] - last_center[1],
            )

            # Reasonable motion: player can't teleport
            max_reasonable_motion = 200.0  # pixels
            if motion_distance > max_reasonable_motion:
                continue

            # Check time gap
            time_gap = frame - lost_info.last_frame
            if time_gap > self.config.max_lost_frames:
                continue

            # All conditions met - recover this ID
            logger.info(
                f"ID Recovery: new={new_track_id} → recovered={lost_id} "
                f"(sim={similarity:.3f}, dist={motion_distance:.1f}, "
                f"gap={time_gap}frames)"
            )

            # Move track from lost to active
            track_info = self._lost_tracks.pop(lost_id)
            track_info.mark_recovered(frame)
            track_info.centers.append(center)
            self._tracks[lost_id] = track_info

            # Log the recovery
            self._successful_recoveries += 1
            self._id_switches.append({
                "frame": frame,
                "old_track_id": lost_id,
                "new_track_id": new_track_id,
                "cosine_similarity": round(similarity, 4),
                "motion_distance": round(motion_distance, 2),
                "time_gap": time_gap,
                "reason": "appearance_recovery",
            })

            return lost_id

        return None

    def mark_lost(self, track_id: int, frame: int) -> None:
        """
        Mark a track as lost.

        Args:
            track_id: Track ID to mark as lost
            frame: Current frame number
        """
        if track_id in self._tracks:
            track_info = self._tracks.pop(track_id)
            track_info.mark_lost(frame)
            self._lost_tracks[track_id] = track_info
            self.cache_manager.mark_lost(track_id)

    def remove_stale(self, current_frame: int) -> None:
        """
        Remove tracks lost beyond the threshold.

        Args:
            current_frame: Current frame number
        """
        stale_ids = [
            tid for tid, info in self._lost_tracks.items()
            if (current_frame - info.last_frame) > self.config.max_lost_frames
        ]
        for tid in stale_ids:
            del self._lost_tracks[tid]

        self.cache_manager.remove_stale(current_frame)

    def get_active_track_ids(self) -> List[int]:
        """Get all active track IDs."""
        return list(self._tracks.keys())

    def get_track_info(self, track_id: int) -> Optional[TrackInfo]:
        """Get info for a specific track."""
        return self._tracks.get(track_id) or self._lost_tracks.get(track_id)

    def get_all_track_info(self) -> List[TrackInfo]:
        """Get info for all tracks (active + lost)."""
        return list(self._tracks.values()) + list(self._lost_tracks.values())

    def get_id_switches(self) -> List[Dict]:
        """Get list of detected ID switches."""
        return self._id_switches

    def clear(self) -> None:
        """Clear all track state."""
        self._tracks.clear()
        self._lost_tracks.clear()
        self._id_switches.clear()
        self.cache_manager.clear()

    def get_stats(self) -> Dict:
        """Get track manager statistics."""
        active_infos = list(self._tracks.values())
        lost_infos = list(self._lost_tracks.values())

        active_lengths = [t.get_track_length() for t in active_infos]
        lost_lengths = [t.get_track_length() for t in lost_infos]

        return {
            "active_tracks": len(active_infos),
            "lost_tracks": len(lost_infos),
            "total_tracks_seen": len(active_infos) + len(lost_infos),
            "avg_active_track_length": round(
                sum(active_lengths) / len(active_lengths), 1
            ) if active_lengths else 0,
            "avg_lost_track_length": round(
                sum(lost_lengths) / len(lost_lengths), 1
            ) if lost_lengths else 0,
            "recovery_attempts": self._recovery_attempts,
            "successful_recoveries": self._successful_recoveries,
            "recovery_rate": round(
                self._successful_recoveries / max(self._recovery_attempts, 1), 3
            ),
            "id_switches_detected": len(self._id_switches),
        }