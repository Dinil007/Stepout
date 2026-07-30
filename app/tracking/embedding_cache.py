"""
Embedding Cache - Per-track appearance feature storage.

Each active track maintains:
- Embedding History (max N embeddings)
- Running Average Embedding
- Latest Embedding
- Cache metadata (age, confidence, etc.)

Automatically discards oldest embeddings when history exceeds max size.
"""

import logging
from collections import deque
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class TrackEmbeddingCache:
    """
    Stores and manages appearance embeddings for a single track.

    Maintains:
    - A deque of recent embeddings (max N)
    - Running average embedding (for stable matching)
    - Latest embedding (for immediate comparison)
    - Metadata: track age, visibility count, lost count, avg confidence
    """

    def __init__(self, track_id: int, max_history: int = 20) -> None:
        """
        Args:
            track_id: ByteTrack track ID
            max_history: Maximum number of embeddings to store
        """
        self.track_id: int = track_id
        self.max_history: int = max_history
        self.history: Deque[np.ndarray] = deque(maxlen=max_history)
        self.running_avg: Optional[np.ndarray] = None
        self.latest: Optional[np.ndarray] = None

        # Metadata
        self.first_seen: int = 0
        self.last_seen: int = 0
        self.frames_visible: int = 0
        self.frames_lost: int = 0
        self.frames_recovered: int = 0
        self.confidence_sum: float = 0.0
        self.confidence_count: int = 0
        self.trajectory: Deque[Tuple[float, float]] = deque(maxlen=100)

    def add_embedding(
        self, embedding: np.ndarray, frame: int, confidence: float,
        center: Optional[Tuple[float, float]] = None
    ) -> None:
        """
        Add a new embedding to the cache.

        Args:
            embedding: L2-normalized feature vector
            frame: Current frame number
            confidence: Detection confidence
            center: (x, y) center position (for trajectory tracking)
        """
        if self.first_seen == 0:
            self.first_seen = frame

        self.latest = embedding
        self.history.append(embedding)
        self.last_seen = frame
        self.frames_visible += 1
        self.confidence_sum += confidence
        self.confidence_count += 1

        # Update running average
        self.running_avg = np.mean(list(self.history), axis=0).astype(np.float32)
        # Re-normalize the average
        norm = np.linalg.norm(self.running_avg)
        if norm > 0:
            self.running_avg /= norm

        # Update trajectory
        if center is not None:
            self.trajectory.append(center)

    def mark_lost(self) -> None:
        """Increment lost frame counter."""
        self.frames_lost += 1

    def mark_recovered(self) -> None:
        """Increment recovery counter."""
        self.frames_recovered += 1

    def get_average_embedding(self) -> Optional[np.ndarray]:
        """
        Get the running average embedding.

        Uses the running average if available, otherwise returns the latest
        embedding, or None if no embeddings exist.

        Returns:
            L2-normalized average embedding or None
        """
        return self.running_avg if self.running_avg is not None else self.latest

    def get_latest_embedding(self) -> Optional[np.ndarray]:
        """Get the most recent embedding."""
        return self.latest

    def get_embedding_history(self) -> List[np.ndarray]:
        """Get all stored embeddings as a list."""
        return list(self.history)

    def get_avg_confidence(self) -> float:
        """Get average detection confidence for this track."""
        if self.confidence_count == 0:
            return 0.0
        return self.confidence_sum / self.confidence_count

    def get_track_age(self, current_frame: int) -> int:
        """Get track age in frames."""
        return current_frame - self.first_seen

    def get_embedding_age(self, current_frame: int) -> int:
        """Get age of the last embedding in frames."""
        return current_frame - self.last_seen

    def get_history_size(self) -> int:
        """Get number of stored embeddings."""
        return len(self.history)

    def is_stale(self, current_frame: int, max_lost_frames: int) -> bool:
        """
        Check if this track is stale (lost for too long).

        Args:
            current_frame: Current frame number
            max_lost_frames: Maximum allowed lost frames

        Returns:
            True if track has been lost longer than max_lost_frames
        """
        return (current_frame - self.last_seen) > max_lost_frames

    def to_dict(self) -> Dict:
        """Serialize cache state for debugging."""
        return {
            "track_id": self.track_id,
            "first_seen": self.first_seen,
            "last_seen": self.last_seen,
            "frames_visible": self.frames_visible,
            "frames_lost": self.frames_lost,
            "frames_recovered": self.frames_recovered,
            "history_size": len(self.history),
            "avg_confidence": round(self.get_avg_confidence(), 4),
            "trajectory_length": len(self.trajectory),
        }


class EmbeddingCacheManager:
    """
    Manages embedding caches for all tracks.

    Provides methods to:
    - Get or create cache for a track
    - Remove stale caches
    - Find visually similar tracks among lost tracks
    """

    def __init__(self, max_history: int = 20, max_lost_frames: int = 25) -> None:
        """
        Args:
            max_history: Maximum embeddings per track
            max_lost_frames: Frames before a lost track cache is removed
        """
        self.max_history = max_history
        self.max_lost_frames = max_lost_frames
        self._caches: Dict[int, TrackEmbeddingCache] = {}
        self._lost_caches: Dict[int, TrackEmbeddingCache] = {}

    def get_or_create(self, track_id: int) -> TrackEmbeddingCache:
        """
        Get existing cache for track_id or create a new one.

        If the track was previously lost and its cache still exists,
        it's moved back from lost to active.

        Args:
            track_id: ByteTrack track ID

        Returns:
            TrackEmbeddingCache for this track
        """
        if track_id in self._caches:
            return self._caches[track_id]

        # Check if it was previously lost
        if track_id in self._lost_caches:
            cache = self._lost_caches.pop(track_id)
            cache.mark_recovered()
            self._caches[track_id] = cache
            logger.debug(f"Track {track_id} recovered from lost cache")
            return cache

        # Create new cache
        cache = TrackEmbeddingCache(track_id, self.max_history)
        self._caches[track_id] = cache
        return cache

    def mark_lost(self, track_id: int) -> None:
        """
        Mark a track as lost. Moves cache to lost storage.

        Args:
            track_id: ByteTrack track ID
        """
        if track_id in self._caches:
            cache = self._caches.pop(track_id)
            cache.mark_lost()
            self._lost_caches[track_id] = cache

    def remove_stale(self, current_frame: int) -> None:
        """
        Remove caches for tracks lost beyond the threshold.

        Args:
            current_frame: Current frame number
        """
        stale_ids = [
            tid for tid, cache in self._lost_caches.items()
            if cache.is_stale(current_frame, self.max_lost_frames)
        ]
        for tid in stale_ids:
            del self._lost_caches[tid]

        logger.debug(f"Removed {len(stale_ids)} stale caches, "
                     f"{len(self._lost_caches)} lost caches remaining")

    def find_similar_lost(
        self,
        query_embedding: np.ndarray,
        similarity_threshold: float,
        current_frame: int,
    ) -> List[Tuple[int, float]]:
        """
        Find lost tracks with appearance similar to query.

        Args:
            query_embedding: L2-normalized embedding to match
            similarity_threshold: Minimum cosine similarity
            current_frame: Current frame (for time gap check)

        Returns:
            List of (track_id, similarity) sorted by similarity descending
        """
        results = []
        for tid, cache in self._lost_caches.items():
            avg_emb = cache.get_average_embedding()
            if avg_emb is None:
                continue

            # Cosine similarity (dot product of L2-normalized vectors)
            similarity = float(np.dot(query_embedding, avg_emb))

            if similarity >= similarity_threshold:
                results.append((tid, similarity))

        # Sort by similarity descending
        results.sort(key=lambda x: x[1], reverse=True)
        return results

    def get_active_cache(self, track_id: int) -> Optional[TrackEmbeddingCache]:
        """Get active cache for a track."""
        return self._caches.get(track_id)

    def get_lost_cache(self, track_id: int) -> Optional[TrackEmbeddingCache]:
        """Get lost cache for a track."""
        return self._lost_caches.get(track_id)

    def get_all_active(self) -> List[TrackEmbeddingCache]:
        """Get all active track caches."""
        return list(self._caches.values())

    def get_all_lost(self) -> List[TrackEmbeddingCache]:
        """Get all lost track caches."""
        return list(self._lost_caches.values())

    def clear(self) -> None:
        """Clear all caches."""
        self._caches.clear()
        self._lost_caches.clear()

    def get_stats(self) -> Dict:
        """Get cache manager statistics."""
        active_ages = [
            cache.get_track_age(cache.last_seen)
            for cache in self._caches.values()
        ]
        lost_ages = [
            cache.get_track_age(cache.last_seen)
            for cache in self._lost_caches.values()
        ]
        return {
            "active_tracks": len(self._caches),
            "lost_tracks": len(self._lost_caches),
            "avg_active_embedding_age": round(
                sum(active_ages) / len(active_ages), 1
            ) if active_ages else 0,
            "avg_lost_age_frames": round(
                sum(lost_ages) / len(lost_ages), 1
            ) if lost_ages else 0,
            "total_embeddings_stored": sum(
                len(c.history) for c in self._caches.values()
            ) + sum(len(c.history) for c in self._lost_caches.values()),
        }