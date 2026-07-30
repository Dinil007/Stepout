"""
Enhanced Tracking Metrics with ReID statistics.

Generates comprehensive tracking reports including:
- Standard ByteTrack metrics
- ReID appearance similarity statistics
- ID recovery and switch analysis
- Per-track detailed history
"""

import csv
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from app.tracking.association import MotionAppearanceAssociator
from app.tracking.embedding_cache import EmbeddingCacheManager
from app.tracking.track_manager import TrackManager

logger = logging.getLogger(__name__)


class TrackerMetrics:
    """
    Collects and reports comprehensive tracking metrics.

    Generates:
    - outputs/reid_debug.csv: Per-frame ReID debug data
    - outputs/id_switch_report.csv: Detected ID switch events
    - outputs/tracking_report.txt: Summary tracking report
    """

    def __init__(self, output_dir: str = "outputs") -> None:
        """
        Args:
            output_dir: Directory for output files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Per-frame debug data
        self.reid_debug_rows: List[Dict] = []

        # ID switch events
        self.id_switch_rows: List[Dict] = []

        # Aggregate stats
        self.total_frames: int = 0
        self.total_detections: int = 0
        self.total_tracked: int = 0
        self.appearance_scores: List[float] = []
        self.motion_scores: List[float] = []
        self.final_scores: List[float] = []
        self.embedding_similarities: List[float] = []
        self.active_track_counts: List[int] = []

    def record_frame(
        self,
        frame: int,
        num_detections: int,
        num_tracked: int,
        active_tracks: int,
    ) -> None:
        """
        Record per-frame tracking statistics.

        Args:
            frame: Frame number
            num_detections: Number of raw detections
            num_tracked: Number of successfully tracked objects
            active_tracks: Number of active track IDs
        """
        self.total_frames += 1
        self.total_detections += num_detections
        self.total_tracked += num_tracked
        self.active_track_counts.append(active_tracks)

    def record_reid_debug(
        self,
        frame: int,
        track_id: int,
        embedding_similarity: float,
        motion_score: float,
        final_score: float,
        track_age: int,
        embedding_history_size: int,
        recovery_status: str = "none",
    ) -> None:
        """
        Record per-track ReID debug data.

        Args:
            frame: Frame number
            track_id: Track ID
            embedding_similarity: Cosine similarity to track average
            motion_score: ByteTrack motion score
            final_score: Combined motion + appearance score
            track_age: Track age in frames
            embedding_history_size: Number of stored embeddings
            recovery_status: 'none', 'recovered', or 'new'
        """
        self.reid_debug_rows.append({
            "frame": frame,
            "track_id": track_id,
            "embedding_similarity": round(embedding_similarity, 4),
            "motion_score": round(motion_score, 4),
            "final_score": round(final_score, 4),
            "track_age": track_age,
            "embedding_history_size": embedding_history_size,
            "recovery_status": recovery_status,
        })

        self.embedding_similarities.append(embedding_similarity)
        self.motion_scores.append(motion_score)
        self.final_scores.append(final_score)

    def record_id_switch(
        self,
        frame: int,
        old_track_id: int,
        new_track_id: int,
        cosine_similarity: float,
        motion_distance: float,
        association_score: float,
        reason: str,
    ) -> None:
        """
        Record a detected ID switch event.

        Args:
            frame: Frame number
            old_track_id: Previous track ID
            new_track_id: New track ID
            cosine_similarity: Appearance similarity
            motion_distance: Spatial distance in pixels
            association_score: Combined score
            reason: Reason for detection
        """
        self.id_switch_rows.append({
            "frame": frame,
            "old_track_id": old_track_id,
            "new_track_id": new_track_id,
            "cosine_similarity": round(cosine_similarity, 4),
            "motion_distance": round(motion_distance, 2),
            "association_score": round(association_score, 4),
            "reason": reason,
        })

    def flush_reid_debug(self) -> None:
        """Write ReID debug CSV."""
        if not self.reid_debug_rows:
            return

        path = self.output_dir / "reid_debug.csv"
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "frame", "track_id", "embedding_similarity",
                "motion_score", "final_score", "track_age",
                "embedding_history_size", "recovery_status",
            ])
            writer.writeheader()
            writer.writerows(self.reid_debug_rows)

        logger.info(f"ReID debug data written to {path} ({len(self.reid_debug_rows)} rows)")

    def flush_id_switch_report(self) -> None:
        """Write ID switch report CSV."""
        if not self.id_switch_rows:
            return

        path = self.output_dir / "id_switch_report.csv"
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "frame", "old_track_id", "new_track_id",
                "cosine_similarity", "motion_distance",
                "association_score", "reason",
            ])
            writer.writeheader()
            writer.writerows(self.id_switch_rows)

        logger.info(f"ID switch report written to {path} ({len(self.id_switch_rows)} rows)")

    def generate_tracking_report(
        self,
        track_manager: TrackManager,
        cache_manager: EmbeddingCacheManager,
        associator: MotionAppearanceAssociator,
        video_resolution: str = "",
        fps: float = 0.0,
        processing_fps: float = 0.0,
    ) -> str:
        """
        Generate comprehensive tracking report.

        Args:
            track_manager: Track manager with state
            cache_manager: Embedding cache manager
            associator: Motion + Appearance associator
            video_resolution: Video resolution string
            fps: Video FPS
            processing_fps: Processing speed

        Returns:
            Formatted report string
        """
        track_stats = track_manager.get_stats()
        cache_stats = cache_manager.get_stats()
        assoc_config = associator.get_config()

        avg_active = (
            sum(self.active_track_counts) / len(self.active_track_counts)
            if self.active_track_counts else 0
        )

        avg_appearance = (
            np.mean(self.embedding_similarities)
            if self.embedding_similarities else 0.0
        )
        avg_motion = (
            np.mean(self.motion_scores)
            if self.motion_scores else 0.0
        )
        avg_final = (
            np.mean(self.final_scores)
            if self.final_scores else 0.0
        )

        report = f"""
{'=' * 60}
TRACKING REPORT
{'=' * 60}

VIDEO INFORMATION
  Resolution: {video_resolution}
  FPS: {fps:.1f}
  Processed Frames: {self.total_frames}
  Processing FPS: {processing_fps:.1f}

TRACKING STATISTICS
  Average Active Tracks: {avg_active:.1f}
  Unique Track IDs: {track_stats['total_tracks_seen']}
  Average Track Duration: {track_stats['avg_active_track_length']:.1f} frames
  Longest Track Duration: {max(
    (t.get_track_length() for t in track_manager.get_all_track_info()),
    default=0
  )} frames

RECOVERY STATISTICS
  Recovery Attempts: {track_stats['recovery_attempts']}
  Successful Recoveries: {track_stats['successful_recoveries']}
  Recovery Rate: {track_stats['recovery_rate'] * 100:.1f}%
  Lost Tracks: {track_stats['lost_tracks']}
  Estimated ID Switches: {track_stats['id_switches_detected']}

APPEARANCE SIMILARITY
  Average Appearance Similarity: {avg_appearance:.4f}
  Average Motion Score: {avg_motion:.4f}
  Average Final Score: {avg_final:.4f}

ASSOCIATION CONFIGURATION
  Motion Weight: {assoc_config['motion_weight']}
  Appearance Weight: {assoc_config['appearance_weight']}
  Similarity Threshold: {assoc_config['similarity_threshold']}

EMBEDDING CACHE
  Active Caches: {cache_stats['active_tracks']}
  Lost Caches: {cache_stats['lost_tracks']}
  Total Embeddings Stored: {cache_stats['total_embeddings_stored']}
  Avg Active Embedding Age: {cache_stats['avg_active_embedding_age']} frames
  Avg Lost Age: {cache_stats['avg_lost_age_frames']} frames

{'=' * 60}
END OF TRACKING REPORT
{'=' * 60}
"""
        return report

    def write_tracking_report(
        self,
        track_manager: TrackManager,
        cache_manager: EmbeddingCacheManager,
        associator: MotionAppearanceAssociator,
        video_resolution: str = "",
        fps: float = 0.0,
        processing_fps: float = 0.0,
    ) -> None:
        """
        Write tracking report to file.

        Args:
            track_manager: Track manager
            cache_manager: Embedding cache manager
            associator: Motion + Appearance associator
            video_resolution: Video resolution string
            fps: Video FPS
            processing_fps: Processing speed
        """
        report = self.generate_tracking_report(
            track_manager, cache_manager, associator,
            video_resolution, fps, processing_fps,
        )

        path = self.output_dir / "tracking_report.txt"
        with open(path, "w") as f:
            f.write(report)

        logger.info(f"Tracking report written to {path}")
        print(report)

    def flush_all(self) -> None:
        """Write all pending data to disk."""
        self.flush_reid_debug()
        self.flush_id_switch_report()

    def get_summary(self) -> Dict:
        """Get summary metrics as dict."""
        return {
            "total_frames": self.total_frames,
            "total_detections": self.total_detections,
            "total_tracked": self.total_tracked,
            "tracking_rate": round(
                self.total_tracked / max(self.total_detections, 1), 4
            ),
            "avg_active_tracks": round(
                np.mean(self.active_track_counts), 1
            ) if self.active_track_counts else 0,
            "avg_appearance_similarity": round(
                np.mean(self.embedding_similarities), 4
            ) if self.embedding_similarities else 0.0,
            "avg_motion_score": round(
                np.mean(self.motion_scores), 4
            ) if self.motion_scores else 0.0,
            "avg_final_score": round(
                np.mean(self.final_scores), 4
            ) if self.final_scores else 0.0,
            "id_switches_detected": len(self.id_switch_rows),
            "reid_debug_entries": len(self.reid_debug_rows),
        }