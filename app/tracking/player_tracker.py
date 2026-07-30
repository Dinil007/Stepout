"""
Player Tracking wrapper using ByteTrack + Appearance ReID.

Pipeline:
  ByteTrack (Primary Motion Tracker)
      ↓
  Appearance ReID (OSNet) → Embedding Extraction
      ↓
  Motion + Appearance Association → Score Fusion
      ↓
  Track Refinement → ID Recovery → Metrics

ByteTrack remains the primary tracker for:
- Kalman prediction
- Motion estimation
- IoU matching
- Lost track handling
- Track lifecycle

ReID is an ADDITIONAL appearance verification layer.

Do NOT modify ByteTrack.
Do NOT replace ByteTrack.
"""

import logging
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from app.core.config import get_config
from app.detection.detection_types import Detection
from app.tracking.association import MotionAppearanceAssociator
from app.tracking.bytetrack import BYTETracker
from app.tracking.embedding_cache import EmbeddingCacheManager
from app.tracking.reid import OSNetReID
from app.tracking.tracker_config import ReIDConfig, TrackerConfig
from app.tracking.track_manager import TrackManager
from app.tracking.tracker_metrics import TrackerMetrics

logger = logging.getLogger(__name__)


class PlayerTracker:
    """
    Wraps ByteTrack for player detection tracking with ReID enhancement.

    ByteTrack handles all motion-based tracking internally.
    ReID adds appearance-based identity verification and ID recovery.
    """

    def __init__(self, config: Optional[Dict] = None) -> None:
        cfg = config or get_config().raw
        self.tracker_cfg = TrackerConfig.from_dict(cfg)
        tracking_cfg = cfg.get("tracking", {})

        # ByteTrack parameters
        self.tracker_type = tracking_cfg.get("tracker_type", "bytetrack")
        self.track_high_thresh = float(tracking_cfg.get("track_high_thresh", 0.35))
        self.track_low_thresh = float(tracking_cfg.get("track_low_thresh", 0.08))
        self.new_track_thresh = float(tracking_cfg.get("new_track_thresh", 0.28))
        self.track_buffer = int(tracking_cfg.get("track_buffer", 150))
        self.match_thresh = float(tracking_cfg.get("match_thresh", 0.72))
        self.fuse_score = bool(tracking_cfg.get("fuse_score", True))
        self.min_track_frames = int(tracking_cfg.get("min_track_frames", 2))

        tracker_config_path = tracking_cfg.get("tracker_config", "app/tracking/bytetrack_custom.yaml")
        self.tracker_config = Path(tracker_config_path)

        # ByteTrack tracker (PRIMARY)
        self.tracker: Optional[BYTETracker] = None

        # ReID components (APPEARANCE LAYER)
        self.reid_enabled = self.tracker_cfg.reid.enabled
        self.reid: Optional[OSNetReID] = None
        self.associator: Optional[MotionAppearanceAssociator] = None
        self.cache_manager: Optional[EmbeddingCacheManager] = None
        self.track_manager: Optional[TrackManager] = None
        self.metrics: Optional[TrackerMetrics] = None

        # Legacy tracking state (preserved for backward compatibility)
        self.track_presence: Dict[int, List[int]] = defaultdict(list)
        self.prev_lost: Dict[int, Tuple[int, Tuple[int, int]]] = {}
        self.possible_switches: List[Dict] = []

        # Frame buffer for ReID cropping (stores last frame)
        self._last_frame: Optional[np.ndarray] = None

        # Initialize all components
        try:
            self.load()
        except Exception as e:
            logger.warning(f"PlayerTracker.load() failed: {e}")

    def load(self) -> None:
        """Initialize ByteTrack tracker and ReID components."""
        # ByteTrack initialization
        self.tracker = BYTETracker(
            str(self.tracker_config),
            frame_rate=25,
        )
        logger.info("ByteTrack tracker initialized")

        # ReID initialization (if enabled and torchreid available)
        if self.reid_enabled:
            try:
                self.reid = OSNetReID(self.tracker_cfg.reid)
                self.reid.load()
                self.associator = MotionAppearanceAssociator(self.tracker_cfg.reid)
                self.cache_manager = EmbeddingCacheManager(
                    max_history=self.tracker_cfg.reid.max_embedding_history,
                    max_lost_frames=self.tracker_cfg.reid.max_lost_frames,
                )
                self.track_manager = TrackManager(
                    config=self.tracker_cfg.reid,
                    associator=self.associator,
                    cache_manager=self.cache_manager,
                )
                self.metrics = TrackerMetrics()
                logger.info("ReID components initialized successfully")
            except Exception as e:
                logger.warning(f"ReID initialization failed (continuing without ReID): {e}")
                self.reid_enabled = False
        else:
            logger.info("ReID disabled, using ByteTrack only")

    def update(
        self,
        detections: List[Detection],
        frame_shape: Tuple[int, int],
        frame_no: int,
        frame: Optional[np.ndarray] = None,
    ) -> List[Detection]:
        """
        Update tracks with new detections.

        Args:
            detections: List of YOLO detections
            frame_shape: (height, width) of frame
            frame_no: Current frame number
            frame: Original video frame (needed for ReID cropping)

        Returns:
            List of tracked Detection objects with track IDs
        """
        if self.tracker is None:
            raise RuntimeError("PlayerTracker.load() must be called before update()")

        if not detections:
            return []

        # Store frame for ReID cropping
        if frame is not None:
            self._last_frame = frame

        # ==========================================
        # STEP 1: ByteTrack Motion Tracking (PRIMARY)
        # ==========================================
        dets_array = []
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            # ByteTrack expects [x1, y1, x2, y2, conf, cls]
            dets_array.append([x1, y1, x2, y2, det.conf, float(det.cls_id)])

        # ByteTrack update
        tracks = self.tracker.update(
            dets_array,
            [(frame_shape[0], frame_shape[1])],
            ((frame_shape[0], frame_shape[1]),),
        )


        # ==========================================
        # STEP 2: Convert ByteTrack Output to Detection objects
        # ==========================================
        tracked_dets = []
        active_ids = set()

        for track in tracks:
            # ByteTrack returns [x1, y1, width, height, class_id, track_id, conf, keep_flag, index]
            if len(track) < 9:
                continue
            x1 = float(track[0])
            y1 = float(track[1])
            w = float(track[2])
            h = float(track[3])
            class_id = float(track[4])
            track_id = int(track[5])
            conf = float(track[6])
            keep_flag = track[7]
            index = track[8]

            # Convert xywh to xyxy
            x2 = x1 + w
            y2 = y1 + h

            # Verification print once for the first track
            if frame_no == 0 and track_id == 1:
                print(f"[VERIFY] ByteTrack -> Detection conversion:")
                print(f"  Original ByteTrack row: {track}")
                print(f"  Track ID: {track_id}")
                print(f"  BBox xyxy: ({x1:.4f}, {y1:.4f}, {x2:.4f}, {y2:.4f})")
                print(f"  Confidence: {conf:.4f}")
                print(f"")

            # Find matching original detection by center proximity
            matched_det = None
            min_dist = float('inf')
            track_cx = (x1 + x2) / 2.0
            track_cy = (y1 + y2) / 2.0
            
            det_matched_flag = getattr(self, '_det_matched_flag', None)
            if det_matched_flag is None:
                object.__setattr__(self, '_det_matched_flag', {})
                det_matched_flag = getattr(self, '_det_matched_flag')
            
            for det in detections:
                det_cx = (det.bbox[0] + det.bbox[2]) / 2.0
                det_cy = (det.bbox[1] + det.bbox[3]) / 2.0
                dist = math.hypot(det_cx - track_cx, det_cy - track_cy)
                if dist < min_dist:
                    min_dist = dist
                    matched_det = det

            # Detect reuse of the same detection object by multiple tracks
            matched_det_id = id(matched_det) if matched_det is not None else None
            if matched_det_id is not None:
                reuse_count = det_matched_flag.get(matched_det_id, 0) + 1
                det_matched_flag[matched_det_id] = reuse_count
                if reuse_count > 1:
                    logger.warning(f"Frame {frame_no}: detection object reused across tracks (count={reuse_count})")

            # Match if within reasonable distance (100px)
            if matched_det and min_dist < 100.0:
                tracked_det = Detection(
                    cls_id=int(class_id),
                    conf=conf,
                    bbox=(int(x1), int(y1), int(x2), int(y2)),
                    track_id=track_id,
                )
                tracked_dets.append(tracked_det)
                active_ids.add(track_id)

                # Legacy tracking (preserved)
                self.track_presence[track_id].append(frame_no)

        # ==========================================
        # STEP 3: ReID Appearance Verification (OPTIONAL)
        # ==========================================
        if self.reid_enabled and self._last_frame is not None and self.reid is not None and self.reid.is_loaded:
            tracked_dets = self._apply_reid(
                tracked_dets, active_ids, frame_no, frame_shape
            )

        # ==========================================
        # STEP 4: Lost Track Handling
        # ==========================================
        self._update_lost_tracks(active_ids, frame_no)

        # ==========================================
        # STEP 5: Metrics Recording
        # ==========================================
        if self.metrics is not None:
            self.metrics.record_frame(
                frame=frame_no,
                num_detections=len(detections),
                num_tracked=len(tracked_dets),
                active_tracks=len(active_ids),
            )

        return tracked_dets

    def _apply_reid(
        self,
        tracked_dets: List[Detection],
        active_ids: set,
        frame_no: int,
        frame_shape: Tuple[int, int],
    ) -> List[Detection]:
        """
        Apply ReID appearance verification to tracked detections.

        This is a POST-PROCESSING step that runs AFTER ByteTrack association.
        It extracts appearance embeddings and updates caches, but does NOT
        change ByteTrack's core tracking decisions.
        """
        if self._last_frame is None or self.reid is None or self.cache_manager is None or self.track_manager is None:
            return tracked_dets

        # Prepare crops for ReID
        crops = []
        valid_indices = []
        for idx, det in enumerate(tracked_dets):
            x1, y1, x2, y2 = det.bbox
            # Ensure crop is within frame bounds
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(frame_shape[1] - 1, x2)
            y2 = min(frame_shape[0] - 1, y2)

            if x2 > x1 and y2 > y1:
                crop = self._last_frame[y1:y2, x1:x2]
                if crop.size > 0:
                    crops.append(crop)
                    valid_indices.append(idx)

        if not crops:
            return tracked_dets

        # Batch embedding extraction
        try:
            embeddings = self.reid.extract_embeddings_batch(crops)
        except Exception as e:
            logger.warning(f"ReID embedding extraction failed: {e}")
            return tracked_dets

        # Process each tracked detection with its embedding
        refined_dets = []
        for idx, emb in zip(valid_indices, embeddings):
            det = tracked_dets[idx]
            track_id = det.track_id
            cx, cy = det.center

            # Get or create embedding cache for this track
            cache = self.cache_manager.get_or_create(track_id)
            cache.add_embedding(
                embedding=emb,
                frame=frame_no,
                confidence=det.conf,
                center=(float(cx), float(cy)),
            )

            # Compute appearance similarity with running average
            avg_emb = cache.get_average_embedding()
            appearance_sim = float(np.dot(emb, avg_emb)) if avg_emb is not None else 0.5

            # Verify match with associator
            final_score = 0.5
            if self.associator is not None:
                _, _, final_score = self.associator.verify_match(
                    track_embedding=avg_emb,
                    detection_embedding=emb,
                    motion_score=0.5,
                )

            # Track manager registration (for ID recovery of new tracks)
            recovery_status = "none"
            if cache.get_history_size() <= 2:
                recovered_id = self.track_manager.register_track(
                    track_id=track_id,
                    frame=frame_no,
                    center=(float(cx), float(cy)),
                    confidence=det.conf,
                    embedding=emb,
                )
                if recovered_id != track_id:
                    logger.info(
                        f"ReID recovered track {track_id} -> {recovered_id} "
                        f"(sim={appearance_sim:.3f})"
                    )
                    det.track_id = recovered_id
                    track_id = recovered_id
                    recovery_status = "recovered"
                    active_ids.discard(track_id)
                    active_ids.add(recovered_id)
            else:
                # For well-established tracks, just update state
                if self.track_manager is not None:
                    self.track_manager.register_track(
                        track_id=track_id,
                        frame=frame_no,
                        center=(float(cx), float(cy)),
                        confidence=det.conf,
                    )

            # Log debug data
            if self.metrics is not None:
                self.metrics.record_reid_debug(
                    frame=frame_no,
                    track_id=track_id,
                    embedding_similarity=appearance_sim,
                    motion_score=0.5,
                    final_score=final_score if self.associator else 0.5,
                    track_age=cache.get_track_age(frame_no),
                    embedding_history_size=cache.get_history_size(),
                    recovery_status=recovery_status,
                )

            refined_dets.append(det)

        return refined_dets

    def _update_lost_tracks(self, active_ids: set, frame_no: int) -> None:
        """
        Update lost track state.

        Only marks tracks as lost if they were active in the previous frame
        but are missing now. This matches ByteTrack's internal lost-track logic.
        """
        # Legacy ID switch detection (preserved)
        for tid in list(self.prev_lost.keys()):
            if tid in active_ids:
                del self.prev_lost[tid]
            else:
                lost_frame, _ = self.prev_lost[tid]
                if frame_no - lost_frame > 1:
                    del self.prev_lost[tid]

        # ReID lost track handling - only mark as lost if was active before
        if self.track_manager is not None:
            # Only consider tracks that were previously active in track_manager
            current_tm_active = set(self.track_manager.get_active_track_ids())
            newly_lost = current_tm_active - active_ids
            
            for lost_id in newly_lost:
                self.track_manager.mark_lost(lost_id, frame_no)

            # Remove stale lost tracks
            self.track_manager.remove_stale(frame_no)

        # Legacy lost track tracking
        for tid, frames in list(self.track_presence.items()):
            if frames and frames[-1] < frame_no and frame_no - frames[-1] == 1:
                last_center = (0, 0)
                self.prev_lost[tid] = (frame_no, last_center)

    def get_metrics(self, current_frame: Optional[int] = None) -> Dict:
        """Get tracking metrics."""
        # Legacy metrics
        lifetimes = []
        lost_tracks = 0
        recovered = 0

        for tid, frames in self.track_presence.items():
            if frames:
                lifetimes.append(len(frames))
                if current_frame is not None and frames[-1] < current_frame:
                    lost_tracks += 1
                    gaps = [b - a for a, b in zip(frames, frames[1:])]
                    recovered += sum(1 for g in gaps if g > 1)

        avg_lifetime = sum(lifetimes) / len(lifetimes) if lifetimes else 0.0
        max_lifetime = max(lifetimes) if lifetimes else 0

        # ReID metrics
        reid_metrics = {}
        if self.track_manager is not None:
            reid_metrics = self.track_manager.get_stats()
        if self.metrics is not None:
            reid_metrics["summary"] = self.metrics.get_summary()
        if self.cache_manager is not None:
            reid_metrics["cache"] = self.cache_manager.get_stats()

        return {
            "total_unique_tracks": len(self.track_presence),
            "active_tracks": sum(
                1 for frames in self.track_presence.values()
                if frames and (current_frame is None or frames[-1] == current_frame)
            ),
            "lost_tracks": lost_tracks,
            "recovered_tracks": recovered,
            "average_track_lifetime": round(avg_lifetime, 2),
            "longest_track_lifetime": max_lifetime,
            "fragmentation_index": round(
                recovered / max(len(self.track_presence), 1), 3
            ),
            "estimated_id_switches": len(self.possible_switches),
            "possible_id_switch_events": self.possible_switches[:100],
            "reid": reid_metrics,
        }

    def flush_metrics(self) -> None:
        """Write all pending metrics to disk."""
        if self.metrics is not None:
            self.metrics.flush_all()

    def write_tracking_report(
        self,
        video_resolution: str = "",
        fps: float = 0.0,
        processing_fps: float = 0.0,
    ) -> None:
        """Write comprehensive tracking report."""
        if self.metrics is not None and self.track_manager is not None:
            self.metrics.write_tracking_report(
                track_manager=self.track_manager,
                cache_manager=self.cache_manager,
                associator=self.associator,
                video_resolution=video_resolution,
                fps=fps,
                processing_fps=processing_fps,
            )

    def is_reid_active(self) -> bool:
        """Check if ReID enhancement is active."""
        return (
            self.reid_enabled
            and self.reid is not None
            and self.reid.is_loaded
        )

    def get_reid_metrics(self) -> Dict:
        """Get ReID-specific metrics."""
        if self.reid is not None and self.reid.is_loaded:
            return self.reid.get_metrics()
        return {"status": "disabled"}