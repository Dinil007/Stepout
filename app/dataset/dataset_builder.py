"""Dataset builder: orchestrates crop extraction and saving across frames."""
from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from app.dataset.crop_extractor import CropExtractor, CropResult


@dataclass
class DatasetBuildConfig:
    enabled: bool = True
    save_every_n_frames: int = 5
    crop_size: int = 256
    padding: bool = True
    output_dir: str = "datasets/person_classifier"
    contact_sheet_cols: int = 5
    contact_sheet_rows: int = 4


@dataclass
class TrackCropRecord:
    track_id: int
    frame: int
    timestamp: float
    bbox: Tuple[int, int, int, int]
    confidence: float
    crop_width: int
    crop_height: int
    image_path: str
    reason: str = ""


@dataclass
class TrackSummary:
    track_id: int
    first_frame: int
    last_frame: int
    total_frames: int
    saved_images: int
    average_confidence: float


class DatasetBuilder:
    def __init__(self, config: DatasetBuildConfig) -> None:
        self.config = config
        self.output_root = Path(config.output_dir)
        self.raw_dir = self.output_root / "raw"
        self.extractor = CropExtractor(crop_size=config.crop_size, padding=config.padding)

        self.track_records: Dict[int, List[TrackCropRecord]] = defaultdict(list)
        self.rejected: List[Dict] = []
        self.total_saved = 0
        self.total_rejected = 0
        self.start_time = 0.0
        self.end_time = 0.0
        self.frames_processed = 0

    def setup(self) -> None:
        self.raw_dir.mkdir(parents=True, exist_ok=True)

    def process_frame(self, frame: np.ndarray, frame_number: int, timestamp: float, player_tracks: Dict[int, Dict]) -> List[TrackCropRecord]:
        self.frames_processed += 1
        records: List[TrackCropRecord] = []
        if not self.config.enabled:
            return records
        if self.frames_processed % self.config.save_every_n_frames != 0:
            return records
        for track_id, pdata in player_tracks.items():
            bbox = pdata.get("bbox")
            conf = float(pdata.get("confidence", 0.0))
            if bbox is None:
                continue
            x1, y1, x2, y2 = [int(v) for v in bbox]
            result: CropResult = self.extractor.extract(frame, (x1, y1, x2, y2), confidence=conf, frame_number=frame_number, timestamp=timestamp)
            track_folder = self.raw_dir / f"track_{track_id:04d}"
            track_folder.mkdir(parents=True, exist_ok=True)
            saved_idx = len(self.track_records[track_id]) + 1
            frame_name = f"frame_{frame_number:06d}.jpg"
            out_path = track_folder / frame_name
            if result.success:
                cv2.imwrite(str(out_path), result.crop)
                rec = TrackCropRecord(
                    track_id=track_id,
                    frame=frame_number,
                    timestamp=timestamp,
                    bbox=result.bbox,
                    confidence=result.confidence,
                    crop_width=result.crop_width,
                    crop_height=result.crop_height,
                    image_path=str(out_path.relative_to(self.output_root)),
                )
                records.append(rec)
                self.track_records[track_id].append(rec)
                self.total_saved += 1
                print(f"[DATASET] Track {track_id} | Frame {frame_number} | Crop saved -> {out_path}")
            else:
                self.total_rejected += 1
                rej = {
                    "track_id": track_id,
                    "frame": frame_number,
                    "reason": result.reason,
                }
                self.rejected.append(rej)
                print(f"[DATASET] Track {track_id} | Frame {frame_number} | Rejected ({result.reason})")
        return records

    def get_summaries(self) -> List[TrackSummary]:
        summaries: List[TrackSummary] = []
        for track_id, recs in self.track_records.items():
            if not recs:
                continue
            frames = [r.frame for r in recs]
            confs = [r.confidence for r in recs]
            summaries.append(TrackSummary(
                track_id=track_id,
                first_frame=min(frames),
                last_frame=max(frames),
                total_frames=len(set(frames)),
                saved_images=len(recs),
                average_confidence=float(np.mean(confs)) if confs else 0.0,
            ))
        summaries.sort(key=lambda s: s.track_id)
        return summaries

    def get_statistics(self, processing_fps: float) -> Dict:
        summaries = self.get_summaries()
        total_tracks = len(summaries)
        total_crops = self.total_saved
        avg_per_track = float(np.mean([s.saved_images for s in summaries])) if summaries else 0.0
        avg_crop_size = 0.0
        crops_sizes = [(r.crop_width, r.crop_height) for recs in self.track_records.values() for r in recs]
        if crops_sizes:
            avg_crop_size = float(np.mean([w for w, h in crops_sizes]))
        return {
            "total_tracks": total_tracks,
            "total_crops": total_crops,
            "average_crops_per_track": round(avg_per_track, 2),
            "average_crop_size": round(avg_crop_size, 2) if avg_crop_size else 0.0,
            "rejected_crops": self.total_rejected,
            "rejection_reasons": {},
            "processing_fps": round(processing_fps, 2),
        }