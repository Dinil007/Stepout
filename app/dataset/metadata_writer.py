"""Metadata writers: metadata.csv, track_summary.csv, labels.csv."""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np

from app.dataset.dataset_builder import DatasetBuilder, TrackCropRecord, TrackSummary


class MetadataWriter:
    def __init__(self, output_root: Path) -> None:
        self.output_root = Path(output_root)

    def write_metadata(self, records: List[TrackCropRecord]) -> None:
        csv_path = self.output_root / "metadata.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["track_id", "frame", "timestamp", "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2", "confidence", "crop_width", "crop_height", "image_path"])
            for r in records:
                writer.writerow([
                    r.track_id,
                    r.frame,
                    f"{r.timestamp:.6f}",
                    r.bbox[0],
                    r.bbox[1],
                    r.bbox[2],
                    r.bbox[3],
                    f"{r.confidence:.6f}",
                    r.crop_width,
                    r.crop_height,
                    r.image_path,
                ])

    def write_track_summary(self, summaries: List[TrackSummary]) -> None:
        csv_path = self.output_root / "track_summary.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["track_id", "first_frame", "last_frame", "total_frames", "saved_images", "average_confidence"])
            for s in summaries:
                writer.writerow([
                    s.track_id,
                    s.first_frame,
                    s.last_frame,
                    s.total_frames,
                    s.saved_images,
                    f"{s.average_confidence:.6f}",
                ])

    def write_labels(self, track_ids: List[int]) -> None:
        csv_path = self.output_root / "labels.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["track_id", "label", "status"])
            for tid in sorted(track_ids):
                writer.writerow([tid, "", "UNLABELED"])

    def write_report(self, stats: dict, builder: DatasetBuilder) -> None:
        report_path = self.output_root / "dataset_report.txt"
        lines = [
            "Dataset Generation Report",
            "=" * 60,
            f"Total tracks: {stats.get('total_tracks', 0)}",
            f"Total crops: {stats.get('total_crops', 0)}",
            f"Average crops per track: {stats.get('average_crops_per_track', 0)}",
            f"Average crop size: {stats.get('average_crop_size', 0)}",
            f"Rejected crops: {stats.get('rejected_crops', 0)}",
            f"Rejection reasons: {stats.get('rejection_reasons', {})}",
            f"Processing FPS: {stats.get('processing_fps', 0)}",
        ]
        report_path.write_text("\n".join(lines), encoding="utf-8")