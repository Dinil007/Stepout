"""Ball tracking metrics collection."""

import csv
from pathlib import Path
from typing import Dict, List, Optional


class BallTrackingMetricsWriter:
    """Writes per-frame ball tracking metrics to CSV."""

    FIELDS = [
        "frame_index",
        "raw_detection",
        "accepted_detection",
        "predicted",
        "missing",
        "confidence",
        "track_length",
        "coverage_ratio",
    ]

    def __init__(self, output_path: Path) -> None:
        self.output_path = output_path
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.rows: List[Dict] = []

    def record(
        self,
        frame_index: int,
        raw_detection: int,
        accepted_detection: int,
        predicted: bool,
        missing: bool,
        confidence: float,
        track_length: int,
        coverage_ratio: float,
    ) -> None:
        self.rows.append(
            {
                "frame_index": frame_index,
                "raw_detection": raw_detection,
                "accepted_detection": accepted_detection,
                "predicted": int(predicted),
                "missing": int(missing),
                "confidence": round(confidence, 3),
                "track_length": track_length,
                "coverage_ratio": round(coverage_ratio, 3),
            }
        )

    def flush(self) -> None:
        if not self.rows:
            return
        exists = self.output_path.exists() and self.output_path.stat().st_size > 0
        with self.output_path.open("a" if exists else "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.FIELDS)
            if not exists:
                writer.writeheader()
            writer.writerows(self.rows)
        self.rows = []

    def summary(self) -> Dict:
        if not self.rows and self.output_path.exists():
            import csv as _csv

            with self.output_path.open("r", newline="", encoding="utf-8") as f:
                rows = list(_csv.DictReader(f))
        else:
            rows = self.rows
        if not rows:
            return {}
        accepted = sum(int(r["accepted_detection"]) for r in rows)
        predicted = sum(int(r["predicted"]) for r in rows)
        missing = sum(int(r["missing"]) for r in rows)
        coverage = sum(float(r["coverage_ratio"]) for r in rows) / len(rows)
        return {
            "frames": len(rows),
            "accepted_detections": accepted,
            "predicted_frames": predicted,
            "missing_frames": missing,
            "average_coverage_ratio": round(coverage, 3),
        }