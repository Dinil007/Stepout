"""Per-frame detection diagnostics CSV writer."""

import csv
from pathlib import Path
from typing import Dict, List, Optional

from app.detection.detection_types import Detection


class DetectionMetricsWriter:
    FIELDS = [
        "frame_index",
        "visible_detections",
        "rejected_detections",
        "average_confidence",
        "average_bbox_height",
        "average_bbox_width",
        "inference_time_ms",
        "ball_detections",
        "preprocessing_steps",
    ]

    def __init__(self, output_path: Path) -> None:
        self.output_path = output_path
        self.rows: List[Dict] = []
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

    def record(
        self,
        frame_index: int,
        players: List[Detection],
        rejected: List[Detection],
        balls: List[Detection],
        inference_time_ms: float,
        preprocessing_steps: str = "",
    ) -> None:
        confs = [p.conf for p in players]
        heights = [p.height for p in players]
        widths = [p.width for p in players]
        self.rows.append(
            {
                "frame_index": frame_index,
                "visible_detections": len(players),
                "rejected_detections": len(rejected),
                "average_confidence": round(sum(confs) / len(confs), 4) if confs else 0.0,
                "average_bbox_height": round(sum(heights) / len(heights), 2) if heights else 0.0,
                "average_bbox_width": round(sum(widths) / len(widths), 2) if widths else 0.0,
                "inference_time_ms": round(inference_time_ms, 2),
                "ball_detections": len(balls),
                "preprocessing_steps": preprocessing_steps,
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
        visible = [int(r["visible_detections"]) for r in rows]
        return {
            "frames": len(rows),
            "average_players_detected": sum(visible) / len(visible),
            "maximum_players_detected": max(visible),
        }
