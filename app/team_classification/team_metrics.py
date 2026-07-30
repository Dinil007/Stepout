"""Team classification metrics collection."""

import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional


class TeamMetricsCollector:
    """Collects per-frame team classification metrics."""

    FIELDS = [
        "frame_index",
        "track_id",
        "predicted_team",
        "confidence",
        "crop_height",
        "feature_distance",
    ]

    def __init__(self, output_path: Path) -> None:
        self.output_path = output_path
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.rows: List[Dict] = []

    def record(
        self,
        frame_index: int,
        track_id: int,
        predicted_team: Optional[str],
        confidence: float,
        crop_height: int,
        feature_distance: float,
    ) -> None:
        self.rows.append(
            {
                "frame_index": frame_index,
                "track_id": track_id,
                "predicted_team": predicted_team or "Unknown",
                "confidence": round(confidence, 3),
                "crop_height": crop_height,
                "feature_distance": round(feature_distance, 3),
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

    def summary(self) -> Dict:
        total = len(self.rows)
        if total == 0:
            return {}
        team_a = sum(1 for r in self.rows if r["predicted_team"] == "Team A")
        team_b = sum(1 for r in self.rows if r["predicted_team"] == "Team B")
        unknown = sum(1 for r in self.rows if r["predicted_team"] == "Unknown")
        confs = [r["confidence"] for r in self.rows if r["confidence"] > 0]
        avg_conf = sum(confs) / len(confs) if confs else 0.0
        return {
            "total_records": total,
            "team_a": team_a,
            "team_b": team_b,
            "unknown": unknown,
            "avg_confidence": round(avg_conf, 3),
        }