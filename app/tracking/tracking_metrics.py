"""Tracking metrics collection for player tracking."""

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional


class TrackingMetricsCollector:
    """Collects and aggregates tracking quality metrics."""

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.track_presence: Dict[int, List[int]] = defaultdict(list)
        self.possible_switches: List[Dict] = []
        self.frame_rows: List[Dict] = []

    def record_frame(self, frame_no: int, tracked_dets: List) -> None:
        """Record tracking data for a single frame."""
        active_ids = [d.track_id for d in tracked_dets if d.track_id >= 0]
        
        for track_id in active_ids:
            self.track_presence[track_id].append(frame_no)
        
        self.frame_rows.append({
            "frame": frame_no,
            "players_tracked": len(active_ids),
        })

    def record_id_switch(self, switch_event: Dict) -> None:
        """Record a potential ID switch event."""
        self.possible_switches.append(switch_event)

    def get_metrics(self, current_frame: int) -> Dict:
        """Calculate tracking quality metrics."""
        lifetimes = []
        lost_tracks = 0
        recovered = 0
        
        for tid, frames in self.track_presence.items():
            if frames:
                lifetimes.append(len(frames))
                if frames[-1] < current_frame:
                    lost_tracks += 1
                    gaps = [b - a for a, b in zip(frames, frames[1:])]
                    recovered += sum(1 for g in gaps if g > 1)
        
        avg_lifetime = sum(lifetimes) / len(lifetimes) if lifetimes else 0.0
        max_lifetime = max(lifetimes) if lifetimes else 0
        
        return {
            "total_unique_tracks": len(self.track_presence),
            "active_tracks": sum(1 for frames in self.track_presence.values() if frames and frames[-1] == current_frame),
            "lost_tracks": lost_tracks,
            "recovered_tracks": recovered,
            "average_track_lifetime": round(avg_lifetime, 2),
            "longest_track_lifetime": max_lifetime,
            "fragmentation_index": round(recovered / max(len(self.track_presence), 1), 3),
            "estimated_id_switches": len(self.possible_switches),
            "possible_id_switch_events": self.possible_switches[:100],
        }

    def flush(self, filename: str = "tracking_quality.json") -> None:
        """Write accumulated metrics to file."""
        if not self.frame_rows:
            return
        
        last_frame = self.frame_rows[-1]["frame"]
        metrics = self.get_metrics(last_frame)
        
        output_path = self.output_dir / filename
        if output_path.exists():
            existing = json.loads(output_path.read_text(encoding="utf-8"))
            existing.update(metrics)
            metrics = existing
        
        output_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    def summary(self) -> Dict:
        """Return a summary of tracking metrics."""
        if not self.frame_rows:
            return {}
        
        last_frame = self.frame_rows[-1]["frame"]
        return self.get_metrics(last_frame)