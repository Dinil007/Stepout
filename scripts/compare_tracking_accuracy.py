"""
Tracking Accuracy Comparison: ByteTrack vs ByteTrack+ReID

This script compares tracking performance between:
1. ByteTrack only (motion-based)
2. ByteTrack + ReID (motion + appearance)

Metrics:
- ID Switch Rate
- Track Fragmentation
- Recovery Rate
- Average Track Length
- Coverage Ratio
"""

import argparse
import csv
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np

from app.core.config import get_config
from app.tracking.player_tracker import PlayerTracker
from app.detection.detection_types import Detection

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TrackingComparison:
    """Compare ByteTrack vs ByteTrack+ReID performance."""

    def __init__(self, video_path: str, max_frames: int = 250):
        self.video_path = video_path
        self.max_frames = max_frames
        self.output_dir = Path("outputs/comparison")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def load_frames(self) -> Tuple[List[np.ndarray], int, int]:
        """Load video frames."""
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {self.video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        frames = []
        for i in range(min(self.max_frames, total)):
            ret, img = cap.read()
            if not ret:
                break
            frames.append(img)

        cap.release()
        logger.info(f"Loaded {len(frames)} frames ({width}x{height} @ {fps}fps)")
        return frames, fps, (height, width)

    def run_tracking(self, frames: List[np.ndarray], frame_shape: Tuple[int, int],
                     enable_reid: bool) -> Dict:
        """
        Run tracking with ReID enabled or disabled.

        Args:
            frames: Video frames
            frame_shape: (height, width)
            enable_reid: If True, enable ReID (requires torchreid)

        Returns:
            Tracking results and metrics
        """
        # Temporarily modify config
        cfg = get_config().raw
        if not enable_reid:
            cfg.setdefault("tracking", {}).setdefault("reid", {})["enabled"] = False

        tracker = PlayerTracker(config=cfg)

        results = []
        track_history = {}
        id_switches = 0
        recoveries = 0

        start_time = time.time()

        for frame_no, frame in enumerate(frames):
            # Create dummy detections (replace with your actual detector)
            # For demo, we'll use placeholder detections
            detections = []

            # TODO: Replace with actual YOLO detection
            # detections = yolo_model(frame, classes=[0])  # Class 0 = person

            # For testing, create dummy detections if none provided
            if not detections and frame_no == 0:
                # Create a dummy detection in the center
                h, w = frame.shape[:2]
                detections.append(
                    Detection(cls_id=0, conf=0.9, bbox=(w//2-50, h//2-50, w//2+50, h//2+50))
                )

            tracked = tracker.update(detections, frame_shape, frame_no, frame)

            # Track ID switches
            current_ids = set(t.d for t in tracked)
            if tracked:
                results.append({
                    "frame": frame_no,
                    "tracks": len(tracked),
                    "track_ids": list(current_ids),
                })

                for tid in current_ids:
                    if tid not in track_history:
                        # Check if this ID was recently lost (potential recovery)
                        for old_tid, last_seen in list(tracker.prev_lost.items()):
                            if frame_no - last_seen[0] < 10:
                                recoveries += 1
                                break
                    track_history[tid] = frame_no

        elapsed = time.time() - start_time

        # Get final metrics
        metrics = tracker.get_metrics(current_frame=len(frames) - 1)
        metrics["processing_fps"] = len(frames) / elapsed if elapsed > 0 else 0

        tracker.flush_metrics()
        tracker.write_tracking_report(
            video_resolution=f"{frame_shape[1]}x{frame_shape[0]}",
            fps=30.0,
            processing_fps=metrics["processing_fps"]
        )

        return {
            "mode": "ByteTrack+ReID" if enable_reid else "ByteTrack Only",
            "reid_enabled": enable_reid,
            "metrics": metrics,
            "frames_processed": len(frames),
            "processing_time_s": elapsed,
        }

    def compare(self):
        """Run comparison between ByteTrack and ByteTrack+ReID."""
        logger.info("Loading video frames...")
        frames, fps, frame_shape = self.load_frames()

        logger.info("Running ByteTrack only...")
        bt_only_results = self.run_tracking(frames, frame_shape, enable_reid=False)

        logger.info("Running ByteTrack+ReID...")
        bt_reid_results = self.run_tracking(frames, frame_shape, enable_reid=True)

        # Compare and display results
        self.display_comparison(bt_only_results, bt_reid_results)

        # Save to CSV
        self.save_comparison_csv(bt_only_results, bt_reid_results)

        return bt_only_results, bt_reid_results

    def display_comparison(self, bt_only: Dict, bt_reid: Dict):
        """Display side-by-side comparison."""
        print("\n" + "=" * 80)
        print("TRACKING ACCURACY COMPARISON")
        print("=" * 80)

        print(f"\n{'Metric':<40} {'ByteTrack':<20} {'ByteTrack+ReID':<20} {'Delta':<10}")
        print("-" * 80)

        metrics_bt = bt_only["metrics"]
        metrics_reid = bt_reid["metrics"]

        # Compare key metrics
        comparisons = [
            ("Total Unique Tracks", "total_unique_tracks", False),
            ("Active Tracks (avg)", "reid", True),
            ("Recovery Attempts", "reid.recovery_attempts", True),
            ("Successful Recoveries", "reid.successful_recoveries", True),
            ("Recovery Rate", "reid.recovery_rate", True),
            ("ID Switches Detected", "reid.id_switches_detected", True),
            ("Processing FPS", "processing_fps", False),
        ]

        for label, key, use_reid in comparisons:
            if use_reid:
                val_bt = self._get_nested(metrics_bt, key, "N/A")
                val_reid = self._get_nested(metrics_reid, key, "N/A")
            else:
                val_bt = metrics_bt.get(key, "N/A")
                val_reid = metrics_reid.get(key, "N/A")

            delta = self._compute_delta(val_bt, val_reid)
            print(f"{label:<40} {str(val_bt):<20} {str(val_reid):<20} {delta:<10}")

        print("\n" + "=" * 80)
        print("REID METRICS (When Enabled)")
        print("=" * 80)

        if "reid" in metrics_reid and "summary" in metrics_reid["reid"]:
            summary = metrics_reid["reid"]["summary"]
            print(f"  Average Appearance Similarity: {summary.get('avg_appearance_similarity', 'N/A')}")
            print(f"  Average Motion Score: {summary.get('avg_motion_score', 'N/A')}")
            print(f"  Average Final Score: {summary.get('avg_final_score', 'N/A')}")
            print(f"  ReID Debug Entries: {summary.get('reid_debug_entries', 'N/A')}")

        print("=" * 80)

    def save_comparison_csv(self, bt_only: Dict, bt_reid: Dict):
        """Save comparison results to CSV."""
        csv_path = self.output_dir / "tracking_comparison.csv"

        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Metric", "ByteTrack Only", "ByteTrack+ReID", "Delta"])

            # Write key metrics
            row_data = [
                ("Total Unique Tracks", bt_only["metrics"]["total_unique_tracks"],
                 bt_reid["metrics"]["total_unique_tracks"]),
                ("Processing FPS", bt_only["metrics"]["processing_fps"],
                 bt_reid["metrics"]["processing_fps"]),
                ("Frames Processed", bt_only["frames_processed"], bt_reid["frames_processed"]),
            ]

            # Add ReID-specific metrics if available
            if "reid" in bt_reid["metrics"]:
                reid_stats = bt_reid["metrics"]["reid"]
                row_data.extend([
                    ("Recovery Attempts", 0, reid_stats.get("recovery_attempts", 0)),
                    ("Successful Recoveries", 0, reid_stats.get("successful_recoveries", 0)),
                    ("Recovery Rate", 0.0, reid_stats.get("recovery_rate", 0.0)),
                    ("ID Switches Detected", 0, reid_stats.get("id_switches_detected", 0)),
                ])

            for metric, bt_val, reid_val in row_data:
                delta = self._compute_delta(bt_val, reid_val)
                writer.writerow([metric, bt_val, reid_val, delta])

        logger.info(f"Comparison saved to {csv_path}")

    def _get_nested(self, d: Dict, key: str, default=None):
        """Get nested dict value."""
        parts = key.split(".")
        val = d
        for part in parts:
            if isinstance(val, dict):
                val = val.get(part, default)
            else:
                return default
        return val

    def _compute_delta(self, val1, val2) -> str:
        """Compute delta between two values."""
        if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
            if val1 == 0:
                return f"+{val2:.2f}" if val2 > 0 else "0.00"
            delta = val2 - val1
            pct = (delta / val1) * 100 if val1 != 0 else 0
            return f"{delta:+.2f} ({pct:+.1f}%)"
        return "N/A"


def main():
    parser = argparse.ArgumentParser(description="Compare tracking accuracy")
    parser.add_argument("video", help="Path to video file")
    parser.add_argument("--max-frames", type=int, default=250, help="Max frames to process")
    args = parser.parse_args()

    comparison = TrackingComparison(args.video, args.max_frames)
    bt_only, bt_reid = comparison.compare()

    # Print report location
    print(f"\nDetailed reports saved to: outputs/comparison/")


if __name__ == "__main__":
    main()