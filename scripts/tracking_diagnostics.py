"""
Tracking diagnostics for ByteTrack identity switch investigation.

Examines:
1. Exact ByteTrack output format
2. Column mapping
3. How track IDs are passed through the pipeline
4. Whether dataset generation preserves correct track IDs
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import math
import cv2
import numpy as np

from app.core.config import get_config
from app.detection.detector import YoloDetector
from app.detection.detection_types import Detection
from app.tracking.player_tracker import PlayerTracker
from app.tracking.bytetrack import BYTETracker
from app.dataset.dataset_builder import DatasetBuilder, DatasetBuildConfig


def inspect_bytetrack_output(video_path: str, max_frames: int = 10):
    """Inspect the exact format returned by BYTETracker.update()."""
    config = get_config().raw
    tracking_cfg = config.get("tracking", {})
    tracker_config_path = tracking_cfg.get("tracker_config", "app/tracking/bytetrack_custom.yaml")

    tracker = BYTETracker(tracker_config_path, frame_rate=25)
    detector = YoloDetector(config=config)
    detector.load()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Cannot open video: {video_path}")
        return

    print("=" * 80)
    print("STEP 1: ByteTrack output format inspection")
    print("=" * 80)

    frame_count = 0
    while frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        frame_shape = (frame.shape[0], frame.shape[1])
        detections = detector.predict(frame)
        player_dets = [d for d in detections if getattr(d, "cls_id", None) == 0]

        # Build dets_array exactly like PlayerTracker does
        dets_array = []
        for det in player_dets:
            x1, y1, x2, y2 = det.bbox
            dets_array.append([x1, y1, x2, y2, det.conf])

        tracks = tracker.update(
            dets_array,
            [(frame_shape[0], frame_shape[1])],
            ((frame_shape[0], frame_shape[1]),),
        )

        print(f"\nFrame {frame_count}:")
        print(f"  Input detections: {len(dets_array)}")
        print(f"  Output tracks shape: {tracks.shape}")
        print(f"  Output dtype: {tracks.dtype}")

        if len(tracks) > 0:
            print(f"  First track row (full): {tracks[0]}")
            print(f"  Columns unpacked in PlayerTracker: x1, y1, x2, y2, conf, track_id = track")
            print("  ")
            print("  ByteTrack standard output format (from ultralytics docs):")
            print("    Column 0: x1 (left)")
            print("    Column 1: y1 (top)")
            print("    Column 2: x2 (right)")
            print("    Column 3: y2 (bottom)")
            print("    Column 4: track_id (integer)")
            print("    Column 5: conf (confidence)")
            print("    Column 6: cls (class)")
            print("    Columns 7-8: may be present depending on tracker version")

            # Verify what we actually got
            print("\n  Actual column verification from this output:")
            for col_idx in range(min(9, tracks.shape[1])):
                col_data = tracks[:, col_idx]
                print(f"    Column {col_idx}: values={col_data[:3] if len(col_data) > 0 else 'N/A'} dtype={col_data.dtype if hasattr(col_data, 'dtype') else 'N/A'}")

        frame_count += 1

    cap.release()


def inspect_player_tracker_mapping(video_path: str, max_frames: int = 10):
    """Inspect how PlayerTracker maps ByteTrack output to Detection objects."""
    print("\n" + "=" * 80)
    print("STEP 2: PlayerTracker update() mapping inspection")
    print("=" * 80)

    config = get_config().raw
    player_tracker = PlayerTracker(config=config)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Cannot open video: {video_path}")
        return

    frame_count = 0
    while frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        frame_shape = (frame.shape[0], frame.shape[1])

        # Use YOLO predict like pipeline does
        detections = player_tracker.tracker if False else None  # placeholder

        # Reuse detector for ground truth detections
        from app.detection.detector import YoloDetector
        detector = YoloDetector(config=config)
        detector.load()
        raw_dets = detector.predict(frame)
        player_dets = [d for d in raw_dets if getattr(d, "cls_id", None) == 0]

        tracked = player_tracker.update(player_dets, frame_shape, frame_count, frame)

        print(f"\nFrame {frame_count}:")
        print(f"  Input detections: {len(player_dets)}")
        print(f"  Output tracked: {len(tracked)}")

        if len(tracked) > 0:
            first = tracked[0]
            print(f"  First Detection object:")
            print(f"    track_id={first.track_id}")
            print(f"    cls_id={first.cls_id}")
            print(f"    conf={first.conf}")
            print(f"    bbox={first.bbox}")
            print(f"    center={first.center}")

        frame_count += 1

    cap.release()


def inspect_dataset_generation(video_path: str, max_frames: int = 10):
    """Inspect dataset generation to verify track_id preservation."""
    print("\n" + "=" * 80)
    print("STEP 3: Dataset generation track_id preservation")
    print("=" * 80)

    config = get_config().raw
    ds_cfg = DatasetBuildConfig(
        enabled=True,
        save_every_n_frames=1,
        crop_size=256,
        padding=True,
        output_dir="datasets/person_classifier_diag",
    )
    builder = DatasetBuilder(config=ds_cfg)
    builder.setup()

    # Use YOLO native tracking like generate_person_dataset.py does
    detector = YoloDetector(config=config)
    detector.load()
    detector.classes = [0]

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Cannot open video: {video_path}")
        return

    frame_count = 0
    while frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        frame_number = frame_count + 1
        timestamp = frame_count / 25.0

        try:
            detections = detector.track(frame, persist=True)
        except Exception:
            detections = []

        player_dets = [d for d in detections if getattr(d, "cls_id", None) == 0]

        player_tracks = {}
        for trk in player_dets:
            raw_tid = getattr(trk, "track_id", -1)
            tid = int(raw_tid) if raw_tid is not None else -1
            if tid < 0:
                continue
            x1, y1, x2, y2 = [int(v) for v in getattr(trk, "bbox", (0, 0, 0, 0))]
            conf = float(getattr(trk, "conf", getattr(trk, "confidence", 0.0)))
            player_tracks[tid] = {
                "bbox": [x1, y1, x2, y2],
                "confidence": conf,
            }

        builder.process_frame(frame, frame_number, timestamp, player_tracks)

        print(f"\nFrame {frame_number}:")
        print(f"  YOLO player tracks: {player_tracks}")
        print(f"  DatasetBuilder track_records keys: {sorted(builder.track_records.keys())}")

        frame_count += 1

    cap.release()

    print("\nDataset summary:")
    summaries = builder.get_summaries()
    for s in summaries:
        print(f"  Track {s.track_id:04d}: frames={s.total_frames}, avg_conf={s.average_confidence:.3f}")


def main():
    video_path = sys.argv[1] if len(sys.argv) > 1 else None

    if video_path is None:
        candidates = [
            "D:/stepout/videos/raw/match30.mp4",
            "datasets/videos/match_soccernet.mp4",
            "datasets/videos/match.mp4",
            "uploads/match.mp4",
        ]
        for c in candidates:
            if Path(c).exists():
                video_path = c
                break

    if video_path is None:
        for p in Path(".").rglob("*.mp4"):
            video_path = str(p)
            break

    if video_path is None:
        print("No video file found")
        return

    print(f"Using video: {video_path}")

    inspect_bytetrack_output(video_path, max_frames=3)
    inspect_player_tracker_mapping(video_path, max_frames=3)
    inspect_dataset_generation(video_path, max_frames=5)


if __name__ == "__main__":
    main()