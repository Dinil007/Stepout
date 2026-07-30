"""
Frame-level tracking inspection to identify exact failure mode.

Runs tracking on a subset of frames and records:
- Track ID continuity
- Position jumps
- Detection confidence
- Lost/recovered tracks

Outputs: outputs/tracking_frame_inspection.json
"""

import json
import time
import cv2
import numpy as np
from pathlib import Path

from ultralytics import YOLO
from app.homography.homography_utils import compute_homography, transform_point
from app.homography.field_config import FIELD_LENGTH_METERS, FIELD_WIDTH_METERS

print("=" * 70)
print("FRAME-LEVEL TRACKING INSPECTION")
print("=" * 70)

# Setup
OUTPUT_DIR = Path("outputs")
PITCH_SRC_POINTS = np.array([[8,347],[1218,328],[1250,529],[54,610]], dtype=np.float32)
PITCH_DST_POINTS = np.array([[0,0],[FIELD_LENGTH_METERS,0],[FIELD_LENGTH_METERS,FIELD_WIDTH_METERS],[0,FIELD_WIDTH_METERS]], dtype=np.float32)
H, _ = compute_homography(PITCH_SRC_POINTS, PITCH_DST_POINTS)

# Load model
model = YOLO("yolov8x.pt")
model.to("cpu")

# Video
video_path = Path("D:/stepout/videos/raw/match30.mp4")
cap = cv2.VideoCapture(str(video_path))
fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"Video: {video_path}")
print(f"FPS: {fps}, Total frames: {total_frames}")

# Inspect first 200 frames
frames_to_inspect = min(200, total_frames)
print(f"Inspecting first {frames_to_inspect} frames...")

track_history = {}  # track_id -> list of (frame, x, y, confidence)
frame_data = []

for frame_idx in range(frames_to_inspect):
    ret, frame = cap.read()
    if not ret:
        break
    
    results = model.track(
        source=frame,
        persist=True,
        tracker="app/tracking/bytetrack_custom.yaml",
        classes=[0],
        conf=0.25,
        iou=0.5,
        imgsz=1280,
        device="cpu",
        verbose=False
    )
    
    current_tracks = {}
    
    if len(results) > 0 and results[0].boxes is not None:
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            track_id = int(box.id[0]) if box.id is not None else -1
            
            if track_id == -1:
                continue
            
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            field_pos = transform_point((cx, cy), H)
            
            current_tracks[track_id] = {
                "frame": frame_idx + 1,
                "bbox": [x1, y1, x2, y2],
                "pixel_pos": [cx, cy],
                "field_pos": [round(field_pos[0], 2), round(field_pos[1], 2)],
                "confidence": round(conf, 3)
            }
            
            if track_id not in track_history:
                track_history[track_id] = []
            track_history[track_id].append({
                "frame": frame_idx + 1,
                "field_pos": [round(field_pos[0], 2), round(field_pos[1], 2)],
                "confidence": round(conf, 3)
            })
    
    frame_data.append({
        "frame": frame_idx + 1,
        "active_tracks": list(current_tracks.keys()),
        "track_count": len(current_tracks)
    })

cap.release()

# Analyze tracking patterns
print(f"\nTotal unique tracks: {len(track_history)}")

# Find high-speed events and classify them
high_speed_events = []
for track_id, history in track_history.items():
    if len(history) < 2:
        continue
    
    for i in range(1, len(history)):
        prev = history[i-1]
        curr = history[i]
        
        dx = curr["field_pos"][0] - prev["field_pos"][0]
        dy = curr["field_pos"][1] - prev["field_pos"][1]
        disp = float(np.sqrt(dx*dx + dy*dy))
        speed = disp / (1.0/fps) * 3.6
        
        if speed > 40:
            # Check if this is an ID switch or lost track
            frame_gap = curr["frame"] - prev["frame"]
            
            # Look at confidence drop
            conf_drop = prev["confidence"] - curr["confidence"]
            
            # Classification logic
            cause = "Unknown"
            if disp > 10:
                cause = "A. ID Switch / Large Position Jump"
            elif disp > 5:
                cause = "B. Lost Track / Recovery"
            elif conf_drop > 0.3:
                cause = "C. Confidence Drop / Detection Error"
            else:
                cause = "D. Speed Calculation (no artifact)"
            
            high_speed_events.append({
                "track_id": track_id,
                "frame": curr["frame"],
                "prev_frame": prev["frame"],
                "frame_gap": frame_gap,
                "displacement_m": round(disp, 2),
                "speed_kmh": round(speed, 2),
                "prev_pos": prev["field_pos"],
                "curr_pos": curr["field_pos"],
                "prev_conf": prev["confidence"],
                "curr_conf": curr["confidence"],
                "cause": cause
            })

# Sort by speed
high_speed_events.sort(key=lambda x: x["speed_kmh"], reverse=True)

print(f"\nHigh-speed events (>40 km/h): {len(high_speed_events)}")
print("\nTop 10 fastest events:")
for i, evt in enumerate(high_speed_events[:10], 1):
    print(f"{i}. Frame {evt['frame']} | Track {evt['track_id']} | "
          f"{evt['speed_kmh']:.1f} km/h | {evt['displacement_m']:.1f}m | {evt['cause']}")

# Classify root causes
causes = {}
for evt in high_speed_events:
    cause = evt["cause"]
    causes[cause] = causes.get(cause, 0) + 1

print("\nRoot cause classification:")
for cause, count in sorted(causes.items(), key=lambda x: x[1], reverse=True):
    pct = count / len(high_speed_events) * 100 if high_speed_events else 0
    print(f"  {cause}: {count} ({pct:.0f}%)")

# Save results
output_json = OUTPUT_DIR / "tracking_frame_inspection.json"
with open(output_json, "w") as f:
    json.dump({
        "summary": {
            "frames_inspected": frames_to_inspect,
            "total_unique_tracks": len(track_history),
            "high_speed_events": len(high_speed_events),
            "root_causes": causes
        },
        "top_events": high_speed_events[:20],
        "frame_data": frame_data[:50]  # Sample
    }, f, indent=2)

print(f"\n✓ Saved: {output_json}")
print("=" * 70)