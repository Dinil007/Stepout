"""
Tracking Validation Script

Runs a short validation clip through the pipeline with current ByteTrack parameters
and generates comparison metrics against the baseline.

This script does NOT modify any pipeline code.
"""

import os
import sys
import json
import time
import logging
from pathlib import Path
from typing import Dict, Any

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("TrackingValidation")

# Configuration
VALIDATION_VIDEO = Path("D:/stepout/videos/raw/match30.mp4")
OUTPUT_DIR = Path("outputs")
MAX_FRAMES = 500  # Short validation clip

# Load baseline (before) metrics from existing analytics
BASELINE_ANALYTICS = OUTPUT_DIR / "analytics.json"


def load_baseline() -> Dict[str, Any]:
    """Load baseline metrics from existing analytics.json."""
    if not BASELINE_ANALYTICS.exists():
        logger.warning(f"Baseline not found: {BASELINE_ANALYTICS}")
        return {}
    
    with open(BASELINE_ANALYTICS, "r") as f:
        data = json.load(f)
    
    # Extract key metrics
    summary = data.get("summary_metrics", {})
    player_stats = data.get("player_statistics", [])
    
    # Calculate aggregate metrics
    speeds = [p.get("max_speed_kmh", 0) for p in player_stats]
    distances = [p.get("total_distance_meters", 0) for p in player_stats]
    
    return {
        "total_players": summary.get("total_players_tracked", 0),
        "max_speed_kmh": max(speeds) if speeds else 0,
        "avg_speed_kmh": sum(p.get("avg_speed_kmh", 0) for p in player_stats) / len(player_stats) if player_stats else 0,
        "team_a_distance_m": summary.get("team_A_total_distance_m", 0),
        "team_b_distance_m": summary.get("team_B_total_distance_m", 0),
        "total_distance_m": sum(distances),
        "frames_processed": summary.get("processed_frames", 0),
    }


def run_validation():
    """Run validation with current tracker parameters."""
    logger.info("Starting tracking validation...")
    logger.info(f"Video: {VALIDATION_VIDEO}")
    logger.info(f"Max frames: {MAX_FRAMES}")
    
    # Import pipeline components
    from app.tracking.tracking import model, device, yolo_device, pitch_polygon
    import cv2
    import numpy as np
    import torch
from app.homography.homography_utils import compute_homography, transform_point
from app.homography.field_config import FIELD_LENGTH_METERS, FIELD_WIDTH_METERS
from app.analytics.speed_estimator import SpeedEstimator
from app.analytics.distance_tracker import DistanceTracker
from app.core.config import get_config
    
    # Homography setup (matched to run_match_analysis.py)
    PITCH_SRC_POINTS = np.array([
        [8,    347],
        [1218, 328],
        [1250, 529],
        [54,   610]
    ], dtype=np.float32)
    
    PITCH_DST_POINTS = np.array([
        [0.0,                     0.0                      ],
        [FIELD_LENGTH_METERS,      0.0                      ],
        [FIELD_LENGTH_METERS,      FIELD_WIDTH_METERS       ],
        [0.0,                      FIELD_WIDTH_METERS       ]
    ], dtype=np.float32)
    
    H, _ = compute_homography(PITCH_SRC_POINTS, PITCH_DST_POINTS)
    
    # Video capture
    cap = cv2.VideoCapture(str(VALIDATION_VIDEO))
    if not cap.isOpened():
        logger.error(f"Cannot open video: {VALIDATION_VIDEO}")
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    logger.info(f"Video FPS: {fps}, Total frames: {total_frames}")
    
    # Initialize analytics modules
    speed_estimator = SpeedEstimator(fps=fps)
    distance_tracker = DistanceTracker()
    
    # Tracking state
    track_positions: Dict[int, tuple] = {}  # track_id -> (x, y) in meters
    track_ages: Dict[int, int] = {}
    track_lost_frames: Dict[int, int] = {}
    active_tracks = set()
    
    # Metrics collection
    all_speeds = []
    all_distances = []
    new_track_count = 0
    lost_track_count = 0
    recovered_track_count = 0
    
    frame_count = 0
    start_time = time.perf_counter()
    
    with torch.inference_mode():
        while cap.isOpened() and frame_count < MAX_FRAMES:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Run tracking (same config as pipeline)
            results = model.track(
                source=frame,
                persist=True,
                tracker="app/tracking/bytetrack_custom.yaml",
                classes=[0, 32],
                conf=0.25,
                iou=0.5,
                imgsz=1280,
                device=device,
                verbose=False
            )
            
            current_frame_tracks = set()
            
            if len(results) > 0 and results[0].boxes is not None:
                boxes = results[0].boxes
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    
                    if cls != 0:  # Only players
                        continue
                    
                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2
                    inside = cv2.pointPolygonTest(pitch_polygon, (float(center_x), float(center_y)), False)
                    if inside < 0:
                        continue
                    
                    track_id = int(box.id[0]) if box.id is not None else -1
                    if track_id == -1:
                        continue
                    
                    current_frame_tracks.add(track_id)
                    
                    # Transform to pitch coordinates
                    pixel_pos = (center_x, center_y)
                    field_pos = transform_point(pixel_pos, H)
                    
                    # Track lifecycle
                    is_new = track_id not in active_tracks
                    if is_new:
                        track_ages[track_id] = 0
                        track_lost_frames[track_id] = 0
                        active_tracks.add(track_id)
                        new_track_count += 1
                    else:
                        track_ages[track_id] += 1
                        if track_id in track_positions and track_id not in active_tracks:
                            recovered_track_count += 1
                    
                    # Update speed and distance
                    if track_id in track_positions:
                        prev_pos = track_positions[track_id]
                        speed_data = speed_estimator.update(track_id, field_pos)
                        dist_m = distance_tracker.update(track_id, field_pos, speed_kmh=speed_data["speed_kmh"] if speed_data else 0.0)
                        
                        if speed_data:
                            all_speeds.append(speed_data["speed_kmh"])
                            all_distances.append(dist_m)
                    
                    track_positions[track_id] = field_pos
                    track_lost_frames[track_id] = 0
            
            # Mark lost tracks
            for tid in list(active_tracks):
                if tid not in current_frame_tracks:
                    track_lost_frames[tid] = track_lost_frames.get(tid, 0) + 1
                    if track_lost_frames[tid] == 1:
                        lost_track_count += 1
                    active_tracks.discard(tid)
            
            frame_count += 1
            if frame_count % 100 == 0:
                logger.info(f"Processed {frame_count}/{MAX_FRAMES} frames")
    
    cap.release()
    
    total_time = time.perf_counter() - start_time
    
    # Collect metrics
    max_speed = max(all_speeds) if all_speeds else 0
    avg_speed = sum(all_speeds) / len(all_speeds) if all_speeds else 0
    total_distance = sum(all_distances) if all_distances else 0
    
    # Get per-player summaries
    player_summaries = []
    for tid in track_positions.keys():
        summary = speed_estimator.get_summary(tid)
        summary["track_id"] = tid
        summary["total_distance_m"] = distance_tracker.get_distance(tid)
        player_summaries.append(summary)
    
    validation_results = {
        "frames_processed": frame_count,
        "processing_time_sec": round(total_time, 2),
        "fps": round(frame_count / total_time, 2) if total_time > 0 else 0,
        "new_tracks": new_track_count,
        "lost_tracks": lost_track_count,
        "recovered_tracks": recovered_track_count,
        "unique_tracks": len(track_positions),
        "max_speed_kmh": round(max_speed, 2),
        "avg_speed_kmh": round(avg_speed, 2),
        "total_distance_m": round(total_distance, 2),
        "player_summaries": player_summaries,
        "track_lifetimes": {str(k): v for k, v in track_ages.items()},
    }
    
    # Save results
    output_file = OUTPUT_DIR / "validation_results.json"
    with open(output_file, "w") as f:
        json.dump(validation_results, f, indent=2)
    
    logger.info(f"Validation complete. Results saved to {output_file}")
    logger.info(f"Max speed: {max_speed:.2f} km/h")
    logger.info(f"Avg speed: {avg_speed:.2f} km/h")
    logger.info(f"Total distance: {total_distance:.2f} m")
    logger.info(f"New tracks: {new_track_count}")
    logger.info(f"Lost tracks: {lost_track_count}")
    logger.info(f"Recovered tracks: {recovered_track_count}")
    
    return validation_results


def compare_with_baseline(baseline: Dict[str, Any], current: Dict[str, Any]) -> None:
    """Print comparison report."""
    print("\n" + "=" * 80)
    print("TRACKING VALIDATION COMPARISON REPORT")
    print("=" * 80)
    
    if not baseline:
        print("WARNING: No baseline data available for comparison")
        print("Current run metrics:")
        print(f"  Frames processed: {current.get('frames_processed', 0)}")
        print(f"  Max speed: {current.get('max_speed_kmh', 0):.2f} km/h")
        print(f"  Avg speed: {current.get('avg_speed_kmh', 0):.2f} km/h")
        print(f"  Total distance: {current.get('total_distance_m', 0):.2f} m")
        print(f"  New tracks: {current.get('new_tracks', 0)}")
        print(f"  Lost tracks: {current.get('lost_tracks', 0)}")
        print(f"  Recovered tracks: {current.get('recovered_tracks', 0)}")
        return
    
    print(f"\n{'Metric':<30} {'Before':>15} {'After':>15} {'Change':>15}")
    print("-" * 80)
    
    # Max speed
    before_speed = baseline.get("max_speed_kmh", 0)
    after_speed = current.get("max_speed_kmh", 0)
    speed_change = ((after_speed - before_speed) / before_speed * 100) if before_speed > 0 else 0
    print(f"{'Max Speed (km/h)':<30} {before_speed:>15.2f} {after_speed:>15.2f} {speed_change:>+14.1f}%")
    
    # Avg speed
    before_avg = baseline.get("avg_speed_kmh", 0)
    after_avg = current.get("avg_speed_kmh", 0)
    avg_change = ((after_avg - before_avg) / before_avg * 100) if before_avg > 0 else 0
    print(f"{'Avg Speed (km/h)':<30} {before_avg:>15.2f} {after_avg:>15.2f} {avg_change:>+14.1f}%")
    
    # Total distance
    before_dist = baseline.get("total_distance_m", 0)
    after_dist = current.get("total_distance_m", 0)
    dist_change = ((after_dist - before_dist) / before_dist * 100) if before_dist > 0 else 0
    print(f"{'Total Distance (m)':<30} {before_dist:>15.2f} {after_dist:>15.2f} {dist_change:>+14.1f}%")
    
    # Team distances
    before_team_a = baseline.get("team_a_distance_m", 0)
    after_team_a = current.get("team_a_distance_m", 0)  # Not in validation, use total
    print(f"{'Team A Distance (m)':<30} {before_team_a:>15.2f} {'N/A':>15} {'N/A':>15}")
    
    before_team_b = baseline.get("team_b_distance_m", 0)
    print(f"{'Team B Distance (m)':<30} {before_team_b:>15.2f} {'N/A':>15} {'N/A':>15}")
    
    # Track counts
    before_frames = baseline.get("frames_processed", 0)
    after_frames = current.get("frames_processed", 0)
    print(f"{'Frames Processed':<30} {before_frames:>15} {after_frames:>15} {'N/A':>15}")
    
    print(f"\n{'Current Run Only Metrics:':}")
    print(f"  New tracks: {current.get('new_tracks', 0)}")
    print(f"  Lost tracks: {current.get('lost_tracks', 0)}")
    print(f"  Recovered tracks: {current.get('recovered_tracks', 0)}")
    print(f"  Unique tracks: {current.get('unique_tracks', 0)}")
    print(f"  Processing time: {current.get('processing_time_sec', 0):.1f}s")
    print(f"  Effective FPS: {current.get('fps', 0):.2f}")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    # Load baseline
    baseline = load_baseline()
    
    # Run validation
    current = run_validation()
    
    # Compare
    compare_with_baseline(baseline, current)