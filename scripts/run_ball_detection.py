"""
Ball Detection + Tracking + Interpolation Pipeline

Complete end-to-end example showing:
1. BallDetector → YOLO inference for ball (class 32, high-res)
2. BallTracker → Kalman filter tracking with association
3. BallInterpolator → Post-processing gap filling

Usage:
    python scripts/run_ball_detection.py --video <path> --max-frames 500

Flow:
    Frame → BallDetector.detect_and_filter() → BallTracker.update() → Track Output
    Post-process: BallInterpolator.interpolate() → Smooth Trajectory
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.detection.ball_detector import BallDetector
from app.tracking.ball_tracker import BallTracker
from app.tracking.ball_interpolation import BallInterpolator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("ball_detection_pipeline")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ball Detection + Tracking + Interpolation Pipeline"
    )
    parser.add_argument(
        "--video",
        type=str,
        default="D:/stepout/videos/raw/match30.mp4",
        help="Path to input video file",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=500,
        help="Maximum number of frames to process",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/ball_detection",
        help="Directory for output files",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8x.pt",
        help="YOLO model path",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.10,
        help="Ball detection confidence threshold",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=960,
        help="Inference image size (higher = better for small balls)",
    )
    parser.add_argument(
        "--max-match-dist",
        type=float,
        default=180.0,
        help="Max association distance in pixels",
    )
    parser.add_argument(
        "--max-missing",
        type=int,
        default=45,
        help="Max missing frames before track is lost",
    )
    parser.add_argument(
        "--interpolate-max-gap",
        type=int,
        default=20,
        help="Max gap frames to interpolate",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Show visualization window",
    )
    parser.add_argument(
        "--save-video",
        action="store_true",
        help="Save annotated output video",
    )
    return parser.parse_args()


def load_pitch_roi() -> Optional[np.ndarray]:
    """
    Load pitch ROI polygon from config or use default.
    
    Returns:
        numpy array of polygon points or None
    """
    # Default pitch ROI for match30.mp4 (1280x720)
    roi = np.array([
        [12, 520],
        [1827, 492],
        [1875, 793],
        [81, 915],
    ], dtype=np.int32)
    
    # Try to load from config file
    config_path = Path("configs/pitch_roi.json")
    if config_path.exists():
        try:
            import json
            with open(config_path, "r") as f:
                data = json.load(f)
            if "roi_polygon" in data:
                roi = np.array(data["roi_polygon"], dtype=np.int32)
                logger.info(f"Loaded pitch ROI from {config_path}")
        except Exception as e:
            logger.warning(f"Failed to load pitch ROI config: {e}")
    
    return roi


def draw_ball_info(
    frame: np.ndarray,
    track_result: Optional[Dict],
    best_det: Optional[object],
    filtered_dets: List,
    frame_num: int,
) -> np.ndarray:
    """Draw ball detection and tracking info on frame."""
    annotated = frame.copy()
    h, w = frame.shape[:2]

    # Draw info text background
    cv2.rectangle(annotated, (0, 0), (w, 60), (0, 0, 0), -1)

    # Frame counter
    cv2.putText(
        annotated,
        f"Frame: {frame_num}",
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
    )

    # Draw all filtered detections (green circles)
    for det in filtered_dets:
        cx, cy = det.center
        cv2.circle(annotated, (cx, cy), 6, (0, 255, 0), 2)
        cv2.putText(
            annotated,
            f"{det.conf:.2f}",
            (cx + 8, cy - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 255, 0),
            1,
        )

    # Draw best detection (red circle)
    if best_det is not None:
        cx, cy = best_det.center
        cv2.circle(annotated, (cx, cy), 8, (0, 0, 255), -1)
        cv2.putText(
            annotated,
            f"Best: {best_det.conf:.3f}",
            (cx + 10, cy + 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            2,
        )

    # Draw track result
    if track_result is not None:
        center = track_result["center"]
        cx, cy = int(center[0]), int(center[1])
        is_predicted = track_result["is_predicted"]

        # Tracked ball (blue if confirmed, yellow if predicted)
        color = (255, 255, 0) if is_predicted else (255, 0, 0)
        cv2.circle(annotated, (cx, cy), 10, color, 2)
        cv2.circle(annotated, (cx, cy), 3, color, -1)

        # Bounding box
        bbox = track_result["bbox"]
        if bbox and len(bbox) == 4:
            cv2.rectangle(annotated, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)

        # Track info
        status = "PREDICTED" if is_predicted else "TRACKED"
        conf_text = f"{track_result['confidence']:.3f}" if not is_predicted else "0.0"
        cv2.putText(
            annotated,
            f"Ball {status} | conf={conf_text} | streak={track_result['longest_streak']}",
            (10, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
        )

        # Draw trajectory history
        history = track_result.get("image_history", [])
        if len(history) > 1:
            for i in range(1, len(history)):
                pt1 = (int(history[i - 1][0]), int(history[i - 1][1]))
                pt2 = (int(history[i][0]), int(history[i][1]))
                alpha = i / len(history)
                trail_color = (0, int(255 * alpha), int(255 * (1 - alpha)))
                cv2.line(annotated, pt1, pt2, trail_color, 2)
    else:
        cv2.putText(
            annotated,
            "Ball not tracked",
            (10, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2,
        )

    return annotated


def main() -> None:
    """Run ball detection + tracking + interpolation pipeline."""
    args = parse_args()

    # Validate video path
    video_path = args.video
    if not os.path.exists(video_path):
        logger.error(f"Video not found: {video_path}")
        return

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ==========================================
    # 1. Initialize Ball Detector
    # ==========================================
    logger.info("=" * 60)
    logger.info("STEP 1: Initializing Ball Detector")
    logger.info("=" * 60)
    logger.info(f"  Model: {args.model}")
    logger.info(f"  Confidence: {args.conf}")
    logger.info(f"  Image Size: {args.imgsz}")
    logger.info(f"  Max Match Dist: {args.max_match_dist}")

    ball_detector = BallDetector(
        model_path=args.model,
        conf=args.conf,
        imgsz=args.imgsz,
    )
    ball_detector.load()

    # Set pitch ROI
    pitch_roi = load_pitch_roi()
    if pitch_roi is not None:
        ball_detector.set_pitch_roi(pitch_roi)
        logger.info(f"  Pitch ROI set: {pitch_roi.tolist()}")

    # ==========================================
    # 2. Initialize Ball Tracker
    # ==========================================
    logger.info("=" * 60)
    logger.info("STEP 2: Initializing Ball Tracker")
    logger.info("=" * 60)
    logger.info(f"  Max Missing Frames: {args.max_missing}")
    logger.info(f"  Max Match Distance: {args.max_match_dist}")

    ball_tracker = BallTracker()
    ball_tracker.max_missing_frames = args.max_missing
    ball_tracker.max_match_dist = args.max_match_dist

    # ==========================================
    # 3. Initialize Ball Interpolator
    # ==========================================
    logger.info("=" * 60)
    logger.info("STEP 3: Initializing Ball Interpolator")
    logger.info("=" * 60)
    logger.info(f"  Max Gap: {args.interpolate_max_gap}")

    ball_interpolator = BallInterpolator(
        max_gap=args.interpolate_max_gap,
        method="linear",
    )

    # ==========================================
    # 4. Open Video and Process
    # ==========================================
    logger.info("=" * 60)
    logger.info("STEP 4: Processing Video")
    logger.info("=" * 60)
    logger.info(f"  Video: {video_path}")
    logger.info(f"  Max Frames: {args.max_frames}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Failed to open video: {video_path}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    logger.info(f"  Resolution: {width}x{height}")
    logger.info(f"  FPS: {fps:.1f}")
    logger.info(f"  Total Frames: {total_frames}")

    # Video writer for annotated output
    writer = None
    if args.save_video:
        output_video_path = str(output_dir / "ball_detection_output.mp4")
        writer = cv2.VideoWriter(
            output_video_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )
        logger.info(f"  Output Video: {output_video_path}")

    # ==========================================
    # 5. Frame Processing Loop
    # ==========================================
    frame_count = 0
    track_history: List[Dict] = []
    detection_times: List[float] = []
    tracking_times: List[float] = []

    logger.info("")
    logger.info("Processing frames...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or frame_count >= args.max_frames:
            break

        frame_count += 1

        # ---- Step A: Ball Detection ----
        # Get predicted center from tracker for scoring proximity
        predicted_center = None
        if ball_tracker.is_active() and ball_tracker._track is not None:
            predicted_center = ball_tracker._track.predicted_center

        best_det, filtered_dets, inference_ms = ball_detector.detect_and_filter(
            frame, predicted_center
        )

        detection_times.append(inference_ms)

        # ---- Step B: Ball Tracking ----
        # Convert detections to dict format for BallTracker
        if best_det is not None:
            detection_list = [ball_detector.detection_to_dict(best_det)]
        else:
            detection_list = []

        track_result = ball_tracker.update(detection_list, frame_count)
        tracking_times.append(inference_ms)

        # Store track result for interpolation
        if track_result is not None:
            track_history.append(track_result)

        # ---- Step C: Visualization ----
        if args.visualize or args.save_video:
            annotated = draw_ball_info(
                frame, track_result, best_det, filtered_dets, frame_count
            )

            if args.visualize:
                cv2.imshow("Ball Detection & Tracking", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            if writer is not None:
                writer.write(annotated)

        # Progress
        if frame_count % 50 == 0:
            logger.info(
                f"  Frame {frame_count}/{min(args.max_frames, total_frames)} | "
                f"Det: {inference_ms:.1f}ms | "
                f"Track: {'active' if track_result else 'lost'}"
            )

    cap.release()
    if writer is not None:
        writer.release()
    if args.visualize:
        cv2.destroyAllWindows()

    logger.info(f"  Processed {frame_count} frames")

    # ==========================================
    # 6. Post-Processing: Interpolation
    # ==========================================
    logger.info("")
    logger.info("=" * 60)
    logger.info("STEP 5: Ball Trajectory Interpolation")
    logger.info("=" * 60)
    logger.info(f"  Track History Entries: {len(track_history)}")
    logger.info(f"  Total Frames: {frame_count}")

    interpolated_trajectory = ball_interpolator.interpolate(
        track_history, frame_count
    )

    logger.info(f"  Interpolated Trajectory: {len(interpolated_trajectory)} frames")

    # ==========================================
    # 7. Results & Metrics
    # ==========================================
    logger.info("")
    logger.info("=" * 60)
    logger.info("RESULTS & METRICS")
    logger.info("=" * 60)

    # Detection metrics
    det_metrics = ball_detector.get_metrics()
    logger.info("Ball Detector Metrics:")
    logger.info(f"  Total Raw Detections: {det_metrics['total_raw_detections']}")
    logger.info(f"  Filtered Detections: {det_metrics['filtered_detections']}")
    logger.info(f"  Accepted Detections: {det_metrics['accepted_detections']}")
    logger.info(f"  Avg Inference Time: {np.mean(detection_times):.1f}ms")

    # Tracking metrics
    track_metrics = ball_tracker.get_metrics()
    logger.info("")
    logger.info("Ball Tracker Metrics:")
    logger.info(f"  Raw Detections: {track_metrics['raw_detections']}")
    logger.info(f"  Accepted Detections: {track_metrics['accepted_detections']}")
    logger.info(f"  Predicted Frames: {track_metrics['predicted_frames']}")
    logger.info(f"  Missing Frames: {track_metrics['missing_frames']}")
    logger.info(f"  Coverage Ratio: {track_metrics['coverage_ratio']*100:.1f}%")
    logger.info(f"  Longest Streak: {track_metrics['longest_continuous_track']} frames")

    # Interpolation metrics
    interp_stats = ball_interpolator.get_stats()
    logger.info("")
    logger.info("Ball Interpolator Metrics:")
    logger.info(f"  Detected Frames: {interp_stats.get('detected_frames', 0)}")
    logger.info(f"  Predicted Frames: {interp_stats.get('predicted_frames', 0)}")
    logger.info(f"  Interpolated Frames: {interp_stats.get('interpolated_frames', 0)}")
    logger.info(f"  Missing Frames: {interp_stats.get('missing_frames', 0)}")
    logger.info(f"  Coverage: {interp_stats.get('coverage_pct', 0)}%")
    logger.info(f"  Longest Gap: {interp_stats.get('longest_missing_gap', 0)} frames")

    # Save trajectory to CSV
    csv_path = output_dir / "ball_trajectory.csv"
    if interpolated_trajectory:
        import csv
        with open(csv_path, "w", newline="") as f:
            writer_csv = csv.writer(f)
            writer_csv.writerow([
                "frame", "center_x", "center_y",
                "is_interpolated", "was_predicted", "was_detected", "confidence"
            ])
            for entry in interpolated_trajectory:
                writer_csv.writerow([
                    entry["frame"],
                    entry["center_x"],
                    entry["center_y"],
                    entry.get("is_interpolated", False),
                    entry.get("was_predicted", False),
                    entry.get("was_detected", False),
                    entry.get("confidence", 0.0),
                ])
        logger.info(f"")
        logger.info(f"Trajectory saved to: {csv_path}")

    # Save metrics to JSON
    import json
    metrics = {
        "detector": det_metrics,
        "tracker": track_metrics,
        "interpolator": interp_stats,
        "pipeline": {
            "total_frames": frame_count,
            "avg_detection_time_ms": round(float(np.mean(detection_times)), 2),
            "video_path": video_path,
            "model": args.model,
            "confidence_threshold": args.conf,
            "image_size": args.imgsz,
            "max_match_distance": args.max_match_dist,
            "max_missing_frames": args.max_missing,
            "interpolation_max_gap": args.interpolate_max_gap,
        },
    }
    metrics_path = output_dir / "ball_detection_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Metrics saved to: {metrics_path}")

    logger.info("")
    logger.info("=" * 60)
    logger.info("BALL DETECTION PIPELINE COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()