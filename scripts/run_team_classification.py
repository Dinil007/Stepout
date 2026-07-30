"""
run_team_classification.py
--------------------------
End-to-end Team Classification Pipeline runner for the StepOut project.

Integrates:
  - YOLOv8 person detection (yolov8x.pt)
  - ByteTrack multi-object tracking
  - Pitch ROI filtering
  - Jersey ColorExtractor  (upper 50% HSV + KMeans dominant color)
  - TeamClassifier         (KMeans, warmup + majority-vote fallback)
  - TeamVisualizer         (coloured bounding boxes + labels)

Output:
  outputs/team_classification/team_classification.mp4
  outputs/team_classification/team_classification_report.txt

Usage:
    python scripts/run_team_classification.py
    python scripts/run_team_classification.py --video videos/raw/match30.mp4 --max-frames 750
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from app.team_classification.color_extractor import ColorExtractor
from app.team_classification.team_classifier import TeamClassifier
from app.team_classification.visualize_teams import TeamVisualizer
from ultralytics import YOLO


# ---------------------------------------------------------------------------
# Pitch ROI  (for match30.mp4 @ 1920x1080)
# Will be auto-scaled if the video resolution differs.
# ---------------------------------------------------------------------------
DEFAULT_PITCH_POLYGON_1920 = np.array([
    [20,  380],
    [1900, 380],
    [1900, 1020],
    [20,  1020],
], dtype=np.int32)


def build_pitch_polygon(frame_w: int, frame_h: int) -> np.ndarray:
    """Return a pitch ROI polygon scaled to the actual frame resolution."""
    sx = frame_w / 1920
    sy = frame_h / 1080
    poly = DEFAULT_PITCH_POLYGON_1920.astype(float)
    poly[:, 0] *= sx
    poly[:, 1] *= sy
    return poly.astype(np.int32)


def is_inside_pitch(pitch_polygon: np.ndarray, bbox: tuple) -> bool:
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    return cv2.pointPolygonTest(pitch_polygon, (float(cx), float(cy)), False) >= 0


# ---------------------------------------------------------------------------
# Overlay helpers
# ---------------------------------------------------------------------------
def draw_hud(
    frame: np.ndarray,
    frame_count: int,
    max_frames: int,
    warmup: int,
    is_trained: bool,
    team_a_count: int,
    team_b_count: int,
    unknown_count: int,
    fps: float,
) -> np.ndarray:
    """Draw a semi-transparent HUD with pipeline status & live counts."""
    h, w = frame.shape[:2]
    overlay = frame.copy()

    # Dark top-bar
    cv2.rectangle(overlay, (0, 0), (w, 55), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)

    phase = "WARMUP - collecting jersey colors..." if frame_count <= warmup else "LIVE CLASSIFICATION"
    phase_col = (0, 200, 255) if frame_count <= warmup else (0, 255, 100)

    cv2.putText(frame, f"StepOut | Team Classifier", (14, 22),
                cv2.FONT_HERSHEY_DUPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, f"Frame {frame_count}/{max_frames}  |  {fps:.1f} fps  |  {phase}",
                (14, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.52, phase_col, 1, cv2.LINE_AA)

    # Bottom-left team count panel
    panel_x, panel_y = 14, h - 80
    cv2.rectangle(frame, (panel_x - 4, panel_y - 22), (panel_x + 290, h - 10), (20, 20, 20), -1)
    cv2.rectangle(frame, (panel_x - 4, panel_y - 22), (panel_x + 290, h - 10), (60, 60, 60),  1)
    cv2.putText(frame, f"Team A: {team_a_count:2d}", (panel_x + 6, panel_y + 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 80, 80), 2, cv2.LINE_AA)
    cv2.putText(frame, f"Team B: {team_b_count:2d}", (panel_x + 6, panel_y + 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.58, (80, 80, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, f"Unknown: {unknown_count:2d}", (panel_x + 6, panel_y + 54),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 220, 220), 1, cv2.LINE_AA)

    # Progress bar
    pct = frame_count / max(max_frames, 1)
    bar_x1, bar_y1 = 0, h - 6
    bar_x2 = int(w * pct)
    cv2.rectangle(frame, (bar_x1, bar_y1), (w, h), (30, 30, 30), -1)
    cv2.rectangle(frame, (bar_x1, bar_y1), (bar_x2, h), (0, 200, 100), -1)

    return frame


def draw_warmup_overlay(frame: np.ndarray, frame_count: int, warmup: int) -> np.ndarray:
    """Show a warm-up progress bar in the top-right corner."""
    h, w = frame.shape[:2]
    pct = frame_count / warmup
    bx, by, bw, bh = w - 220, 10, 200, 16
    cv2.rectangle(frame, (bx - 2, by - 2), (bx + bw + 2, by + bh + 2), (40, 40, 40), -1)
    cv2.rectangle(frame, (bx, by), (bx + int(bw * pct), by + bh), (0, 180, 255), -1)
    cv2.putText(frame, f"Warmup {int(pct * 100)}%", (bx, by + bh + 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 180, 255), 1, cv2.LINE_AA)
    return frame


# ---------------------------------------------------------------------------
# Main pipeline runner
# ---------------------------------------------------------------------------
def run(
    video_path: str,
    output_path: str,
    model_weights: str,
    tracker_config: str,
    warmup_frames: int,
    max_frames: int,
    conf_thresh: float,
    iou_thresh: float,
    imgsz: int,
) -> None:

    video_path = str(ROOT_DIR / video_path) if not os.path.isabs(video_path) else video_path
    output_path = str(ROOT_DIR / output_path) if not os.path.isabs(output_path) else output_path
    model_weights = str(ROOT_DIR / model_weights) if not os.path.isabs(model_weights) else model_weights
    tracker_config = str(ROOT_DIR / tracker_config) if not os.path.isabs(tracker_config) else tracker_config

    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Input video not found: {video_path}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # ---- Open video ----
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    frame_w  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    src_fps  = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_src= int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    max_frames = min(max_frames, total_src) if max_frames > 0 else total_src

    print(f"\n{'='*60}")
    print(f"  StepOut — Team Classification Pipeline")
    print(f"{'='*60}")
    print(f"  Input      : {video_path}")
    print(f"  Output     : {output_path}")
    print(f"  Resolution : {frame_w}x{frame_h}  |  FPS: {src_fps:.2f}")
    print(f"  Frames     : {max_frames}  |  Warmup: {warmup_frames}")
    print(f"  Model      : {model_weights}")
    print(f"{'='*60}\n")

    # ---- Pitch polygon (auto-scaled) ----
    pitch_polygon = build_pitch_polygon(frame_w, frame_h)

    # ---- Components ----
    color_extractor  = ColorExtractor(jersey_ratio=0.5)
    team_classifier  = TeamClassifier(history_len=30)
    team_visualizer  = TeamVisualizer()

    # ---- Video writer ----
    writer = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        src_fps,
        (frame_w, frame_h),
    )
    if not writer.isOpened():
        raise IOError(f"Cannot open VideoWriter for: {output_path}")

    # ---- Load YOLO ----
    print(f"[INFO] Loading YOLO model: {model_weights} ...")
    model = YOLO(model_weights)
    print("[INFO] YOLO model loaded.\n")

    # ---- State ----
    collected_colors: list   = []
    track_color_samples: dict = {}
    is_trained = False

    frame_count  = 0
    t_start      = time.time()
    proc_times   = []

    # ---- Stats ----
    stats_team_a  = 0
    stats_team_b  = 0
    stats_unknown = 0
    frame_stats: list = []

    # ---- Main loop ----
    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame_count >= max_frames:
                break

            frame_count += 1
            t0 = time.time()

            # ── Detection + Tracking ──────────────────────────────────────
            results = model.track(
                source=frame,
                persist=True,
                tracker=tracker_config,
                classes=[0],
                conf=conf_thresh,
                iou=iou_thresh,
                imgsz=imgsz,
                verbose=False,
            )

            player_detections: list[tuple] = []

            if results and results[0].boxes is not None:
                boxes = results[0].boxes
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    bbox = (x1, y1, x2, y2)
                    if not is_inside_pitch(pitch_polygon, bbox):
                        continue
                    track_id = int(box.id[0]) if box.id is not None else -1
                    player_detections.append((bbox, track_id))

            # ── Phase 1 : Warmup color collection ────────────────────────
            if frame_count <= warmup_frames:
                for bbox, tid in player_detections:
                    if tid == -1:
                        continue
                    color = color_extractor.get_player_color(frame, bbox)
                    if color is not None:
                        collected_colors.append(color)
                        track_color_samples.setdefault(tid, []).append(color)

                # Train at end of warmup
                if frame_count == warmup_frames:
                    if len(collected_colors) >= 2:
                        print(f"\n[INFO] Training TeamClassifier on {len(collected_colors)} color samples ...")
                        team_classifier.fit(collected_colors)
                        is_trained = True
                        for tid, cols in track_color_samples.items():
                            if cols:
                                avg_c = np.mean(cols, axis=0)
                                team_classifier.assign_player(tid, avg_c)
                        print("[INFO] TeamClassifier ready. Starting live classification.\n")
                    else:
                        print("[WARNING] Not enough color samples collected. Extending warmup by 50 frames.")
                        warmup_frames += 50

            # ── Phase 2 : Live classification + visualisation ─────────────
            annotated = frame.copy()
            fa_count = fb_count = fu_count = 0

            for bbox, tid in player_detections:
                team_name = "Unknown"

                if is_trained and tid != -1:
                    if tid in team_classifier.player_teams and team_classifier.player_teams[tid] is not None:
                        team_label = team_classifier.player_teams[tid]
                    else:
                        color = color_extractor.get_player_color(frame, bbox)
                        team_label = team_classifier.assign_player(tid, color)
                    team_name = team_classifier.get_team_name(team_label)

                if team_name == "Team A":
                    fa_count += 1
                elif team_name == "Team B":
                    fb_count += 1
                else:
                    fu_count += 1

                annotated = team_visualizer.draw_player(annotated, bbox, tid, team_name)

            # During warmup, also draw detections with "Warmup" label
            if not is_trained:
                for bbox, tid in player_detections:
                    x1, y1, x2, y2 = bbox
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 180, 255), 2)
                    cv2.putText(annotated, f"ID {tid}", (x1 + 4, y1 - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 180, 255), 1, cv2.LINE_AA)
                annotated = draw_warmup_overlay(annotated, frame_count, warmup_frames)

            # Accumulate stats
            stats_team_a  += fa_count
            stats_team_b  += fb_count
            stats_unknown += fu_count
            frame_stats.append({
                "frame": frame_count,
                "team_a": fa_count,
                "team_b": fb_count,
                "unknown": fu_count,
            })

            # Compute instantaneous FPS
            t1 = time.time()
            dt = t1 - t0
            proc_times.append(dt)
            inst_fps = 1.0 / max(dt, 1e-9)

            # Draw HUD
            annotated = draw_hud(
                annotated, frame_count, max_frames, warmup_frames,
                is_trained, fa_count, fb_count, fu_count, inst_fps
            )

            writer.write(annotated)

            # Console progress
            elapsed = time.time() - t_start
            avg_fps = frame_count / max(elapsed, 1e-9)
            eta = (max_frames - frame_count) / max(avg_fps, 1e-9)
            print(
                f"\rFrame {frame_count:4d}/{max_frames} | "
                f"fps={avg_fps:.1f} | "
                f"A={fa_count} B={fb_count} ?={fu_count} | "
                f"ETA {eta:.0f}s   ",
                end="", flush=True
            )

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user.")
    finally:
        cap.release()
        writer.release()

    # ---- Summary ----
    total_time = time.time() - t_start
    total_detections = stats_team_a + stats_team_b + stats_unknown
    classification_rate = ((stats_team_a + stats_team_b) / max(total_detections, 1)) * 100

    print(f"\n\n{'='*60}")
    print(f"  TEAM CLASSIFICATION PIPELINE COMPLETE")
    print(f"{'='*60}")
    print(f"  Frames Processed   : {frame_count}")
    print(f"  Total Time         : {total_time:.1f}s  ({frame_count / max(total_time, 1):.1f} fps avg)")
    print(f"  Team A detections  : {stats_team_a}")
    print(f"  Team B detections  : {stats_team_b}")
    print(f"  Unknown detections : {stats_unknown}")
    print(f"  Classification Rate: {classification_rate:.1f}%")
    print(f"  Output Video       : {output_path}")
    print(f"{'='*60}\n")

    # ---- Write text report ----
    report_path = str(Path(output_path).parent / "team_classification_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("StepOut — Team Classification Pipeline Report\n")
        f.write("=" * 60 + "\n")
        f.write(f"Input Video         : {video_path}\n")
        f.write(f"Output Video        : {output_path}\n")
        f.write(f"Frames Processed    : {frame_count}\n")
        f.write(f"Total Processing    : {total_time:.1f}s\n")
        f.write(f"Average FPS         : {frame_count / max(total_time, 1):.2f}\n")
        f.write(f"Warmup Frames       : {warmup_frames}\n")
        f.write(f"Conf Threshold      : {conf_thresh}\n")
        f.write(f"IOU Threshold       : {iou_thresh}\n")
        f.write(f"Image Size          : {imgsz}\n")
        f.write("-" * 60 + "\n")
        f.write(f"Team A Detections   : {stats_team_a}\n")
        f.write(f"Team B Detections   : {stats_team_b}\n")
        f.write(f"Unknown Detections  : {stats_unknown}\n")
        f.write(f"Classification Rate : {classification_rate:.1f}%\n")
        f.write("-" * 60 + "\n")
        f.write("Per-Frame Stats (frame,team_a,team_b,unknown):\n")
        for row in frame_stats:
            f.write(f"  {row['frame']:4d}, {row['team_a']:2d}, {row['team_b']:2d}, {row['unknown']:2d}\n")

    print(f"[INFO] Report saved to: {report_path}")


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="StepOut Team Classification Pipeline")
    p.add_argument("--video",        default="videos/raw/match30.mp4",
                   help="Path to input video (relative to project root)")
    p.add_argument("--output",       default="outputs/team_classification/team_classification.mp4",
                   help="Path to output annotated video")
    p.add_argument("--weights",      default="yolov8x.pt",
                   help="YOLO model weights file")
    p.add_argument("--tracker",      default="app/tracking/bytetrack_custom.yaml",
                   help="ByteTrack YAML config")
    p.add_argument("--warmup",       type=int,   default=100,
                   help="Number of warmup frames for color collection")
    p.add_argument("--max-frames",   type=int,   default=0,
                   help="Max frames to process (0 = all)")
    p.add_argument("--conf",         type=float, default=0.35,
                   help="Detection confidence threshold")
    p.add_argument("--iou",          type=float, default=0.50,
                   help="NMS IOU threshold")
    p.add_argument("--imgsz",        type=int,   default=1280,
                   help="YOLO inference image size")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(
        video_path     = args.video,
        output_path    = args.output,
        model_weights  = args.weights,
        tracker_config = args.tracker,
        warmup_frames  = args.warmup,
        max_frames     = args.max_frames,
        conf_thresh    = args.conf,
        iou_thresh     = args.iou,
        imgsz          = args.imgsz,
    )
