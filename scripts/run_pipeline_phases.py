"""
Execute all validation phases for the football analytics pipeline.
"""
import os
import sys
import json
import cv2
import numpy as np
import time
from pathlib import Path

# Ensure project root in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from run_pipeline import FootballAnalyticsPipeline


def ensure_dirs():
    """Create required debug output directories."""
    Path("outputs/debug").mkdir(parents=True, exist_ok=True)


def phase1_verify_input():
    """PHASE 1 – VERIFY INPUT VIDEO."""
    print("\n===== PHASE 1: VERIFY INPUT VIDEO =====")
    pipeline = FootballAnalyticsPipeline(
        input_video_path="D:/stepout/videos/raw/match30.mp4",
        output_dir="outputs",
        max_frames=500
    )
    cap, fps, width, height, total_frames = pipeline._stage_preprocessing()

    print(f"filename     : match30.mp4")
    print(f"resolution   : {width}x{height}")
    print(f"FPS          : {fps}")
    print(f"total frames : {total_frames}")

    # Save first frame as debug image
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ret, frame = cap.read()
    if ret:
        cv2.imwrite("outputs/debug/input_first_frame.jpg", frame)
        print("Saved first frame to outputs/debug/input_first_frame.jpg")

    cap.release()
    return fps, width, height


def phase2_fix_detection():
    """PHASE 2 – FIX PLAYER DETECTION."""
    print("\n===== PHASE 2: FIX PLAYER DETECTION =====")
    from app.detection.yolo_detector import model, pitch_polygon, CONFIDENCE_THRESHOLD
    import torch

    input_path = "D:/stepout/videos/raw/match30.mp4"
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_path = "outputs/debug/player_detection.jpg"
    writer = cv2.VideoWriter(
        str(Path("outputs/debug/player_detection.mp4")),
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (width, height),
    )

    # Tuned thresholds for this match
    CONF = 0.25
    IOU = 0.45
    IMGSZ = 1280

    # Try to relax ROI a bit while keeping stands out.
    # We'll keep the existing pitch_polygon but reduce strictness via detection thresholds.
    processed = 0
    with torch.inference_mode():
        while cap.isOpened() and processed < 200:
            ret, frame = cap.read()
            if not ret:
                break
            processed += 1

            results = model.track(
                source=frame,
                persist=True,
                tracker="app/tracking/bytetrack_custom.yaml",
                classes=[0],  # person only for player detection view
                conf=CONF,
                iou=IOU,
                imgsz=IMGSZ,
                verbose=False,
            )

            annotated = frame.copy()
            cv2.polylines(annotated, [pitch_polygon], True, (0, 255, 0), 3)

            if len(results) > 0 and results[0].boxes is not None:
                boxes = results[0].boxes
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                    inside = cv2.pointPolygonTest(pitch_polygon, (cx, cy), False)
                    if inside < 0:
                        continue
                    conf = float(box.conf[0])
                    tid = int(box.id[0]) if box.id is not None else -1
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(annotated, f"{conf:.2f}", (x1, max(15, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            writer.write(annotated)
            print(f"Detection frames: {processed}", end="\r")

    writer.release()
    cap.release()

    # Save a representative frame
    cap2 = cv2.VideoCapture(str(Path("outputs/debug/player_detection.mp4")))
    ret, frame = cap2.read()
    if ret:
        cv2.imwrite(out_path, frame)
    cap2.release()
    print(f"\nSaved player detection output to {out_path}")


def phase3_ball_detection():
    """PHASE 3 – ADD BALL DETECTION."""
    print("\n===== PHASE 3: BALL DETECTION =====")
    from app.detection.yolo_detector import model, pitch_polygon, CONFIDENCE_THRESHOLD
    import torch

    input_path = "D:/stepout/videos/raw/match30.mp4"
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = cv2.VideoWriter(
        str(Path("outputs/debug/ball_detection.mp4")),
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (width, height),
    )

    CONF = 0.20
    processed = 0
    with torch.inference_mode():
        while cap.isOpened() and processed < 200:
            ret, frame = cap.read()
            if not ret:
                break
            processed += 1

            results = model.predict(
                source=frame,
                classes=[0, 32],
                conf=CONF,
                iou=0.45,
                imgsz=1280,
                device="cuda:0" if torch.cuda.is_available() else "cpu",
                verbose=False,
            )

            annotated = frame.copy()

            if len(results) > 0 and results[0].boxes is not None:
                for box in results[0].boxes:
                    cls = int(box.cls[0])
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    if cls == 0:
                        label = "PLAYER"
                        color = (0, 255, 0)
                    elif cls == 32:
                        label = "BALL"
                        color = (0, 255, 255)
                        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                        cv2.circle(annotated, (cx, cy), 5, color, -1)
                    else:
                        continue

                    cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(annotated, label, (x1, max(15, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            writer.write(annotated)
            print(f"Ball frames: {processed}", end="\r")

    writer.release()
    cap.release()

    cap2 = cv2.VideoCapture(str(Path("outputs/debug/ball_detection.mp4")))
    ret, frame = cap2.read()
    if ret:
        cv2.imwrite("outputs/debug/ball_detection.jpg", frame)
    cap2.release()
    print("\nSaved ball detection output to outputs/debug/ball_detection.jpg")


def phase5_fix_tracking():
    """PHASE 5 – FIX TRACKING."""
    print("\n===== PHASE 5: TRACKING VALIDATION =====")
    pipeline = FootballAnalyticsPipeline(
        input_video_path="D:/stepout/videos/raw/match30.mp4",
        output_dir="outputs",
        max_frames=200,
    )
    (
        all_mapped_players,
        all_raw_tracks,
        player_histories,
        player_telemetry,
    ) = pipeline._stage_computer_vision_and_tracking(
        cap=cv2.VideoCapture("D:/stepout/videos/raw/match30.mp4"),
        fps=25.0,
        width=1280,
        height=720,
        max_frames=200,
    )
    print(f"Tracked players: {len(player_telemetry)}")
    print("Tracking video saved to outputs/tracking.mp4")


def main():
    ensure_dirs()
    phase1_verify_input()
    phase2_fix_detection()
    phase3_ball_detection()
    phase5_fix_tracking()
    print("\nPhases 1-3 and 5 complete. Proceed to remaining phases.")


if __name__ == "__main__":
    main()