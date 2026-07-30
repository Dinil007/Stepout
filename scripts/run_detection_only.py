"""Detection-only validation pipeline."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import os
import time
import json
from typing import List, Dict, Tuple

import cv2
import numpy as np

from app.core.config import get_config
from app.detection.detector import YoloDetector
from app.detection.detection_types import Detection
from app.preprocessing.adaptive_preprocessor import AdaptivePreprocessor


def draw_detections(frame: np.ndarray, detections: List[Detection]) -> np.ndarray:
    annotated = frame.copy()
    for det in detections:
        if det.cls_id == 0:  # player
            x1, y1, x2, y2 = det.bbox
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(annotated, f"P {det.conf:.2f}", (x1, max(0, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        elif det.cls_id == 32:  # ball
            x1, y1, x2, y2 = det.bbox
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.putText(annotated, f"B {det.conf:.2f}", (x1, max(0, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
    return annotated


def main():
    config = get_config().raw
    video_path = Path(config.get("video", {}).get("input_path", "videos/raw/match30.mp4"))
    max_frames = 750
    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_video = output_dir / "detected_video.mp4"
    output_report = output_dir / "detection_report.txt"

    # Initialize modules
    preprocessor = AdaptivePreprocessor()
    detector = YoloDetector(config=config)
    detector.load()

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = cv2.VideoWriter(str(output_video), cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    frame_count = 0
    detections_per_frame: List[int] = []
    confidences: List[float] = []
    players_per_frame: List[int] = []
    balls_per_frame: List[int] = []
    false_positives = 0

    while cap.isOpened():
        if max_frames and frame_count >= max_frames:
            break
        ret, frame = cap.read()
        if not ret:
            break

        # Preprocess
        try:
            m = preprocessor.measure(frame, frame_count)
            processed, _ = preprocessor.apply(frame, m)
        except Exception:
            processed = frame

        # Detect
        try:
            dets: List[Detection] = detector.predict(processed)
        except Exception as e:
            print(f"Detection error at frame {frame_count}: {e}")
            dets = []

        players = [d for d in dets if d.cls_id == 0]
        balls = [d for d in dets if d.cls_id == 32]
        detections_per_frame.append(len(dets))
        players_per_frame.append(len(players))
        balls_per_frame.append(len(balls))
        confidences.extend([float(getattr(d, "conf", 0.0)) or 0.0 for d in dets])

        # Simple heuristic false positives: high detections with low avg confidence
        avg_conf = np.mean([float(getattr(d, "conf", 0.0)) or 0.0 for d in dets]) if dets else 0.0
        if len(dets) > 22 and avg_conf < 0.4:
            false_positives += max(0, len(dets) - 22)

        annotated = draw_detections(processed, dets)
        writer.write(annotated)
        frame_count += 1

    cap.release()
    writer.release()

    # Report
    total_detections = sum(detections_per_frame)
    avg_detections = float(np.mean(detections_per_frame)) if detections_per_frame else 0.0
    avg_conf = float(np.mean(confidences)) if confidences else 0.0
    min_conf = float(np.min(confidences)) if confidences else 0.0
    max_conf = float(np.max(confidences)) if confidences else 0.0

    with open(output_report, "w") as f:
        f.write("Detection Report\n")
        f.write("=" * 40 + "\n")
        f.write(f"Video: {video_path}\n")
        f.write(f"Frames processed: {frame_count}\n")
        f.write(f"Total detections: {total_detections}\n")
        f.write(f"Average detections per frame: {avg_detections:.2f}\n")
        f.write(f"Players detected per frame: {players_per_frame}\n")
        f.write(f"Balls detected per frame: {balls_per_frame}\n")
        f.write(f"False positives (heuristic): {false_positives}\n")
        f.write(f"Confidence statistics:\n")
        f.write(f"  min: {min_conf:.4f}\n")
        f.write(f"  avg: {avg_conf:.4f}\n")
        f.write(f"  max: {max_conf:.4f}\n")

    print(f"Saved video: {output_video}")
    print(f"Saved report: {output_report}")
    print(f"Frames: {frame_count}")
    print(f"Total detections: {total_detections}")
    print(f"Avg detections/frame: {avg_detections:.2f}")
    print(f"False positives: {false_positives}")


if __name__ == "__main__":
    main()