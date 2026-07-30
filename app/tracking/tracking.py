"""
ByteTrack Player & Ball Tracking Module - GPU Accelerated

Combines Ultralytics YOLOv8 object detection with ByteTrack multi-object tracking,
accelerated on NVIDIA CUDA GPUs with FP16 half-precision and inference_mode optimizations.
"""

import os
import cv2
import numpy as np
import torch
from ultralytics import YOLO

# ==========================================
# 1. Device Setup & Diagnostics
# ==========================================
device = "cuda:0" if torch.cuda.is_available() else "cpu"
cuda_version = torch.version.cuda if torch.cuda.is_available() else "N/A"
gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None (CPU)"

print("=" * 50)
print("GPU & PyTorch Diagnostics (Tracking Pipeline)")
print("=" * 50)
print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {cuda_version}")
print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"GPU: {gpu_name}")

# ==========================================
# 2. Model Initialization & Optimizations
# ==========================================
# Load YOLO model once during application startup
model = YOLO("yolov8x.pt")

# Move model to CUDA GPU device
model.to(device)

# Fuse Conv2d + BatchNorm2d layers
try:
    model.fuse()
except Exception:
    pass

# Enable FP16 half precision when CUDA is available
if torch.cuda.is_available():
    model.model.half()

# Verify active parameter device
yolo_device = next(model.model.parameters()).device
print(f"YOLO Device: {yolo_device}")
print("=" * 50)

# ==========================================
# 3. Paths & Video Stream Setup
# ==========================================
INPUT_VIDEO = "outputs/preprocessed/preprocessed_video.mp4"
FALLBACK_VIDEO = "videos/input.mp4"
source_video = INPUT_VIDEO if os.path.exists(INPUT_VIDEO) else FALLBACK_VIDEO

OUTPUT_DIR = "outputs/tracking"
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_VIDEO = os.path.join(OUTPUT_DIR, "tracked_video.mp4")

cap = cv2.VideoCapture(source_video)
if not cap.isOpened():
    print(f"Cannot open video source: {source_video}")
    exit(1)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

writer = cv2.VideoWriter(
    OUTPUT_VIDEO,
    cv2.VideoWriter_fourcc(*'mp4v'),
    fps,
    (width, height)
)

pitch_polygon = np.array([
    [8, 347],
    [1218, 328],
    [1250, 529],
    [54, 610]
], dtype=np.int32)

MAX_FRAMES = 1000
frame_count = 0

print("\nStarting CUDA-Accelerated ByteTrack Tracking...\n")

# ==========================================
# 4. Tracking Loop with GPU Acceleration
# ==========================================
with torch.inference_mode():
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or frame_count >= MAX_FRAMES:
            break

        # Explicit GPU tracking execution
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

        annotated = frame.copy()

        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes

            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls = int(box.cls[0])

                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2

                inside = cv2.pointPolygonTest(
                    pitch_polygon,
                    (center_x, center_y),
                    False
                )

                if inside < 0:
                    continue

                track_id = int(box.id[0]) if box.id is not None else -1

                if cls == 0:
                    label = f"Player ID:{track_id}"
                    color = (0, 255, 0)
                else:
                    label = "Ball"
                    color = (0, 0, 255)

                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    annotated,
                    label,
                    (x1, max(15, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2
                )

        writer.write(annotated)
        frame_count += 1
        print(f"Processed Frame : {frame_count}/{MAX_FRAMES} (Device: {yolo_device})", end="\r")

cap.release()
writer.release()

print("\n")
print("=" * 50)
print("GPU Tracking Completed Successfully")
print(f"Frames Processed : {frame_count}")
print(f"Execution Device : {yolo_device}")
print(f"Saved To         : {OUTPUT_VIDEO}")
print("=" * 50)