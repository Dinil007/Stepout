"""
YOLOv8 Detection Module - GPU Accelerated

Performs high-performance object detection (Players + Ball) using Ultralytics YOLOv8
accelerated on NVIDIA CUDA GPUs with FP16 Half Precision and PyTorch inference_mode.
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
print("GPU & PyTorch Diagnostics")
print("=" * 50)
print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {cuda_version}")
print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"GPU: {gpu_name}")

# ==========================================
# 2. Model Initialization & Optimizations
# ==========================================
# Load YOLO model once during startup
model = YOLO("yolov8m.pt")

# Move model parameters directly to CUDA device
model.to(device)

# Fuse Conv2d + BatchNorm2d layers for faster forward pass
try:
    model.fuse()
except Exception:
    pass

# Convert model weights to FP16 half-precision on CUDA for 2x tensor core speedup
if torch.cuda.is_available():
    model.model.half()

# Verify active parameter execution device
yolo_device = next(model.model.parameters()).device
print(f"YOLO Device: {yolo_device}")
print("=" * 50)

# ==========================================
# 3. Input & Output Paths
# ==========================================
INPUT_VIDEO = "D:/stepout/videos/raw/match30.mp4"
FALLBACK_VIDEO = "D:/stepout/videos/raw/match30.mp4"

source_video = INPUT_VIDEO if os.path.exists(INPUT_VIDEO) else FALLBACK_VIDEO

OUTPUT_DIR = "outputs/detection"
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_VIDEO = os.path.join(OUTPUT_DIR, "detected_video.mp4")

# ==========================================
# 4. Open Video Stream
# ==========================================
cap = cv2.VideoCapture(source_video)

if not cap.isOpened():
    print(f"Cannot open video source: {source_video}")
    exit(1)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

print("\n" + "=" * 50)
print("Video Information")
print("=" * 50)
print(f"Source     : {source_video}")
print(f"Resolution : {width} x {height}")
print(f"FPS        : {fps:.1f}")
print("=" * 50)

writer = cv2.VideoWriter(
    OUTPUT_VIDEO,
    cv2.VideoWriter_fourcc(*'mp4v'),
    fps,
    (width, height)
)

# Pitch ROI Polygon (excludes crowd & non-pitch regions)
pitch_polygon = np.array([
    [8, 347],
    [1218, 328],
    [1250, 529],
    [54, 610]
], dtype=np.int32)

MAX_FRAMES = 500
CONFIDENCE_THRESHOLD = 0.40
frame_count = 0

print("\nStarting CUDA-Accelerated Detection Pipeline...\n")

# ==========================================
# 5. Frame Processing Loop (with torch.inference_mode)
# ==========================================
# Disables autograd tracking completely, reducing GPU overhead and memory allocations
with torch.inference_mode():
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or frame_count >= MAX_FRAMES:
            break

        # Explicit GPU inference using model.predict with target device
        results = model.predict(
            source=frame,
            classes=[0, 32],      # 0: Person, 32: Sports Ball
            device=device,
            verbose=False
        )

        annotated = frame.copy()

        # Draw ROI Polygon overlay
        cv2.polylines(
            annotated,
            [pitch_polygon],
            True,
            (0, 255, 0),
            3
        )

        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes

            for box in boxes:
                conf = float(box.conf[0])
                if conf < CONFIDENCE_THRESHOLD:
                    continue

                cls = int(box.cls[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2

                # Filter out detections outside pitch ROI
                inside = cv2.pointPolygonTest(
                    pitch_polygon,
                    (center_x, center_y),
                    False
                )

                if inside < 0:
                    continue

                if cls == 0:
                    label = f"Player {conf:.2f}"
                    color = (0, 255, 0)
                else:
                    label = f"Ball {conf:.2f}"
                    color = (0, 0, 255)

                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                cv2.circle(annotated, (center_x, center_y), 4, (255, 0, 0), -1)
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

# ==========================================
# 6. Cleanup & Summary
# ==========================================
cap.release()
writer.release()

print("\n")
print("=" * 50)
print("GPU Detection Completed Successfully!")
print(f"Frames Processed : {frame_count}")
print(f"Execution Device : {yolo_device}")
print(f"Saved To         : {OUTPUT_VIDEO}")
print("=" * 50)