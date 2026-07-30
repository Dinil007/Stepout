"""
Debug script to identify VideoWriter failure.
Tests the exact same parameters as run_pipeline.py
"""
import cv2
import numpy as np
from pathlib import Path

# Test parameters from config
input_video_path = "D:/stepout/videos/raw/match30.mp4"
output_dir = Path("outputs")
output_path = output_dir / "test_output.mp4"

# Ensure output directory exists
output_dir.mkdir(parents=True, exist_ok=True)

# Open input video
cap = cv2.VideoCapture(input_video_path)
if not cap.isOpened():
    print(f"ERROR: Cannot open input video: {input_video_path}")
    exit(1)

# Get video properties
fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"=" * 60)
print(f"INPUT VIDEO PROPERTIES")
print(f"=" * 60)
print(f"Path: {input_video_path}")
print(f"Resolution: {width}x{height}")
print(f"FPS: {fps}")
print(f"Total Frames: {total_frames}")
print(f"=" * 60)

# Test VideoWriter initialization
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
print(f"\nAttempting to create VideoWriter...")
print(f"Output path: {output_path}")
print(f"FourCC: mp4v (0x7634706D)")
print(f"FPS: {fps}")
print(f"Frame size: ({width}, {height})")

writer = cv2.VideoWriter(
    str(output_path),
    fourcc,
    fps,
    (width, height)
)

print(f"\nVideoWriter.isOpened(): {writer.isOpened()}")

if not writer.isOpened():
    print("\nERROR: VideoWriter FAILED to open!")
    print("Possible causes:")
    print("  1. Codec 'mp4v' not supported")
    print("  2. Invalid output path")
    print("  3. Insufficient disk space")
    print("  4. Permission denied")
    exit(1)

print("\nVideoWriter opened successfully. Testing frame write...")

# Read and write first 10 frames
frames_read = 0
frames_written = 0
errors = []

for i in range(10):
    ret, frame = cap.read()
    if not ret:
        print(f"Failed to read frame {i}")
        break
    
    frames_read += 1
    
    try:
        writer.write(frame)
        frames_written += 1
        print(f"Frame {i}: Written successfully")
    except Exception as e:
        error_msg = f"Frame {i}: Write failed - {e}"
        errors.append(error_msg)
        print(error_msg)

cap.release()
writer.release()

print(f"\n" + "=" * 60)
print(f"RESULTS")
print(f"=" * 60)
print(f"Frames read: {frames_read}")
print(f"Frames written: {frames_written}")
print(f"Errors: {len(errors)}")

if errors:
    print("\nError details:")
    for err in errors:
        print(f"  {err}")

# Check output file size
if output_path.exists():
    size = output_path.stat().st_size
    print(f"\nOutput file size: {size} bytes")
    if size < 1000:
        print("WARNING: Output file is too small (< 1KB). Likely empty or corrupt.")
else:
    print("\nERROR: Output file was not created!")

print(f"=" * 60)