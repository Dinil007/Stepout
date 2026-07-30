"""
Debug script to trace exactly what happens in the pipeline's tracking stage.
"""
import cv2
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent))

from run_pipeline import FootballAnalyticsPipeline

# Create pipeline
pipeline = FootballAnalyticsPipeline(
    input_video_path="D:/stepout/videos/raw/match30.mp4",
    output_dir="outputs",
    max_frames=50  # Just 50 frames to test
)

print("=" * 60)
print("TESTING PIPELINE TRACKING STAGE")
print("=" * 60)

# Run preprocessing to get video properties
cap, fps, width, height, total_frames = pipeline._stage_preprocessing()

print(f"\nVideo properties:")
print(f"  Resolution: {width}x{height}")
print(f"  FPS: {fps}")
print(f"  Total frames: {total_frames}")

# Now test the tracking stage directly
print(f"\nStarting tracking stage...")
try:
    result = pipeline._stage_computer_vision_and_tracking(
        cap=cap,
        fps=fps,
        width=width,
        height=height,
        max_frames=50
    )
    print(f"\nTracking stage completed successfully")
    print(f"Result: {len(result)} items returned")
except Exception as e:
    print(f"\nERROR in tracking stage: {e}")
    import traceback
    traceback.print_exc()
    cap.release()
    exit(1)

# Check if output files were created and have content
print(f"\n" + "=" * 60)
print(f"CHECKING OUTPUT FILES")
print(f"=" * 60)

output_files = [
    "outputs/detection.mp4",
    "outputs/tracking.mp4",
    "outputs/team_classification.mp4",
    "outputs/pitch_view.mp4"
]

for filepath in output_files:
    path = Path(filepath)
    if path.exists():
        size = path.stat().st_size
        print(f"{filepath}: {size} bytes")
    else:
        print(f"{filepath}: NOT FOUND")

print(f"=" * 60)