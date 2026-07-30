"""Analyze tracking dataset and generate tracking_validation.md."""
from pathlib import Path
import json

dataset_dir = Path("datasets/person_classifier/raw")
track_folders = sorted([d for d in dataset_dir.iterdir() if d.is_dir()])

print(f"Total track folders: {len(track_folders)}")

# Check track_0002 specifically
track2 = dataset_dir / "track_0002"
if track2.exists():
    images = sorted(track2.glob("*.jpg"))
    print(f"track_0002: {len(images)} images")
    if images:
        print(f"  First: {images[0].name}")
        print(f"  Last: {images[-1].name}")

# Check each folder
for tf in track_folders:
    images = sorted(tf.glob("*.jpg"))
    if len(images) == 0:
        print(f"WARNING: {tf.name} is empty!")
    frame_nums = []
    for img in images:
        try:
            fn = int(img.stem.split("_")[1])
            frame_nums.append(fn)
        except:
            pass
    if frame_nums:
        min_f = min(frame_nums)
        max_f = max(frame_nums)
        unique = len(set(frame_nums))
        print(f"{tf.name}: {len(images)} crops, frames {min_f}-{max_f}, unique frames: {unique}")

# Read debug report
debug_path = Path("datasets/person_classifier/debug_report.txt")
if debug_path.exists():
    print()
    print("=== DEBUG REPORT ===")
    print(debug_path.read_text())

# Read metadata for track summary
metadata_path = Path("datasets/person_classifier/metadata/track_summary.csv")
if metadata_path.exists():
    print()
    print("=== TRACK SUMMARY ===")
    print(metadata_path.read_text())