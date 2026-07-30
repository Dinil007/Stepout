"""
Update all video path references across the project to use the new video.
"""
import re
from pathlib import Path

NEW_VIDEO = "videos/raw/match30.mp4"
NEW_VIDEO_UNIX = "videos/raw/match30.mp4"  # Already unix-style

# Files that need updating with their replacement rules
files_to_update = {
    "run_pipeline.py": [
        ('input_video_path: str = "videos/input.mp4"',
         f'input_video_path: str = "{NEW_VIDEO_UNIX}"'),
        ('input_video_path="videos/input.mp4"',
         f'input_video_path="{NEW_VIDEO_UNIX}"'),
        ('fallback_video = Path("outputs/preprocessed/preprocessed_video.mp4")',
         '# Fallback not needed - using direct path'),
    ],
    "app/core/config.py": [
        ('"videos/sample_video.mp4"',
         f'"{NEW_VIDEO_UNIX}"'),
    ],
    "app/detection/yolo_detector.py": [
        ('INPUT_VIDEO = "outputs/preprocessed/preprocessed_video.mp4"',
         f'INPUT_VIDEO = "{NEW_VIDEO_UNIX}"'),
        ('FALLBACK_VIDEO = "videos/input.mp4"',
         f'FALLBACK_VIDEO = "{NEW_VIDEO_UNIX}"'),
    ],
    "app/tracking/tracking.py": [
        ('INPUT_VIDEO = "outputs/preprocessed/preprocessed_video.mp4"',
         f'INPUT_VIDEO = "{NEW_VIDEO_UNIX}"'),
        ('FALLBACK_VIDEO = "videos/input.mp4"',
         f'FALLBACK_VIDEO = "{NEW_VIDEO_UNIX}"'),
    ],
    "app/preprocessing/preprocess.py": [
        ('cv2.VideoCapture("videos/raw/football_match.mp4")',
         f'cv2.VideoCapture("{NEW_VIDEO_UNIX}")'),
    ],
    "app/preprocessing/preprocess_video.py": [
        ('input_video = "videos/raw/football_match.mp4"',
         f'input_video = "{NEW_VIDEO_UNIX}"'),
    ],
    "app/preprocessing/video_reader.py": [
        ('cv2.VideoCapture("videos/raw/football_match.mp4")',
         f'cv2.VideoCapture("{NEW_VIDEO_UNIX}")'),
    ],
    "app/team_classification/team_pipeline.py": [
        ('input_video_path: str = "outputs/preprocessed/preprocessed_video.mp4"',
         f'input_video_path: str = "{NEW_VIDEO_UNIX}"'),
    ],
    "scripts/run_match_analysis.py": [
        ('FALLBACK_VIDEO = "outputs/preprocessed/preprocessed_video.mp4"',
         f'FALLBACK_VIDEO = "{NEW_VIDEO_UNIX}"'),
    ],
    "scripts/run_pose_analysis.py": [
        ('video_source = "videos/input.mp4"',
         f'video_source = "{NEW_VIDEO_UNIX}"'),
        ('video_source = "outputs/preprocessed/preprocessed_video.mp4"',
         f'video_source = "{NEW_VIDEO_UNIX}"'),
    ],
    "inspect_tracking_frames.py": [
        ('Path("videos/raw/Second Half.mp4")',
         f'Path("{NEW_VIDEO_UNIX}")'),
    ],
    "homography_calibration_audit.py": [
        ('FRAME_SAMPLE = Path("uploads/Second Half.mp4")',
         f'FRAME_SAMPLE = Path("{NEW_VIDEO_UNIX}")'),
    ],
    "app/utils/select_pitch_roi.py": [
        ('video = "outputs/preprocessed/preprocessed_video.mp4"',
         f'video = "{NEW_VIDEO_UNIX}"'),
    ],
    "app/api/dependencies.py": [],  # Check if video paths exist here
}

updates_applied = []

for fpath, rules in files_to_update.items():
    p = Path(fpath)
    if not p.exists():
        print(f"SKIP: {fpath} not found")
        continue
    
    content = p.read_text(encoding='utf-8')
    original = content
    
    for old_str, new_str in rules:
        if old_str in content:
            content = content.replace(old_str, new_str)
            updates_applied.append(f"{fpath}: {old_str[:60]} -> {new_str[:60]}")
    
    if content != original:
        p.write_text(content, encoding='utf-8')
        print(f"UPDATED: {fpath}")

print(f"\nTotal updates applied: {len(updates_applied)}")
for u in updates_applied:
    print(f"  {u}")
</parameter>
<task_progress>
- [x] Find all old video references
- [x] Update run_pipeline.py
- [ ] Update app/core/config.py
- [ ] Update app/detection/yolo_detector.py
- [ ] Update app/tracking/tracking.py
- [ ] Update app/preprocessing/*.py
- [ ] Update app/team_classification/team_pipeline.py
- [ ] Update scripts/*.py
- [ ] Update other files with old references
- [ ] Clear cached outputs
- [ ] Run pipeline with new video
    - [ ] Verify all outputs come from match30.mp4
</task_progress>
</write_to_file>