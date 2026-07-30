"""
Debug script to inspect ByteTrack tracking pipeline.

Analyzes video frames to verify tracking behavior.
"""

import sys
sys.path.insert(0, 'd:/stepout')

import math
import cv2
import numpy as np
from pathlib import Path

from app.detection.detection_types import Detection
from app.tracking.player_tracker import PlayerTracker
from app.core.config import get_config


class TrackingDebugger:
    def __init__(self):
        self.config = get_config().raw
        self.player_tracker = PlayerTracker(config=self.config)
        
    def inspect_frames(self, video_path: str, target_frames: list = None):
        """Inspect frames for tracking errors."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Cannot open video {video_path}")
            return
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        for frame_num in range(total_frames):
            ret, frame = cap.read()
            if not ret:
                break
            
            if target_frames is not None and frame_num not in target_frames:
                continue
            
            frame_shape = (frame.shape[0], frame.shape[1])
            
            # Get detections
            detections = self._get_detections(frame, frame_num)
            
            # Run tracking
            tracked_dets = self.player_tracker.update(
                detections, frame_shape, frame_num, frame
            )
            
            print(f"Frame {frame_num}: {len(detections)} detections -> {len(tracked_dets)} tracks")
            
        cap.release()
    
    def _get_detections(self, frame, frame_num):
        """Get detections for a frame."""
        try:
            from app.detection.detector import YoloDetector
            detector = YoloDetector(config=self.config)
            detector.load()
            detections = detector.predict(frame)
            player_dets = [d for d in detections if getattr(d, 'cls_id', None) == 0]
            return player_dets
        except Exception as e:
            print(f"Detection failed for frame {frame_num}: {e}")
            return []


def main():
    debugger = TrackingDebugger()
    
    # Find a video file
    video_candidates = [
        "datasets/videos/match_soccernet.mp4",
        "datasets/videos/match.mp4",
        "uploads/match.mp4",
    ]
    
    video_path = None
    for candidate in video_candidates:
        if Path(candidate).exists():
            video_path = candidate
            break
    
    if video_path is None:
        for p in Path(".").rglob("*.mp4"):
            video_path = str(p)
            break
    
    if video_path is None:
        print("Error: No video file found")
        return
    
    print(f"Analyzing video: {video_path}")
    
    # Process all frames
    debugger.inspect_frames(video_path)
    
    print("Analysis complete.")


if __name__ == "__main__":
    main()