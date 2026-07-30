"""ByteTrack wrapper for player tracking."""
from typing import List, Tuple
import argparse
import yaml
from pathlib import Path
import numpy as np
import inspect

try:
    from ultralytics.trackers.byte_tracker import BYTETracker as _BYTETracker
except Exception:
    try:
        from ultralytics.trackers.byte_tracker import BYTETracker as _BYTETracker
    except Exception:
        _BYTETracker = None


class ByteTrackWrapper:
    """Thin wrapper so imports don't break."""
    def __init__(self, yaml_path: str, frame_rate: int = 25):
        self.yaml_path = yaml_path
        self.frame_rate = frame_rate
        self.tracker = None
        
        if _BYTETracker is None:
            raise ImportError("BYTETracker could not be imported from ultralytics")
        
        # Load YAML config
        yaml_file = Path(yaml_path)
        if not yaml_file.exists():
            raise FileNotFoundError(f"ByteTrack config not found: {yaml_path}")
        
        with open(yaml_file, 'r') as f:
            config = yaml.safe_load(f)
        
        # Build args Namespace with required attributes
        args = argparse.Namespace(
            track_high_thresh=config.get('track_high_thresh', 0.6),
            track_low_thresh=config.get('track_low_thresh', 0.1),
            new_track_thresh=config.get('new_track_thresh', 0.7),
            match_thresh=config.get('match_thresh', 0.9),
            track_buffer=config.get('track_buffer', 30),
            fuse_score=config.get('fuse_score', True),
            min_track_frames=config.get('min_track_frames', 1),
            frame_rate=frame_rate
        )
        
        # Remove frame_rate if BYTETracker doesn't accept it
        try:
            sig = inspect.signature(_BYTETracker.__init__)
            if 'frame_rate' not in sig.parameters:
                delattr(args, 'frame_rate')
        except Exception:
            pass
        
        self.tracker = _BYTETracker(args)

    def update(self, detections: List[List[float]], img_info: List[Tuple[int, int]], img_size: Tuple[int, int]):
        if self.tracker is None:
            raise RuntimeError("ByteTrack tracker is None - cannot update")
        
        # Convert detections to numpy array in xyxy format with confidence and class
        # Expected format: [x1, y1, x2, y2, conf, cls]
        if len(detections) == 0:
            # Return empty array for no detections
            return np.array([], dtype=np.float32).reshape(0, 7)
        
        det_array = np.array(detections, dtype=np.float32)
        
        # Create a results-like object that matches what ultralytics BYTETracker expects
        class ResultsWrapper:
            def __init__(self, dets, img_info, img_size):
                # Store full detection array
                self._dets = dets
                self.img_info = img_info
                self.img_size = img_size
                # Compute attributes on the fly from _dets
                if len(dets) > 0:
                    self.xyxy = dets[:, :4]
                    # Convert xyxy to xywh: [x1,y1,x2,y2] -> [cx,cy,w,h]
                    x1, y1, x2, y2 = self.xyxy.T
                    w = x2 - x1
                    h = y2 - y1
                    cx = x1 + w / 2
                    cy = y1 + h / 2
                    self.xywh = np.stack([cx, cy, w, h], axis=1)
                    self.xywhr = np.concatenate([self.xywh, np.zeros((len(dets), 1))], axis=1)
                    self.conf = dets[:, 4] if dets.shape[1] > 4 else np.zeros(len(dets))
                    self.cls = dets[:, 5] if dets.shape[1] > 5 else np.zeros(len(dets))
                    self.idx = np.arange(len(dets))
                else:
                    self.xyxy = np.zeros((0, 4))
                    self.xywh = np.zeros((0, 4))
                    self.xywhr = np.zeros((0, 5))
                    self.conf = np.zeros(0)
                    self.cls = np.zeros(0)
                    self.idx = np.zeros(0, dtype=int)
            
            def __len__(self):
                return len(self.xyxy)
            
            def __getitem__(self, idx):
                # Support both boolean masks and integer indexing
                new_dets = self._dets[idx]
                return ResultsWrapper(new_dets, self.img_info, self.img_size)
            
            def __repr__(self):
                return f"ResultsWrapper(n={len(self)})"
        
        results = ResultsWrapper(det_array, img_info, img_size)
        result = self.tracker.update(results)
        
        return result


# Keep old alias
BYTETracker = ByteTrackWrapper