"""Homography visualization module."""

import logging
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class HomographyVisualizer:
    """Generate homography debug visualization."""

    def __init__(self, pitch_model, output_path: Optional[Path] = None) -> None:
        self.pitch_model = pitch_model
        self.output_path = output_path

    def draw_pitch_overlay(self, frame: np.ndarray, estimator) -> np.ndarray:
        """Draw pitch outline on frame."""
        vis = frame.copy()
        if estimator is None or estimator.calibration_result is None:
            return vis
        corners = self.pitch_model.corners
        for i in range(4):
            p1 = tuple(corners[i].astype(int))
            p2 = tuple(corners[(i + 1) % 4].astype(int))
            cv2.line(vis, p1, p2, (0, 255, 255), 2)
        cv2.putText(vis, "Homography active", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        return vis

    def generate_debug_video(self, frames: List[np.ndarray], estimator, output_path: Path) -> None:
        """Generate debug video with homography overlay."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if not frames:
            return
        h, w = frames[0].shape[:2]
        writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*'mp4v'), 25.0, (w, h))
        for frame in frames:
            vis = self.draw_pitch_overlay(frame, estimator)
            writer.write(vis)
        writer.release()
        logger.info(f"Homography debug video saved to {output_path}")