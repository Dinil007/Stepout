"""
Test script for BallPathRenderer
"""

import sys
from pathlib import Path
import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.visualization import BallPathRenderer
from app.core.config import get_config

config = get_config()
cfg = config.raw

renderer = BallPathRenderer(cfg)

# Simulate a sequence with team possession changes and a pass
renderer.update(0, 0.0, (100, 100), (10.0, 10.0), 'team1', 1, True, 0.9, 5.0, False, False)
renderer.update(1, 0.04, (105, 105), (11.0, 11.0), 'team1', 1, True, 0.9, 6.0, False, False)
renderer.update(2, 0.08, (110, 110), (12.0, 12.0), 'team1', 1, True, 0.9, 7.0, False, False)
renderer.update(3, 0.12, (200, 200), (20.0, 20.0), 'team1', 1, True, 0.9, 25.0, True, False)
renderer.update(4, 0.16, (250, 250), (25.0, 25.0), 'team2', 5, True, 0.8, 10.0, False, True)
renderer.update(5, 0.20, (260, 260), (26.0, 26.0), 'team2', 5, True, 0.8, 8.0, False, False)

frame = np.zeros((renderer.canvas_size[1], renderer.canvas_size[0], 3), dtype=np.uint8)
rendered = renderer.render(frame)
rendered = renderer.draw_debug_overlay(rendered, cfg)

import cv2
cv2.imwrite("outputs/ball_path_test.png", rendered)
print("[OK] Saved test image to outputs/ball_path_test.png")

success = renderer.generate_debug_video("outputs/ball_path_debug.mp4", fps=25.0)
if success:
    print("[OK] Saved debug video to outputs/ball_path_debug.mp4")
else:
    print("[WARN] Debug video generation failed")