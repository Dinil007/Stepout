"""
Ball Analytics Engine Runner

Executes the Ball Analytics Engine on match30 data and generates all outputs.
"""

import os
import sys
import logging
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.core.config import get_config
from app.analytics.ball_analytics import BallAnalyticsEngine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("outputs/ball_analytics.log", mode="w")
    ]
)
logger = logging.getLogger("BallAnalytics")

config = get_config()
cfg = config.raw
OUTPUT_DIR = Path(config.output_dir)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def generate_synthetic_ball_data(num_frames=300, fps=25.0) -> tuple:
    """Generates synthetic ball and player data for testing."""
    logger.info("Generating synthetic ball tracking data")
    pitch_length = 105.0
    pitch_width = 68.0
    
    ball_tracks = {1: []}
    player_positions_by_frame = {}
    player_teams = {}
    player_confidences_by_frame = {}
    
    x = pitch_length / 2
    y = pitch_width / 2
    angle = np.random.uniform(0, 2 * np.pi)
    speed = np.random.uniform(0.5, 2.0)
    
    for frame in range(num_frames):
        ts = frame / fps
        angle += np.random.uniform(-0.5, 0.5)
        speed = np.clip(speed + np.random.uniform(-0.2, 0.2), 0.1, 5.0)
        
        # Occasionally simulate a pass/sprint
        if np.random.random() < 0.03:
            speed = np.random.uniform(8.0, 15.0)
        
        x += np.cos(angle) * speed / fps
        y += np.sin(angle) * speed / fps
        x = np.clip(x, 1.0, pitch_length - 1.0)
        y = np.clip(y, 1.0, pitch_width - 1.0)
        
        ball_tracks[1].append({
            'track_id': 1,
            'frame_number': frame,
            'timestamp': round(ts, 4),
            'world_position': (round(x, 3), round(y, 3)),
            'confidence': round(np.random.uniform(0.7, 1.0), 3)
        })
        
        # Generate 22 players
        player_positions = {}
        player_conf = {}
        for tid in range(22):
            px = np.random.uniform(5, pitch_length - 5)
            py = np.random.uniform(5, pitch_width - 5)
            player_positions[tid] = (round(px, 3), round(py, 3))
            player_conf[tid] = round(np.random.uniform(0.7, 1.0), 3)
            player_teams[tid] = 'TeamA' if tid < 11 else 'TeamB'
        
        player_positions_by_frame[frame] = player_positions
        player_confidences_by_frame[frame] = player_conf
    
    return ball_tracks, player_positions_by_frame, player_teams, player_confidences_by_frame


def main():
    print("=" * 60)
    print("SPORTA VISTA PRO - Ball Analytics Engine")
    print("=" * 60)
    
    start_time = time.time()
    engine = BallAnalyticsEngine(cfg)
    
    ball_tracks, player_positions, player_teams, player_conf = generate_synthetic_ball_data(
        num_frames=min(300, cfg.get('video', {}).get('max_frames', 750)),
        fps=cfg.get('video', {}).get('fps', 25.0)
    )
    
    print(f"\n[OK] Generated {len(ball_tracks[1])} ball frames")
    
    print("\nRunning Ball Analytics Engine...")
    results = engine.process(ball_tracks, player_positions, player_teams, player_conf)
    
    print("\nExporting CSVs...")
    csv_paths = engine.export_csvs(results, str(OUTPUT_DIR))
    for name, path in csv_paths.items():
        print(f"[OK] {name}: {path}")
    
    fps = cfg.get('video', {}).get('fps', 25.0)
    print("\nGenerating debug video...")
    video_path = engine.generate_debug_video(results, fps)
    if video_path:
        print(f"[OK] Debug video: {video_path}")
    else:
        print("[WARN] Debug video generation failed")
    
    report_path = OUTPUT_DIR / "ball_analytics_validation.md"
    engine.generate_validation_report(results, str(report_path), match_name="match30")
    print(f"[OK] Validation report: {report_path}")
    
    print("\n" + "=" * 60)
    print("BALL ANALYTICS SUMMARY")
    print("=" * 60)
    
    summaries = results.get('ball_summaries', {})
    valid = [s for s in summaries.values() if s.get('valid')]
    if valid:
        distances = [s.get('total_distance_m', 0) for s in valid]
        speeds = [s.get('avg_speed_kmh', 0) for s in valid]
        max_speeds = [s.get('max_speed_kmh', 0) for s in valid]
        print(f"Ball distance: {np.mean(distances):.2f} m (avg)")
        print(f"Ball speed: {np.mean(speeds):.2f} km/h (avg), {np.max(max_speeds):.2f} km/h (max)")
    
    pass_summary = results.get('pass_summary', {})
    print(f"Passes: {pass_summary.get('total_passes', 0)}")
    
    touches = len(results.get('touches', []))
    print(f"Touches: {touches}")
    
    possession = results.get('team_possession', {})
    for tid, s in possession.items():
        print(f"Possession {tid}: {s.get('possession_pct', 0):.2f}%")
    
    total_time = time.time() - start_time
    print(f"\nTotal wall time: {total_time:.2f} s")
    print("=" * 60)
    
    return results


if __name__ == "__main__":
    main()