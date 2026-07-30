"""
Player Kinematics Engine Runner

Executes the Player Kinematics Engine on match30.mp4 tracking data
and generates all required outputs.
"""

import os
import sys
import json
import logging
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

# Add project root to sys.path
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.core.config import get_config
from app.analytics.player_kinematics import PlayerKinematicsEngine

# ==========================================
# Configuration
# ==========================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("outputs/player_kinematics.log", mode="w")
    ]
)
logger = logging.getLogger("PlayerKinematics")

config = get_config()
cfg = config.raw

OUTPUT_DIR = Path(config.output_dir)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ==========================================
# Synthetic Tracking Data Generator (for standalone run)
# ==========================================
def generate_synthetic_tracking_data(num_players=22, num_frames=300, fps=25.0) -> Dict[int, List[Dict]]:
    """
    Generates synthetic tracking data for testing when real tracking data is unavailable.
    In production, this would be replaced with actual tracking pipeline output.
    """
    logger.info("Generating synthetic tracking data for %d players over %d frames", num_players, num_frames)
    
    all_tracks = {}
    pitch_length = 105.0
    pitch_width = 68.0
    
    for track_id in range(num_players):
        track_data = []
        # Random starting position
        x = np.random.uniform(5, pitch_length - 5)
        y = np.random.uniform(5, pitch_width - 5)
        angle = np.random.uniform(0, 2 * np.pi)
        speed = np.random.uniform(0.5, 3.0)
        
        for frame in range(num_frames):
            timestamp = frame / fps
            # Random walk with momentum
            angle += np.random.uniform(-0.3, 0.3)
            speed = np.clip(speed + np.random.uniform(-0.2, 0.2), 0.1, 4.0)
            
            x += np.cos(angle) * speed / fps
            y += np.sin(angle) * speed / fps
            
            # Keep within pitch bounds
            x = np.clip(x, 1.0, pitch_length - 1.0)
            y = np.clip(y, 1.0, pitch_width - 1.0)
            
            # Occasionally generate sprint
            if np.random.random() < 0.05:
                speed = np.random.uniform(6.0, 9.0)
                x += np.cos(angle) * speed / fps
                y += np.sin(angle) * speed / fps
            
            track_data.append({
                'track_id': track_id,
                'frame_number': frame,
                'timestamp': round(timestamp, 4),
                'world_position': (round(x, 3), round(y, 3)),
                'confidence': round(np.random.uniform(0.7, 1.0), 3)
            })
        
        all_tracks[track_id] = track_data
    
    return all_tracks


def load_tracking_from_csv(csv_path: Path) -> Dict[int, List[Dict]]:
    """
    Loads tracking data from a CSV file with columns:
    track_id, frame_number, timestamp, world_x, world_y, confidence
    """
    logger.info("Loading tracking data from %s", csv_path)
    
    if not csv_path.exists():
        raise FileNotFoundError(f"Tracking CSV not found: {csv_path}")
    
    df = pd.read_csv(csv_path)
    
    # Validate required columns
    required_cols = ['track_id', 'frame_number', 'timestamp', 'world_x', 'world_y']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in tracking CSV: {missing}")
    
    all_tracks = {}
    for track_id, group in df.groupby('track_id'):
        track_data = []
        for _, row in group.iterrows():
            track_data.append({
                'track_id': int(row['track_id']),
                'frame_number': int(row['frame_number']),
                'timestamp': float(row['timestamp']),
                'world_position': (float(row['world_x']), float(row['world_y'])),
                'confidence': float(row.get('confidence', 1.0))
            })
        all_tracks[int(track_id)] = track_data
    
    logger.info("Loaded %d tracks from CSV", len(all_tracks))
    return all_tracks


# ==========================================
# Main Execution
# ==========================================
def main():
    print("=" * 60)
    print("SPORTA VISTA PRO - Player Kinematics Engine")
    print("=" * 60)
    
    start_time = time.time()
    
    # Initialize engine
    logger.info("Initializing Player Kinematics Engine")
    engine = PlayerKinematicsEngine(cfg)
    
    # Load or generate tracking data
    tracking_csv = OUTPUT_DIR / "tracking_data.csv"
    
    if tracking_csv.exists():
        try:
            all_tracks = load_tracking_from_csv(tracking_csv)
        except Exception as e:
            logger.warning("Failed to load tracking CSV: %s. Using synthetic data.", e)
            all_tracks = generate_synthetic_tracking_data(
                num_players=22,
                num_frames=min(300, cfg.get('video', {}).get('max_frames', 750)),
                fps=cfg.get('video', {}).get('fps', 25.0)
            )
    else:
        logger.info("No tracking CSV found. Generating synthetic data for demonstration.")
        all_tracks = generate_synthetic_tracking_data(
            num_players=22,
            num_frames=min(300, cfg.get('video', {}).get('max_frames', 750)),
            fps=cfg.get('video', {}).get('fps', 25.0)
        )
    
    print(f"\n[OK] Loaded {len(all_tracks)} player tracks")
    total_frames = sum(len(t) for t in all_tracks.values())
    print(f"[OK] Total frames: {total_frames}")
    
    # Run kinematics pipeline
    print("\nRunning Player Kinematics Engine...")
    results = engine.process(all_tracks)
    
    # Export CSV outputs
    print("\nExporting CSV outputs...")
    csv_paths = engine.export_csvs(results, str(OUTPUT_DIR))
    
    print(f"[OK] player_kinematics.csv -> {csv_paths['kinematics']}")
    print(f"[OK] player_summary.csv -> {csv_paths['summary']}")
    print(f"[OK] player_validation.csv -> {csv_paths['validation']}")
    
    # Generate debug video
    fps = cfg.get('video', {}).get('fps', 25.0)
    debug_video_path = OUTPUT_DIR / "player_kinematics_debug.mp4"
    
    print("\nGenerating debug video...")
    video_success = engine.generate_debug_video(results, fps)
    if video_success:
        print(f"[OK] Debug video saved to {debug_video_path}")
    else:
        print("[WARN] Debug video generation failed or skipped")
    
    # Generate validation report
    report_path = OUTPUT_DIR / "player_kinematics_validation.md"
    engine.generate_validation_report(results, str(report_path), match_name="match30")
    print(f"[OK] Validation report saved to {report_path}")
    
    # Print summary statistics
    print("\n" + "=" * 60)
    print("PROCESSING SUMMARY")
    print("=" * 60)
    
    summaries = results.get('player_summaries', {})
    valid_summaries = [s for s in summaries.values() if s.get('valid')]
    
    if valid_summaries:
        distances = [s.get('total_distance_m', 0) for s in valid_summaries]
        speeds = [s.get('avg_speed_kmh', 0) for s in valid_summaries]
        max_speeds = [s.get('max_speed_kmh', 0) for s in valid_summaries]
        
        print(f"Players processed: {len(valid_summaries)}")
        print(f"Average distance covered: {np.mean(distances):.2f} m")
        print(f"Maximum distance: {np.max(distances):.2f} m")
        print(f"Average speed: {np.mean(speeds):.2f} km/h")
        print(f"Maximum speed: {np.max(max_speeds):.2f} km/h")
        print(f"Processing time: {results.get('processing_time_s', 0):.2f} s")
        print(f"Processing FPS: {results.get('processing_fps', 0):.2f} tracks/s")
    
    sprint_summary = results.get('sprint_summary', {})
    total_sprints = sum(s.get('sprint_count', 0) for s in sprint_summary.values())
    print(f"Total sprints detected: {total_sprints}")
    
    validation = results.get('global_validation', {})
    total_issues = validation.get('total_issues', 0)
    print(f"Total validation issues: {total_issues}")
    
    total_time = time.time() - start_time
    print(f"\nTotal wall time: {total_time:.2f} s")
    print("=" * 60)
    print("[DONE] Player Kinematics Engine completed successfully")
    
    return results


if __name__ == "__main__":
    main()