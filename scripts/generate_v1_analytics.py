"""
Generate V1 Analytics Artifacts

Consumes existing pipeline outputs and generates:
- Team Analytics summary
- Match Heat Map
- Team Density Maps
- Ball Density Map
- Ball Trajectory
"""

import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd

# Add project root to sys.path for module imports
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from app.analytics.team_density import TeamDensityAnalytics
from app.homography.visualize_pitch import PitchVisualizer
from app.homography.field_config import (
    PITCH_IMAGE_WIDTH,
    PITCH_IMAGE_HEIGHT,
    SCALE_X,
    SCALE_Y,
    FIELD_LENGTH_METERS,
    FIELD_WIDTH_METERS,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_team_analytics(
    player_telemetry: Dict,
    df_stats: pd.DataFrame,
    possession_events: List[Dict],
    total_frames: int,
) -> Dict:
    """
    Generate team analytics summary.

    Returns:
        Dict with team A and team B metrics
    """
    # Map numeric team IDs to team names
    team_mapping = {0: "Red", 1: "Blue"}

    # Calculate possession percentages
    team_possession_frames = {"Red": 0, "Blue": 0}
    for event in possession_events:
        team_id = event.get("team_id")
        if team_id in team_mapping:
            team_name = team_mapping[team_id]
            frames = event.get("frame_end", 0) - event.get("frame_start", 0) + 1
            team_possession_frames[team_name] += frames

    total_possession_frames = sum(team_possession_frames.values())
    if total_possession_frames == 0:
        total_possession_frames = total_frames  # fallback

    red_possession_pct = (team_possession_frames["Red"] / total_possession_frames) * 100 if total_possession_frames > 0 else 0.0
    blue_possession_pct = (team_possession_frames["Blue"] / total_possession_frames) * 100 if total_possession_frames > 0 else 0.0

    # Calculate team metrics from player stats
    team_metrics = {"Red": {}, "Blue": {}}

    for team_name in ["Red", "Blue"]:
        team_df = df_stats[df_stats["team_id"] == team_name]
        if team_df.empty:
            team_df = df_stats[df_stats["team_id"].astype(str) == team_name]

        if not team_df.empty:
            total_distance = float(team_df["total_distance_meters"].sum())
            avg_speed = float(team_df["avg_speed_kmh"].mean())
            top_speed = float(team_df["max_speed_kmh"].max())
        else:
            total_distance = 0.0
            avg_speed = 0.0
            top_speed = 0.0

        team_metrics[team_name] = {
            "possession_pct": round(red_possession_pct if team_name == "Red" else blue_possession_pct, 1),
            "total_distance_m": round(total_distance, 2),
            "avg_speed_kmh": round(avg_speed, 2),
            "top_speed_kmh": round(top_speed, 2),
        }

    return team_metrics


def generate_ball_trajectory(
    ball_positions_m: List[Tuple[float, float]],
    output_path: Path,
    base_pitch: Optional[np.ndarray] = None,
) -> Path:
    """
    Generate ball trajectory visualization.

    Args:
        ball_positions_m: List of (x_m, y_m) ball positions in chronological order
        output_path: Path to save trajectory image
        base_pitch: Optional base pitch image

    Returns:
        Path to saved image
    """
    if base_pitch is None:
        visualizer = PitchVisualizer()
        base_pitch = visualizer.base_pitch_image.copy()
    else:
        base_pitch = base_pitch.copy()
        if base_pitch.shape[:2] != (PITCH_IMAGE_HEIGHT, PITCH_IMAGE_WIDTH):
            base_pitch = cv2.resize(base_pitch, (PITCH_IMAGE_WIDTH, PITCH_IMAGE_HEIGHT))

    # Convert positions to pixels
    px_positions = []
    for x_m, y_m in ball_positions_m:
        px = int(round(x_m * SCALE_X))
        py = int(round(y_m * SCALE_Y))
        px_positions.append((px, py))

    # Draw trajectory line
    if len(px_positions) >= 2:
        pts = np.array(px_positions, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(base_pitch, [pts], isClosed=False, color=(0, 255, 255), thickness=2, lineType=cv2.LINE_AA)

    # Draw start and end points
    if px_positions:
        start_pt = px_positions[0]
        end_pt = px_positions[-1]

        # Start point (green)
        cv2.circle(base_pitch, start_pt, 6, (0, 255, 0), -1)
        cv2.putText(base_pitch, "START", (start_pt[0] - 20, start_pt[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # End point (red)
        cv2.circle(base_pitch, end_pt, 6, (0, 0, 255), -1)
        cv2.putText(base_pitch, "END", (end_pt[0] - 15, end_pt[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), base_pitch)
    logger.info("Ball trajectory saved: %s", output_path)
    return output_path


def generate_all_analytics(output_dir: Path, config: Dict) -> None:
    """
    Main function to generate all V1 analytics artifacts.
    """
    logger.info("Starting V1 analytics generation...")

    output_dir = Path(output_dir)

    # Load existing outputs
    telemetry_path = output_dir / "tracking_telemetry.json"
    if not telemetry_path.exists():
        logger.warning("tracking_telemetry.json not found at %s. Skipping analytics generation.", telemetry_path)
        return

    with open(telemetry_path, "r") as f:
        player_telemetry = json.load(f)

    # Load player stats
    csv_path = output_dir / "player_statistics.csv"
    df_stats = pd.DataFrame()
    if csv_path.exists():
        df_stats = pd.read_csv(csv_path)

    # Load possession summary
    possession_path = output_dir / "team_possession_summary.json"
    possession_events = []
    total_frames = 0

    if possession_path.exists():
        with open(possession_path, "r") as f:
            poss_data = json.load(f)
            total_frames = poss_data.get("total_frames", 0)
            possession_events = poss_data.get("possession_events", [])
    else:
        logger.info("team_possession_summary.json not found. Using default possession data.")

    # Initialize density analytics
    density_analytics = TeamDensityAnalytics()
    visualizer = PitchVisualizer()
    base_pitch = visualizer.base_pitch_image.copy()

    # Prepare player positions by frame
    player_positions_by_frame = {}
    for track_id, data in player_telemetry.items():
        frames = data.get("frames", [])
        positions_px = data.get("positions_px", [])
        team_id = data.get("team_id")

        for i, frame_num in enumerate(frames):
            if i < len(positions_px):
                x_px, y_px = positions_px[i]
                x_m = x_px / SCALE_X
                y_m = y_px / SCALE_Y
                player_positions_by_frame.setdefault(frame_num, {})[int(track_id)] = (x_m, y_m)

    # 1. Generate Team Analytics
    logger.info("Generating team analytics...")
    team_analytics = generate_team_analytics(
        player_telemetry,
        df_stats if not df_stats.empty else pd.DataFrame([{"team_id": 0, "total_distance_meters": 0, "avg_speed_kmh": 0, "max_speed_kmh": 0}]),
        possession_events,
        total_frames,
    )

    with open(output_dir / "team_analytics.json", "w") as f:
        json.dump(team_analytics, f, indent=2)
    logger.info("Team analytics saved.")

    # 2. Generate Match Heat Map
    logger.info("Generating match heat map...")
    match_heatmap = density_analytics.generate_match_density_map(
        player_positions_by_frame,
        base_pitch=base_pitch,
        alpha=0.6,
    )
    TeamDensityAnalytics.save_image(match_heatmap, output_dir / "match_heatmap.png")

    # 3. Generate Team Density Maps
    logger.info("Generating team density maps...")
    # Extract team assignments from telemetry: track_id -> team_id
    team_assignments = {int(k): str(v.get("team_id", "Unknown")) for k, v in player_telemetry.items()}

    team_a_density = density_analytics.generate_team_density_map(
        player_positions_by_frame,
        team_assignments,
        target_team=0,
        base_pitch=base_pitch,
        alpha=0.6,
    )
    TeamDensityAnalytics.save_image(team_a_density, output_dir / "team_a_density_map.png")

    team_b_density = density_analytics.generate_team_density_map(
        player_positions_by_frame,
        team_assignments,
        target_team=1,
        base_pitch=base_pitch,
        alpha=0.6,
    )
    TeamDensityAnalytics.save_image(team_b_density, output_dir / "team_b_density_map.png")

    # 4. Generate Ball Density Map
    logger.info("Generating ball density map...")
    # Collect ball positions from tracking telemetry (team_id = ball or track_id = 0)
    ball_positions = []
    for track_id, data in player_telemetry.items():
        if data.get("team_id") == "ball" or track_id == "0":
            positions_m = data.get("positions_m", [])
            ball_positions.extend(positions_m)

    if ball_positions:
        logger.info("Generating ball density map with %d ball positions...", len(ball_positions))
        ball_density = density_analytics.generate_ball_density_map(
            ball_positions,
            base_pitch=base_pitch,
            alpha=0.6,
        )
        TeamDensityAnalytics.save_image(ball_density, output_dir / "ball_density_map.png")
    else:
        logger.warning("No ball position data available. Skipping ball density map.")

    # 5. Generate Ball Trajectory
    logger.info("Generating ball trajectory...")
    ball_trajectory_path = output_dir / "ball_trajectory.png"

    # Collect ball positions from tracking telemetry (team_id = ball or track_id = 0)
    ball_positions_m = []
    for track_id, data in player_telemetry.items():
        if data.get("team_id") == "ball" or track_id == "0":
            positions_m = data.get("positions_m", [])
            ball_positions_m.extend(positions_m)

    if ball_positions_m:
        generate_ball_trajectory(ball_positions_m, ball_trajectory_path, base_pitch)
    else:
        logger.warning("No ball trajectory data available.")

    logger.info("V1 analytics generation completed!")
    logger.info("Generated files:")
    logger.info("  - team_analytics.json")
    logger.info("  - match_heatmap.png")
    logger.info("  - team_a_density_map.png")
    logger.info("  - team_b_density_map.png")
    logger.info("  - ball_density_map.png")
    logger.info("  - ball_trajectory.png")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate V1 analytics artifacts")
    parser.add_argument("--output", type=str, default="outputs", help="Output directory")
    parser.add_argument("--config", type=str, default="config.yaml", help="Config file path")
    args = parser.parse_args()

    # Load config if exists
    config = {}
    config_path = Path(args.config)
    if config_path.exists():
        import yaml
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

    generate_all_analytics(Path(args.output), config)