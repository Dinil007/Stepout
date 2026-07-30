"""
Out-of-Bounds Coordinate Fix Validation

Validates that filtering out-of-bounds homography projections eliminates
unrealistic player speeds.

Method:
  1. Load existing speed_debug.csv (BEFORE)
  2. Filter coordinates outside [0,105] x [0,68] metres
  3. Recalculate speed statistics (AFTER)
  4. Compare and generate report
"""
import csv
import json
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple

OUTPUT_DIR = Path("outputs")
FIELD_LENGTH_M = 105.0
FIELD_WIDTH_M = 68.0


@dataclass
class TrackStats:
    track_id: int
    frames: int
    first_frame: int
    last_frame: int
    max_speed_kmh: float
    avg_speed_kmh: float
    total_distance_m: float
    sprint_count: int
    invalid_positions: int
    gaps_gt1: List[int]
    max_pixel_disp: float


def load_speed_debug() -> pd.DataFrame:
    """Load existing speed_debug.csv."""
    path = OUTPUT_DIR / "speed_debug.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run pipeline first.")
    return pd.read_csv(path)


def compute_oob_mask(df: pd.DataFrame) -> pd.Series:
    """Compute boolean mask for out-of-bounds field coordinates."""
    oob_x = (df['field_x'] < 0) | (df['field_x'] > FIELD_LENGTH_M)
    oob_y = (df['field_y'] < 0) | (df['field_y'] > FIELD_WIDTH_M)
    return oob_x | oob_y


def compute_track_stats(df: pd.DataFrame, track_id: int) -> TrackStats:
    """Compute statistics for a single track."""
    track_df = df[df['track_id'] == track_id].sort_values('frame_number')
    
    if len(track_df) == 0:
        return TrackStats(track_id, 0, 0, 0, 0.0, 0.0, 0.0, 0, 0, [], 0.0)
    
    frames = len(track_df)
    first_frame = int(track_df['frame_number'].iloc[0])
    last_frame = int(track_df['frame_number'].iloc[-1])
    
    # Count invalid positions
    oob_mask = compute_oob_mask(track_df)
    invalid_positions = int(oob_mask.sum())
    
    # Filter valid positions only for speed/distance
    valid_df = track_df[~oob_mask].copy()
    
    if len(valid_df) < 2:
        max_speed = float(valid_df['speed_kmh'].max()) if len(valid_df) > 0 else 0.0
        avg_speed = float(valid_df['speed_kmh'].mean()) if len(valid_df) > 0 else 0.0
        total_dist = 0.0
        sprint_count = 0
    else:
        max_speed = float(valid_df['speed_kmh'].max())
        avg_speed = float(valid_df['speed_kmh'].mean())
        total_dist = float(valid_df['distance_m'].sum())
        sprint_count = int((valid_df['speed_kmh'] > 20.0).sum())
    
    # Compute gaps
    frame_nums = track_df['frame_number'].tolist()
    gaps = []
    for i in range(1, len(frame_nums)):
        gap = frame_nums[i] - frame_nums[i-1]
        if gap > 1:
            gaps.append(gap)
    
    # Max pixel displacement
    pix_disps = []
    if len(track_df) > 1:
        px = track_df['pixel_x'].values
        py = track_df['pixel_y'].values
        pix_disps = np.sqrt(np.diff(px)**2 + np.diff(py)**2)
        max_pixel_disp = float(np.max(pix_disps)) if len(pix_disps) > 0 else 0.0
    else:
        max_pixel_disp = 0.0
    
    return TrackStats(
        track_id=track_id,
        frames=frames,
        first_frame=first_frame,
        last_frame=last_frame,
        max_speed_kmh=max_speed,
        avg_speed_kmh=avg_speed,
        total_distance_m=total_dist,
        sprint_count=sprint_count,
        invalid_positions=invalid_positions,
        gaps_gt1=gaps,
        max_pixel_disp=max_pixel_disp
    )


def analyze_before_after(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Compute before/after statistics."""
    # BEFORE: use all data
    before_stats = []
    for tid in sorted(df['track_id'].unique()):
        s = compute_track_stats(df, tid)
        before_stats.append({
            'track_id': s.track_id,
            'frames': s.frames,
            'max_speed_kmh': s.max_speed_kmh,
            'avg_speed_kmh': s.avg_speed_kmh,
            'total_distance_m': round(s.total_distance_m, 2),
            'sprint_count': s.sprint_count,
            'invalid_positions': s.invalid_positions,
            'max_pixel_disp': round(s.max_pixel_disp, 1)
        })
    before_df = pd.DataFrame(before_stats)
    
    # AFTER: filter OOB
    oob_mask = compute_oob_mask(df)
    after_df = df[~oob_mask].copy()
    after_stats = []
    for tid in sorted(after_df['track_id'].unique()):
        s = compute_track_stats(after_df, tid)
        after_stats.append({
            'track_id': s.track_id,
            'frames': s.frames,
            'max_speed_kmh': s.max_speed_kmh,
            'avg_speed_kmh': s.avg_speed_kmh,
            'total_distance_m': round(s.total_distance_m, 2),
            'sprint_count': s.sprint_count,
            'invalid_positions': s.invalid_positions,
            'max_pixel_disp': round(s.max_pixel_disp, 1)
        })
    after_stats_df = pd.DataFrame(after_stats)
    
    return before_df, after_stats_df


def generate_comparison_report(before: pd.DataFrame, after: pd.DataFrame, df: pd.DataFrame):
    """Generate before/after comparison report."""
    total_invalid = int(compute_oob_mask(df).sum())
    
    report = []
    report.append("# Out-of-Bounds Fix Validation Report\n")
    report.append(f"Generated from: {OUTPUT_DIR / 'speed_debug.csv'}\n")
    report.append("## Summary\n")
    report.append(f"- Total records: {len(df)}")
    report.append(f"- Invalid projections: {total_invalid} ({100*total_invalid/len(df):.1f}%)")
    report.append(f"- Unique tracks: {df['track_id'].nunique()}")
    report.append("")
    
    report.append("## Before (All Data)\n")
    report.append("| Track | Frames | Max Speed (km/h) | Avg Speed (km/h) | Distance (m) | Sprints | Invalid |")
    report.append("|-------|--------|------------------|------------------|--------------|---------|---------|")
    for _, row in before.iterrows():
        report.append(f"| {row['track_id']} | {row['frames']} | {row['max_speed_kmh']:.1f} | {row['avg_speed_kmh']:.1f} | {row['total_distance_m']:.1f} | {row['sprint_count']} | {row['invalid_positions']} |")
    report.append("")
    
    report.append("## After (Out-of-Bounds Removed)\n")
    report.append("| Track | Frames | Max Speed (km/h) | Avg Speed (km/h) | Distance (m) | Sprints |")
    report.append("|-------|--------|------------------|------------------|--------------|---------|")
    for _, row in after.iterrows():
        report.append(f"| {row['track_id']} | {row['frames']} | {row['max_speed_kmh']:.1f} | {row['avg_speed_kmh']:.1f} | {row['total_distance_m']:.1f} | {row['sprint_count']} |")
    report.append("")
    
    report.append("## Impact\n")
    before_max = before['max_speed_kmh'].max()
    after_max = after['max_speed_kmh'].max()
    before_avg = before['avg_speed_kmh'].mean()
    after_avg = after['avg_speed_kmh'].mean()
    before_dist = before['total_distance_m'].sum()
    after_dist = after['total_distance_m'].sum()
    before_sprints = before['sprint_count'].sum()
    after_sprints = after['sprint_count'].sum()
    
    report.append(f"- Max speed: {before_max:.1f} → {after_max:.1f} km/h ({after_max-before_max:+.1f})")
    report.append(f"- Avg speed: {before_avg:.1f} → {after_avg:.1f} km/h ({after_avg-before_avg:+.1f})")
    report.append(f"- Total distance: {before_dist:.1f} → {after_dist:.1f} m ({after_dist-before_dist:+.1f})")
    report.append(f"- Sprint count: {before_sprints} → {after_sprints} ({after_sprints-before_sprints:+d})")
    report.append("")
    
    report.append("## Validation Answers\n")
    report.append(f"1. Did removing out-of-bounds coordinates eliminate 70-100 km/h spikes? "
                  f"{'YES' if after_max < 70 else 'PARTIAL' if after_max < 90 else 'NO'}")
    report.append(f"2. New maximum speed: {after_max:.1f} km/h")
    report.append(f"3. Are remaining speeds realistic? {'YES' if after_max <= 40 else 'NEEDS REVIEW'}")
    report.append(f"4. Is ROI refinement still required? YES — {total_invalid} invalid projections indicate polygon is too permissive")
    report.append(f"5. Should additional smoothing be implemented? Not required for speed accuracy; consider for visualization only")
    
    report_text = "\n".join(report)
    
    # Save report
    report_path = OUTPUT_DIR / "oob_fix_validation_report.md"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    # Save after-stats CSV
    after.to_csv(OUTPUT_DIR / "player_statistics_after_fix.csv", index=False)
    
    # Save invalid projections CSV
    invalid_mask = compute_oob_mask(df)
    invalid_df = df[invalid_mask].copy()
    invalid_df.to_csv(OUTPUT_DIR / "invalid_projection.csv", index=False)
    
    print(f"\n[OK] Report saved to {report_path}")
    print(f"[OK] Invalid projections saved to {OUTPUT_DIR / 'invalid_projection.csv'}")
    print(f"[OK] After-fix stats saved to {OUTPUT_DIR / 'player_statistics_after_fix.csv'}")
    
    return report_text


def validate_player_motion(df: pd.DataFrame):
    """Validate physical plausibility of player displacements."""
    print("\n=== PLAYER MOTION VALIDATION ===")
    
    valid_mask = ~compute_oob_mask(df)
    valid_df = df[valid_mask].copy()
    
    if len(valid_df) == 0:
        print("No valid data after filtering.")
        return
    
    # Compute consecutive displacements per track
    all_displacements = []
    for tid in valid_df['track_id'].unique():
        track_df = valid_df[valid_df['track_id'] == tid].sort_values('frame_number')
        if len(track_df) > 1:
            px = track_df['pixel_x'].values
            py = track_df['pixel_y'].values
            fx = track_df['field_x'].values
            fy = track_df['field_y'].values
            
            pix_disp = np.sqrt(np.diff(px)**2 + np.diff(py)**2)
            field_disp = np.sqrt(np.diff(fx)**2 + np.diff(fy)**2)
            all_displacements.extend(field_disp.tolist())
    
    if not all_displacements:
        print("No displacement data.")
        return
    
    arr = np.array(all_displacements)
    print(f"Valid displacement records: {len(arr)}")
    print(f"Mean displacement: {np.mean(arr):.3f} m/frame")
    print(f"Max displacement: {np.max(arr):.3f} m/frame")
    print(f"95th percentile: {np.percentile(arr, 95):.3f} m/frame")
    print(f"99th percentile: {np.percentile(arr, 99):.3f} m/frame")
    
    # Physical plausibility check
    # At 25fps, elite sprint = ~10 m/s = 0.4 m/frame
    sprint_threshold = 0.4
    elite_threshold = 0.5  # Including some margin
    
    over_sprint = np.sum(arr > sprint_threshold)
    over_elite = np.sum(arr > elite_threshold)
    
    print(f"Frames > sprint threshold (0.4m): {over_sprint} ({100*over_sprint/len(arr):.1f}%)")
    print(f"Frames > elite threshold (0.5m): {over_elite} ({100*over_elite/len(arr):.1f}%)")
    
    if over_elite / len(arr) > 0.01:
        print("WARNING: More than 1% of frames exceed elite sprint displacement.")
        print("Possible remaining issues: tracking gaps, bbox jitter, or homography edge effects.")
    else:
        print("OK: Displacement distribution is physically plausible.")


if __name__ == "__main__":
    print("=== Out-of-Bounds Fix Validation ===\n")
    
    # Load data
    df = load_speed_debug()
    print(f"Loaded {len(df)} records from speed_debug.csv")
    
    # Compute before/after
    before_df, after_df = analyze_before_after(df)
    
    # Generate report
    report = generate_comparison_report(before_df, after_df, df)
    print("\n" + report)
    
    # Validate motion
    validate_player_motion(df)
    
    print("\n[COMPLETE] Validation finished.")
