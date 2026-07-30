"""
Final Root Cause Analysis - Stage-by-Stage Investigation

For every speed > 40 km/h event, trace:
  YOLO bbox (not available) -> bottom-center anchor -> homography -> canvas -> field -> speed

We have pixel anchor, field coordinates, distance, speed in speed_debug.csv.
Reconstruct canvas coords as canvas = field * 10 (since field = canvas * 0.1).

Determine which stage FIRST introduces the unrealistic jump.
"""
import csv
import numpy as np
import pandas as pd
from pathlib import Path

OUTPUT_DIR = Path("outputs")
SPEED_THRESHOLD = 40.0


def load_speed_debug() -> pd.DataFrame:
    df = pd.read_csv(OUTPUT_DIR / "speed_debug.csv")
    # Reconstruct canvas coordinates
    df['canvas_x'] = df['field_x'] * 10.0
    df['canvas_y'] = df['field_y'] * 10.0
    return df


def analyze_spikes(df: pd.DataFrame):
    """For every track, compute consecutive displacements and find spikes."""
    results = []
    
    for tid in sorted(df['track_id'].unique()):
        track_df = df[df['track_id'] == tid].sort_values('frame_number').reset_index(drop=True)
        if len(track_df) < 2:
            continue
        
        for i in range(1, len(track_df)):
            prev = track_df.iloc[i-1]
            curr = track_df.iloc[i]
            
            speed = float(curr['speed_kmh'])
            if speed > SPEED_THRESHOLD:
                # Compute displacements
                pix_dx = float(curr['pixel_x'] - prev['pixel_x'])
                pix_dy = float(curr['pixel_y'] - prev['pixel_y'])
                pix_disp = np.sqrt(pix_dx**2 + pix_dy**2)
                
                canvas_dx = float(curr['canvas_x'] - prev['canvas_x'])
                canvas_dy = float(curr['canvas_y'] - prev['canvas_y'])
                canvas_disp = np.sqrt(canvas_dx**2 + canvas_dy**2)
                
                field_disp = float(curr['distance_m'])
                
                # Effective metres per pixel = field_disp / pix_disp
                eff_m_per_px = field_disp / pix_disp if pix_disp > 0 else float('inf')
                
                # Determine origin
                if pix_disp > 10.0:
                    origin = "A/B: Large pixel displacement (detection jitter)"
                elif eff_m_per_px > 0.12:
                    origin = "C/D: Homography amplification near edge"
                else:
                    origin = "Unknown"
                
                results.append({
                    'track_id': tid,
                    'frame': int(curr['frame_number']),
                    'prev_frame': int(prev['frame_number']),
                    'speed_kmh': speed,
                    'pix_disp': round(pix_disp, 2),
                    'canvas_disp': round(canvas_disp, 2),
                    'field_disp': round(field_disp, 3),
                    'eff_m_per_px': round(eff_m_per_px, 4),
                    'origin': origin,
                    'pixel_x': float(curr['pixel_x']),
                    'pixel_y': float(curr['pixel_y']),
                    'canvas_x': float(curr['canvas_x']),
                    'canvas_y': float(curr['canvas_y']),
                    'field_x': float(curr['field_x']),
                    'field_y': float(curr['field_y']),
                })
    
    return pd.DataFrame(results)


def print_spike_equations(df_spikes: pd.DataFrame):
    """Print exact equations for each spike."""
    print("\n=== SPIKE EQUATIONS ===\n")
    for _, row in df_spikes.iterrows():
        print(f"Track {row['track_id']}, Frame {row['frame']} (prev {row['prev_frame']}): {row['speed_kmh']:.1f} km/h")
        print(f"  Pixel displacement = {row['pix_disp']:.2f} px")
        print(f"  Canvas displacement = {row['canvas_disp']:.2f} px")
        print(f"  Field displacement  = {row['field_disp']:.3f} m")
        print(f"  Effective m/px      = {row['eff_m_per_px']:.4f}")
        print(f"  Origin              = {row['origin']}")
        print()


def summarize_by_stage(df_spikes: pd.DataFrame):
    """Summarize which stage first introduces the unrealistic jump."""
    print("\n=== STAGE ORIGIN SUMMARY ===\n")
    if df_spikes.empty:
        print("No spikes > 40 km/h found.")
        return
    
    origin_counts = df_spikes['origin'].value_counts()
    for origin, count in origin_counts.items():
        pct = 100.0 * count / len(df_spikes)
        print(f"  {origin}: {count} ({pct:.1f}%)")
    
    # Additional statistics
    print(f"\nTotal spikes: {len(df_spikes)}")
    print(f"Mean pixel displacement: {df_spikes['pix_disp'].mean():.2f} px")
    print(f"Mean effective m/px: {df_spikes['eff_m_per_px'].mean():.4f}")
    print(f"Mean field displacement: {df_spikes['field_disp'].mean():.3f} m")
    
    # Categorize
    large_pix = df_spikes[df_spikes['pix_disp'] > 10.0]
    amplified = df_spikes[(df_spikes['pix_disp'] <= 10.0) & (df_spikes['eff_m_per_px'] > 0.12)]
    
    print(f"\nLarge pixel displacement (>10px): {len(large_pix)} ({100.0*len(large_pix)/len(df_spikes):.1f}%)")
    print(f"Homography amplified (<=10px, m/px>0.12): {len(amplified)} ({100.0*len(amplified)/len(df_spikes):.1f}%)")


def detailed_case_studies(df_spikes: pd.DataFrame):
    """Show a few concrete examples with full numbers."""
    print("\n=== CASE STUDIES ===\n")
    # Pick a few representative spikes
    for _, row in df_spikes.head(5).iterrows():
        dt = 0.04  # 1/25s
        expected_speed_at_10ms = row['field_disp'] / dt
        expected_speed_kmh = expected_speed_at_10ms * 3.6
        print(f"Track {row['track_id']} Frame {row['frame']}:")
        print(f"  Pixel disp = {row['pix_disp']:.2f} px")
        print(f"  Canvas disp = {row['canvas_disp']:.2f} px")
        print(f"  Field disp = {row['field_disp']:.3f} m")
        print(f"  Speed = {row['speed_kmh']:.1f} km/h")
        print(f"  Origin = {row['origin']}")
        print()


if __name__ == "__main__":
    df = load_speed_debug()
    spikes = analyze_spikes(df)
    print_spike_equations(spikes)
    summarize_by_stage(spikes)
    detailed_case_studies(spikes)
    
    # Save detailed spike log
    spikes.to_csv(OUTPUT_DIR / "spike_stage_analysis.csv", index=False)
    print(f"\n[OK] Spike stage analysis saved to {OUTPUT_DIR / 'spike_stage_analysis.csv'}")
