"""
Speed Root Cause Analysis

Investigates unrealistic speeds (>40 km/h) to determine:
1. Homography calibration issues
2. Tracking instability
3. Projection errors
4. Speed estimation issues
"""
import json
import csv
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple

OUTPUT_DIR = Path("outputs")


def load_json(path: Path) -> any:
    if path.exists():
        with open(path, "r") as f:
            return json.load(f)
    return {}


def analyze_homography_validation():
    """Validate homography projection scale and accuracy."""
    print("\n" + "="*60)
    print("STEP 1: HOMOGRAPHY VALIDATION")
    print("="*60)
    
    # Load homography matrix from pipeline config
    # The homography is computed in stage_init_models
    # We'll validate using the speed_debug.csv data
    
    speed_data = []
    with open(OUTPUT_DIR / "speed_debug.csv", "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            speed_data.append({
                "frame": int(row["frame_number"]),
                "track_id": int(row["track_id"]),
                "pixel_x": float(row["pixel_x"]),
                "pixel_y": float(row["pixel_y"]),
                "field_x": float(row["field_x"]),
                "field_y": float(row["field_y"]),
                "distance_m": float(row["distance_m"]),
                "delta_time": float(row["delta_time"]),
                "speed_kmh": float(row["speed_kmh"]),
            })
    
    # Group by track_id
    tracks = {}
    for row in speed_data:
        tid = row["track_id"]
        if tid not in tracks:
            tracks[tid] = []
        tracks[tid].append(row)
    
    # Analyze high-speed events
    high_speed_threshold = 40.0
    high_speed_events = []
    
    for tid, frames in tracks.items():
        for i in range(1, len(frames)):
            prev = frames[i-1]
            curr = frames[i]
            
            if curr["speed_kmh"] > high_speed_threshold:
                # Calculate metrics
                dx_px = curr["pixel_x"] - prev["pixel_x"]
                dy_px = curr["pixel_y"] - prev["pixel_y"]
                pixel_disp = np.sqrt(dx_px**2 + dy_px**2)
                
                dx_m = curr["field_x"] - prev["field_x"]
                dy_m = curr["field_y"] - prev["field_y"]
                world_disp = curr["distance_m"]
                
                # Metres per pixel ratio
                m_per_px = world_disp / pixel_disp if pixel_disp > 0 else 0
                
                high_speed_events.append({
                    "track_id": tid,
                    "frame": curr["frame"],
                    "prev_frame": prev["frame"],
                    "speed_kmh": curr["speed_kmh"],
                    "pixel_disp": pixel_disp,
                    "world_disp": world_disp,
                    "m_per_px": m_per_px,
                    "prev_x_px": prev["pixel_x"],
                    "prev_y_px": prev["pixel_y"],
                    "curr_x_px": curr["pixel_x"],
                    "curr_y_px": curr["pixel_y"],
                    "prev_x_m": prev["field_x"],
                    "prev_y_m": prev["field_y"],
                    "curr_x_m": curr["field_x"],
                    "curr_y_m": curr["field_y"],
                })
    
    print(f"\nHigh-speed events (>40 km/h): {len(high_speed_events)}")
    
    if high_speed_events:
        # Sort by speed
        high_speed_events.sort(key=lambda x: x["speed_kmh"], reverse=True)
        
        print("\nTop 10 highest speed events:")
        print(f"{'Track':<6} {'Frame':<6} {'Speed':<8} {'Px Disp':<10} {'World Disp':<12} {'m/px':<10}")
        print("-" * 60)
        for evt in high_speed_events[:10]:
            print(f"{evt['track_id']:<6} {evt['frame']:<6} {evt['speed_kmh']:<8.1f} "
                  f"{evt['pixel_disp']:<10.1f} {evt['world_disp']:<12.3f} {evt['m_per_px']:<10.4f}")
        
        # Statistics
        speeds = [e["speed_kmh"] for e in high_speed_events]
        pixel_disps = [e["pixel_disp"] for e in high_speed_events]
        world_disps = [e["world_disp"] for e in high_speed_events]
        m_per_px_ratios = [e["m_per_px"] for e in high_speed_events]
        
        print(f"\nSpeed statistics:")
        print(f"  Min: {min(speeds):.1f} km/h")
        print(f"  Max: {max(speeds):.1f} km/h")
        print(f"  Mean: {np.mean(speeds):.1f} km/h")
        
        print(f"\nPixel displacement statistics:")
        print(f"  Min: {min(pixel_disps):.1f} px")
        print(f"  Max: {max(pixel_disps):.1f} px")
        print(f"  Mean: {np.mean(pixel_disps):.1f} px")
        
        print(f"\nWorld displacement statistics:")
        print(f"  Min: {min(world_disps):.3f} m")
        print(f"  Max: {max(world_disps):.3f} m")
        print(f"  Mean: {np.mean(world_disps):.3f} m")
        
        print(f"\nMetres-per-pixel ratio statistics:")
        print(f"  Min: {min(m_per_px_ratios):.4f} m/px")
        print(f"  Max: {max(m_per_px_ratios):.4f} m/px")
        print(f"  Mean: {np.mean(m_per_px_ratios):.4f} m/px")
        
        # Identify cause
        print("\n" + "="*60)
        print("ROOT CAUSE ANALYSIS")
        print("="*60)
        
        # Check if pixel displacements are reasonable
        # Typical player movement: 1-5 pixels/frame at 25fps
        if max(pixel_disps) > 10:
            print("\n[ISSUE] Large pixel displacements detected (>10px)")
            print("  → Likely tracking instability or bounding box jitter")
        
        # Check if metres-per-pixel ratio is consistent
        if max(m_per_px_ratios) > 0.1 or min(m_per_px_ratios) < 0.01:
            print("\n[ISSUE] Inconsistent metres-per-pixel ratio")
            print("  → Likely homography calibration issue")
        
        # Check if world displacements are reasonable
        # Max realistic player speed: ~10 m/s (36 km/h) = 0.4m/frame at 25fps
        if max(world_disps) > 0.5:
            print("\n[ISSUE] Large world displacements (>0.5m/frame)")
            print("  → Likely homography amplification of pixel noise")
        
        # Determine primary cause
        avg_pixel_disp = np.mean(pixel_disps)
        avg_world_disp = np.mean(world_disps)
        avg_m_per_px = np.mean(m_per_px_ratios)
        
        print(f"\nAverage metrics for high-speed events:")
        print(f"  Pixel displacement: {avg_pixel_disp:.1f} px")
        print(f"  World displacement: {avg_world_disp:.3f} m")
        print(f"  m/px ratio: {avg_m_per_px:.4f}")
        
        if avg_pixel_disp < 5 and avg_world_disp > 0.3:
            print("\n[CONCLUSION] Homography amplification of small pixel movements")
            print("  Root cause: Projection scale too aggressive")
        elif avg_pixel_disp > 5:
            print("\n[CONCLUSION] Tracking instability (large pixel jumps)")
            print("  Root cause: Bounding box jitter or ID switches")
        else:
            print("\n[CONCLUSION] Requires further investigation")
    
    # Save analysis
    with open(OUTPUT_DIR / "homography_validation.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "track_id", "frame", "prev_frame", "speed_kmh",
            "pixel_disp", "world_disp", "m_per_px",
            "prev_x_px", "prev_y_px", "curr_x_px", "curr_y_px",
            "prev_x_m", "prev_y_m", "curr_x_m", "curr_y_m"
        ])
        writer.writeheader()
        writer.writerows(high_speed_events)
    
    print(f"\n[OK] Homography validation saved to {OUTPUT_DIR / 'homography_validation.csv'}")


def analyze_tracking_stability():
    """Check for ID switches and tracking instability."""
    print("\n" + "="*60)
    print("STEP 2: TRACKING STABILITY ANALYSIS")
    print("="*60)
    
    speed_data = []
    with open(OUTPUT_DIR / "speed_debug.csv", "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            speed_data.append({
                "frame": int(row["frame_number"]),
                "track_id": int(row["track_id"]),
                "speed_kmh": float(row["speed_kmh"]),
            })
    
    # Group by track_id
    tracks = {}
    for row in speed_data:
        tid = row["track_id"]
        if tid not in tracks:
            tracks[tid] = []
        tracks[tid].append(row)
    
    # Check for ID switches (large position jumps between consecutive frames)
    id_switches = []
    
    for tid, frames in tracks.items():
        for i in range(1, len(frames)):
            prev = frames[i-1]
            curr = frames[i]
            
            # If speed > 40 km/h, flag as potential ID switch
            if curr["speed_kmh"] > 40.0:
                id_switches.append({
                    "track_id": tid,
                    "frame": curr["frame"],
                    "prev_frame": prev["frame"],
                    "speed_kmh": curr["speed_kmh"],
                    "reason": "High speed detected"
                })
    
    print(f"\nPotential ID switches or tracking errors: {len(id_switches)}")
    
    if id_switches:
        # Group by track_id
        by_track = {}
        for switch in id_switches:
            tid = switch["track_id"]
            if tid not in by_track:
                by_track[tid] = []
            by_track[tid].append(switch)
        
        print("\nBreakdown by track ID:")
        for tid, switches in by_track.items():
            print(f"  Track {tid}: {len(switches)} high-speed events")
    
    return id_switches


def generate_speed_root_cause_report():
    """Generate comprehensive root cause analysis report."""
    print("\n" + "="*60)
    print("STEP 3: ROOT CAUSE DETERMINATION")
    print("="*60)
    
    # Load analytics summary
    analytics = load_json(OUTPUT_DIR / "analytics.json")
    
    # Load ball tracks to check tracking quality
    ball_tracks = load_json(OUTPUT_DIR / "ball_tracks.json")
    
    report = []
    report.append("# Speed Root Cause Analysis\n")
    report.append(f"Generated: 2026-07-27\n")
    report.append(f"Dataset: 100 frames @ 25 fps\n")
    report.append("")
    
    # Analyze speed data
    speed_data = []
    with open(OUTPUT_DIR / "speed_debug.csv", "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            speed_data.append(row)
    
    speeds = [float(row["speed_kmh"]) for row in speed_data]
    max_speed = max(speeds) if speeds else 0
    avg_speed = np.mean(speeds) if speeds else 0
    
    report.append("## Summary\n")
    report.append(f"- Total speed records: {len(speed_data)}")
    report.append(f"- Maximum speed: {max_speed:.1f} km/h")
    report.append(f"- Average speed: {avg_speed:.1f} km/h")
    report.append(f"- Threshold: 40.0 km/h")
    report.append(f"- Records exceeding threshold: {sum(1 for s in speeds if s > 40)}")
    report.append("")
    
    # Identify affected tracks
    affected_tracks = set()
    for row in speed_data:
        if float(row["speed_kmh"]) > 40.0:
            affected_tracks.add(int(row["track_id"]))
    
    report.append(f"## Affected Track IDs\n")
    report.append(f"{sorted(affected_tracks)}\n")
    
    # Detailed analysis
    report.append("## Detailed Analysis\n")
    
    # Group high-speed events by track
    high_speed_by_track = {}
    for row in speed_data:
        speed = float(row["speed_kmh"])
        if speed > 40.0:
            tid = int(row["track_id"])
            if tid not in high_speed_by_track:
                high_speed_by_track[tid] = []
            high_speed_by_track[tid].append(row)
    
    for tid in sorted(high_speed_by_track.keys()):
        events = high_speed_by_track[tid]
        report.append(f"### Track ID {tid}\n")
        report.append(f"- High-speed events: {len(events)}")
        report.append(f"- Max speed: {max(float(e['speed_kmh']) for e in events):.1f} km/h")
        report.append(f"- Frames affected: {[int(e['frame_number']) for e in events[:10]]}")
        report.append("")
    
    # Root cause determination
    report.append("## Root Cause Determination\n")
    
    # Check if high speeds correlate with specific tracks
    if len(affected_tracks) <= 2:
        report.append("**Finding:** High speeds concentrated on 1-2 track IDs\n")
        report.append("**Likely cause:** Tracking instability or ID switches affecting specific players\n")
    else:
        report.append("**Finding:** High speeds distributed across multiple tracks\n")
        report.append("**Likely cause:** Systematic projection or speed estimation issue\n")
    
    # Check pixel displacements
    high_pixel_disp = False
    for i in range(1, len(speed_data)):
        prev_row = speed_data[i-1]
        curr_row = speed_data[i]
        if int(curr_row["track_id"]) == int(prev_row["track_id"]):
            dx = abs(float(curr_row["pixel_x"]) - float(prev_row["pixel_x"]))
            dy = abs(float(curr_row["pixel_y"]) - float(prev_row["pixel_y"]))
            pixel_disp = np.sqrt(dx**2 + dy**2)
            if pixel_disp > 10:
                high_pixel_disp = True
                break
    
    if high_pixel_disp:
        report.append("**Finding:** Large pixel displacements (>10px) detected\n")
        report.append("**Conclusion:** Bounding box jitter or tracking instability is amplifying homography projection\n")
    else:
        report.append("**Finding:** Pixel displacements are moderate (<10px)\n")
        report.append("**Conclusion:** Homography scale may be too aggressive, amplifying small movements\n")
    
    # Final recommendation
    report.append("## Recommended Fix\n")
    report.append("**Priority:** Investigate homography calibration\n")
    report.append("")
    report.append("**Rationale:** ")
    if high_pixel_disp:
        report.append("Large pixel displacements suggest tracking issues, but the homography is amplifying these into unrealistic world displacements.")
    else:
        report.append("Moderate pixel displacements are being projected to unrealistic world speeds, indicating homography scale issues.")
    report.append("")
    report.append("**Next steps:**")
    report.append("1. Verify homography source points (pitch corners) are correct")
    report.append("2. Verify homography destination points (field dimensions) are correct")
    report.append("3. Check if bottom-center bounding box anchor is appropriate")
    report.append("4. Consider adjusting projection scale or adding speed validation")
    report.append("")
    
    # Save report
    with open(OUTPUT_DIR / "speed_root_cause.md", "w") as f:
        f.write("\n".join(report))
    
    print(f"\n[OK] Root cause analysis saved to {OUTPUT_DIR / 'speed_root_cause.md'}")


if __name__ == "__main__":
    analyze_homography_validation()
    analyze_tracking_stability()
    generate_speed_root_cause_report()