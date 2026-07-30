"""PHASE 4: Tracking validation."""
import json
from pathlib import Path
import pandas as pd

def main():
    print("\n=== PHASE 4: TRACKING VALIDATION ===")
    
    # Load tracking data from pipeline output
    tracking_file = Path("outputs/tracking_data.json")
    if not tracking_file.exists():
        print("STATUS: PENDING - Run pipeline first")
        return
    
    with open(tracking_file) as f:
        data = json.load(f)
    
    tracks = data.get("tracks", [])
    total_tracks = len(tracks)
    
    # Calculate metrics
    durations = [t.get("duration", 0) for t in tracks]
    avg_duration = sum(durations) / len(durations) if durations else 0
    median_duration = sorted(durations)[len(durations)//2] if durations else 0
    
    id_switches = data.get("id_switches", [])
    
    result = {
        "total_tracks": total_tracks,
        "avg_track_duration": avg_duration,
        "median_track_duration": median_duration,
        "id_switches": id_switches,
        "track_fragmentation": len(id_switches),
        "false_tracks": data.get("false_tracks", 0),
        "duplicate_tracks": data.get("duplicate_tracks", 0),
        "track_loss": data.get("track_loss", 0)
    }
    
    print(f"Total tracks: {result['total_tracks']}")
    print(f"Average track duration: {result['avg_track_duration']:.2f}s")
    print(f"Median track duration: {result['median_track_duration']:.2f}s")
    print(f"ID switches: {result['id_switches']}")
    
    out = Path("outputs/tracking_validation.json")
    with open(out, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"Saved to {out}")

if __name__ == "__main__":
    main()