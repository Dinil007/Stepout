"""PHASE 3: Speed validation."""
import pandas as pd
from pathlib import Path

def main():
    debug = Path("outputs/speed_debug.csv")
    if not debug.exists():
        print("ERROR: outputs/speed_debug.csv not found")
        return
    
    df = pd.read_csv(debug)
    speeds = df['speed_kmh']
    
    stats = {
        "min_kmh": float(speeds.min()),
        "avg_kmh": float(speeds.mean()),
        "median_kmh": float(speeds.median()),
        "p95_kmh": float(speeds.quantile(0.95)),
        "max_kmh": float(speeds.max()),
        "over_30_kmh": int((speeds > 30).sum()),
        "over_35_kmh": int((speeds > 35).sum()),
        "over_40_kmh": int((speeds > 40).sum())
    }
    
    print("\n=== SPEED VALIDATION ===")
    for k, v in stats.items():
        print(f"{k}: {v}")
    
    if stats["over_40_kmh"] > 0:
        print("\nWARNING: Speeds > 40 km/h detected")
        # Find the offenders
        fast = df[speeds > 40][['frame_number', 'track_id', 'speed_kmh']]
        print(fast.to_string(index=False))
    
    # Save results
    import json
    out = Path("outputs/speed_validation.json")
    with open(out, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"\nSaved to {out}")

if __name__ == "__main__":
    main()