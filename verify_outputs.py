"""Output Validation Script"""
import json
import csv
import os
from pathlib import Path

OUTPUT_DIR = Path("outputs")

required_files = {
    "videos": ["preprocessing.mp4", "detection.mp4", "tracking.mp4", "team_classification.mp4", "pitch_view.mp4"],
    "images": ["heatmap.png"],
    "tabular": ["player_statistics.csv", "analytics.json"]
}

print("=" * 60)
print("OUTPUT FILE VALIDATION")
print("=" * 60)

all_ok = True
for category, files in required_files.items():
    print(f"\n{category.upper()}:")
    for fname in files:
        fpath = OUTPUT_DIR / fname
        exists = fpath.exists()
        size = fpath.stat().st_size if exists else 0
        status = "✓" if exists and size > 0 else "✗"
        print(f"  {status} {fname}: {size} bytes")
        if not exists or size == 0:
            all_ok = False

# Validate analytics.json content
print("\nANALYTICSJSON VALIDATION:")
analytics_path = OUTPUT_DIR / "analytics.json"
if analytics_path.exists():
    try:
        with open(analytics_path) as f:
            data = json.load(f)
        
        # Check structure
        assert "match_info" in data, "Missing match_info"
        assert "summary_metrics" in data, "Missing summary_metrics"
        assert "player_statistics" in data, "Missing player_statistics"
        
        # Check for impossible speeds
        players = data["player_statistics"]
        max_speed = max(p.get("max_speed_kmh", 0) for p in players)
        avg_speed = max(p.get("avg_speed_kmh", 0) for p in players)
        
        print(f"  ✓ Valid JSON structure")
        print(f"  ✓ Players tracked: {len(players)}")
        print(f"  ✓ Max speed: {max_speed:.1f} km/h")
        print(f"  ✓ Max avg speed: {avg_speed:.1f} km/h")
        
        if max_speed > 40:
            print(f"  ⚠ WARNING: Max speed {max_speed:.1f} km/h exceeds plausible limit (40 km/h)")
        else:
            print(f"  ✓ All speeds within plausible range")
            
    except Exception as e:
        print(f"  ✗ analytics.json validation failed: {e}")
        all_ok = False
else:
    print("  ✗ analytics.json not found")
    all_ok = False

# Validate player_statistics.csv
print("\nPLAYER_STATISTICS.CSV VALIDATION:")
csv_path = OUTPUT_DIR / "player_statistics.csv"
if csv_path.exists():
    try:
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        
        print(f"  ✓ Valid CSV with {len(rows)} players")
        
        # Check for invalid speeds
        invalid_speeds = []
        for row in rows:
            max_spd = float(row.get("max_speed_kmh", 0))
            if max_spd > 40:
                invalid_speeds.append((row["track_id"], max_spd))
        
        if invalid_speeds:
            print(f"  ⚠ WARNING: {len(invalid_speeds)} players with speed > 40 km/h")
            for tid, spd in invalid_speeds[:5]:
                print(f"    - Track {tid}: {spd:.1f} km/h")
        else:
            print(f"  ✓ All player speeds within plausible range")
            
    except Exception as e:
        print(f"  ✗ player_statistics.csv validation failed: {e}")
        all_ok = False
else:
    print("  ✗ player_statistics.csv not found")
    all_ok = False

print("\n" + "=" * 60)
if all_ok:
    print("✓ ALL OUTPUTS VALID")
else:
    print("✗ SOME OUTPUTS MISSING OR INVALID")
print("=" * 60)