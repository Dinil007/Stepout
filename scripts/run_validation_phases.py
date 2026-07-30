"""Run all 10 validation phases sequentially."""
import argparse
import json
import time
from pathlib import Path

def run_calibration_phase():
    """PHASE 1: Real calibration."""
    print("\n=== PHASE 1: REAL CALIBRATION ===")
    # Use existing calibration from configs/homography_calibration.json
    cal_file = Path("configs/homography_calibration.json")
    if not cal_file.exists():
        print("STATUS: FAIL - No calibration file found")
        return False
    
    with open(cal_file) as f:
        cal = json.load(f)
    print(f"Reprojection error: {cal.get('validation', {}).get('mean_reprojection_error', 'N/A')}")
    print(f"Validation passed: {cal.get('validation', {}).get('validation_passed', False)}")
    return cal.get('validation', {}).get('validation_passed', False)


def run_100_frame_phase():
    """PHASE 2: Verify outputs/validation_100.json"""
    print("\n=== PHASE 2: 100 FRAME VALIDATION ===")
    out = Path("outputs/validation_100.json")
    if not out.exists():
        print("STATUS: PENDING - Run pipeline first")
        return None
    with open(out) as f:
        data = json.load(f)
    print(f"Pipeline success: {data.get('success', False)}")
    print(f"Has NaN: {data.get('has_nan', 'N/A')}")
    return data.get('success', False)


def run_speed_validation_phase():
    """PHASE 3: Speed validation."""
    print("\n=== PHASE 3: SPEED VALIDATION ===")
    debug = Path("outputs/speed_debug.csv")
    if not debug.exists():
        print("STATUS: PENDING")
        return None
    import pandas as pd
    df = pd.read_csv(debug)
    speeds = df['speed_kmh']
    stats = {
        "min": float(speeds.min()),
        "avg": float(speeds.mean()),
        "median": float(speeds.median()),
        "p95": float(speeds.quantile(0.95)),
        "max": float(speeds.max()),
        "over_30": int((speeds > 30).sum()),
        "over_35": int((speeds > 35).sum()),
        "over_40": int((speeds > 40).sum())
    }
    for k, v in stats.items():
        print(f"  {k}: {v}")
    return stats


def run_tracking_validation_phase():
    """PHASE 4: Tracking validation."""
    print("\n=== PHASE 4: TRACKING VALIDATION ===")
    tv = Path("outputs/tracking_validation.json")
    if not tv.exists():
        print("STATUS: PENDING")
        return None
    with open(tv) as f:
        data = json.load(f)
    print(f"Total tracks: {data.get('total_tracks', 'N/A')}")
    print(f"ID switches: {data.get('id_switches', [])}")
    return data


def run_analytics_validation_phase():
    """PHASE 5: Analytics validation."""
    print("\n=== PHASE 5: ANALYTICS VALIDATION ===")
    modules = ["player_tracking", "ball_tracking", "speed", "distance", "heatmap", 
               "pass_network", "formation", "tactical", "xg", "xa", "xt", "intelligence"]
    for m in modules:
        print(f"  {m}: PENDING")
    return modules


def run_frame_escalation_phase():
    """PHASE 6: Frame escalation."""
    print("\n=== PHASE 6: FRAME ESCALATION ===")
    frames = [100, 500, 1000, 2000, 10000]
    results = {}
    for n in frames:
        print(f"  {n} frames: PENDING")
        results[n] = {"status": "pending"}
    return results


def run_full_match_phase():
    """PHASE 7: Full match validation."""
    print("\n=== PHASE 7: FULL MATCH VALIDATION ===")
    fm = Path("outputs/full_match_validation.json")
    if not fm.exists():
        print("STATUS: PENDING")
        return None
    with open(fm) as f:
        return json.load(f)


def run_backend_validation_phase():
    """PHASE 8: Backend validation."""
    print("\n=== PHASE 8: BACKEND VALIDATION ===")
    services = ["FastAPI", "PostgreSQL", "SQLAlchemy", "JWT", "Redis", "Celery"]
    results = {}
    for s in services:
        results[s] = "PENDING"
        print(f"  {s}: PENDING")
    return results


def run_docker_validation_phase():
    """PHASE 9: Docker validation."""
    print("\n=== PHASE 9: DOCKER VALIDATION ===")
    print("  Build: PENDING")
    print("  Services start: PENDING")
    return {"status": "pending"}


def run_final_report_phase():
    """PHASE 10: Final report."""
    print("\n=== PHASE 10: FINAL REPORT ===")
    return {"status": "pending"}


def main():
    parser = argparse.ArgumentParser(description="Run all 10 validation phases")
    parser.add_argument("--phase", type=int, default=0, help="Run single phase (0=all)")
    args = parser.parse_args()
    
    phases = [
        run_calibration_phase,
        run_100_frame_phase,
        run_speed_validation_phase,
        run_tracking_validation_phase,
        run_analytics_validation_phase,
        run_frame_escalation_phase,
        run_full_match_phase,
        run_backend_validation_phase,
        run_docker_validation_phase,
        run_final_report_phase,
    ]
    
    if args.phase > 0:
        phases[args.phase - 1]()
    else:
        for fn in phases:
            try:
                fn()
            except Exception as e:
                print(f"ERROR: {e}")


if __name__ == "__main__":
    main()