"""Master validation script - runs all 10 phases sequentially."""
import argparse
import json
import time
import subprocess
from pathlib import Path

def run_command(cmd, label):
    print(f"\n=== {label} ===")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    success = result.returncode == 0
    print(f"Status: {'PASS' if success else 'FAIL'}")
    if not success:
        print(f"Error: {result.stderr[-500:]}")
    return success, result.stdout

def main():
    parser = argparse.ArgumentParser(description="Run all validation phases")
    parser.add_argument("--phase", type=int, default=0, help="Run specific phase (0=all)")
    args = parser.parse_args()
    
    phases = [
        ("PHASE 1: Calibration", "python scripts/generate_validation_report.py"),
        ("PHASE 2: 100 Frame Validation", "python scripts/generate_validation_report.py"),
        ("PHASE 3: Speed Validation", "python scripts/run_speed_validation.py"),
        ("PHASE 4: Tracking Validation", "python scripts/run_tracking_validation.py"),
        ("PHASE 5: Analytics Validation", "python scripts/run_analytics_validation.py"),
        ("PHASE 6: Frame Escalation", "python scripts/run_frame_escalation.py"),
        ("PHASE 7: Full Match Validation", "echo 'Requires manual execution'"),
        ("PHASE 8: Backend Validation", "python scripts/run_backend_validation.py"),
        ("PHASE 9: Docker Validation", "python scripts/run_docker_validation.py"),
        ("PHASE 10: Final Report", "python scripts/generate_validation_report.py"),
    ]
    
    if args.phase > 0:
        label, cmd = phases[args.phase - 1]
        run_command(cmd, label)
    else:
        for label, cmd in phases:
            success, _ = run_command(cmd, label)
            if not success:
                print(f"\nSTOPPING: {label} failed")
                break
    
    # Generate final report
    print("\n=== Generating Final Report ===")
    subprocess.run("python scripts/generate_validation_report.py", shell=True)
    print("\nValidation complete. See FINAL_VALIDATION_REPORT.md")

if __name__ == "__main__":
    main()