"""
Generate speed_debug.csv from pipeline outputs.
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


def generate_speed_debug():
    # Load required data
    ball_tracks = load_json(OUTPUT_DIR / "ball_tracks.json")
    analytics = load_json(OUTPUT_DIR / "analytics.json")
    
    # We need to reconstruct speed data from the pipeline
    # Since speed_estimator state is not saved, we'll compute from mapped_players
    # This requires access to the pipeline's internal state
    
    print("Speed debug generation requires access to pipeline state.")
    print("This script should be integrated into the main pipeline.")
    
    # For now, create a placeholder report based on available data
    report = {
        "status": "requires_pipeline_integration",
        "message": "Speed debug CSV must be generated during pipeline execution",
        "recommendation": "Add speed_debug export to stage_save_outputs in run_match_analysis.py"
    }
    
    with open(OUTPUT_DIR / "speed_debug_report.json", "w") as f:
        json.dump(report, f, indent=4)
    
    print(f"Report saved to {OUTPUT_DIR / 'speed_debug_report.json'}")


if __name__ == "__main__":
    generate_speed_debug()