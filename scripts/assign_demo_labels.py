"""Assign demo labels to tracks for pipeline testing.

Since the dataset generation is complete but labels are not yet assigned,
this script assigns simulated labels based on track ID ranges for testing
the training pipeline.
"""
import csv
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.classification.config import CLASS_NAMES

def assign_demo_labels(
    labels_file: Path = Path("datasets/person_classifier/labels.csv"),
    seed: int = 42,
) -> None:
    """Assign demo labels to all unlabeled tracks."""
    random.seed(seed)
    
    # Read existing labels
    rows = []
    with open(labels_file, "r") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for row in reader:
            rows.append(row)
    
    # Get all track IDs
    track_ids = [row["track_id"] for row in rows]
    
    # Assign labels: distribute across classes with some imbalance
    # to simulate real-world scenario
    class_assignments = []
    n = len(track_ids)
    n_a = max(1, int(n * 0.4))  # 40% TEAM_A
    n_b = max(1, int(n * 0.35))  # 35% TEAM_B
    n_r = max(1, int(n * 0.15))  # 15% REFEREE
    n_c = n - n_a - n_b - n_r  # remaining COACH
    
    class_assignments = ["TEAM_A"] * n_a + ["TEAM_B"] * n_b + ["REFEREE"] * n_r + ["COACH"] * n_c
    random.shuffle(class_assignments)
    
    # Update rows
    for i, row in enumerate(rows):
        if row.get("label", "").strip() == "":
            row["label"] = class_assignments[i]
            row["status"] = "LABELED"
    
    # Write back
    with open(labels_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    
    print(f"Assigned demo labels to {n} tracks")
    print(f"TEAM_A: {n_a}, TEAM_B: {n_b}, REFEREE: {n_r}, COACH: {n_c}")


if __name__ == "__main__":
    assign_demo_labels()