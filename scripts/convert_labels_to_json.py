"""Convert labels.csv to labels.json for the training pipeline."""
import csv
import json
from pathlib import Path

def convert_csv_to_json(
    csv_path: Path = Path("datasets/person_classifier/labels.csv"),
    json_path: Path = Path("datasets/person_classifier/metadata/labels.json"),
) -> None:
    """Convert CSV labels to JSON format expected by the pipeline."""
    labels = {}
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            track_id = row.get("track_id", "").strip()
            label = row.get("label", "").strip()
            status = row.get("status", "").strip()
            if track_id and label and label != "UNLABELED":
                # Store with both formats to ensure matching
                labels[track_id] = label
                labels[f"track_{track_id.zfill(4)}"] = label

    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w") as f:
        json.dump(labels, f, indent=2)

    print(f"Converted {len(labels)} labels to JSON")
    print(f"CSV: {csv_path}")
    print(f"JSON: {json_path}")

if __name__ == "__main__":
    convert_csv_to_json()