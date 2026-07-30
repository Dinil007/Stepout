"""Run Track Quality Validator on existing dataset.
Validates all tracks, generates quality report, moves rejected tracks,
and creates visual contact sheets for verification."""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import csv
from collections import defaultdict
from typing import Dict, List

from app.classification.track_validator import TrackValidator, TrackQualityResult
from app.dataset.dataset_builder import TrackCropRecord


def load_metadata(dataset_root: Path) -> Dict[int, List[TrackCropRecord]]:
    """Load metadata.csv and reconstruct TrackCropRecord objects."""
    metadata_path = dataset_root / "metadata.csv"
    track_records: Dict[int, List[TrackCropRecord]] = defaultdict(list)

    with open(str(metadata_path), "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rec = TrackCropRecord(
                track_id=int(row["track_id"]),
                frame=int(row["frame"]),
                timestamp=float(row["timestamp"]),
                bbox=(
                    int(float(row["bbox_x1"])),
                    int(float(row["bbox_y1"])),
                    int(float(row["bbox_x2"])),
                    int(float(row["bbox_y2"])),
                ),
                confidence=float(row["confidence"]),
                crop_width=int(row["crop_width"]),
                crop_height=int(row["crop_height"]),
                image_path=row["image_path"],
            )
            track_records[rec.track_id].append(rec)

    return track_records


def load_summaries(dataset_root: Path) -> List:
    """Load track_summary.csv and reconstruct TrackSummary objects."""
    from app.dataset.dataset_builder import TrackSummary

    summary_path = dataset_root / "track_summary.csv"
    summaries = []
    with open(str(summary_path), "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            s = TrackSummary(
                track_id=int(row["track_id"]),
                first_frame=int(row["first_frame"]),
                last_frame=int(row["last_frame"]),
                total_frames=int(row["total_frames"]),
                saved_images=int(row["saved_images"]),
                average_confidence=float(row["average_confidence"]),
            )
            summaries.append(s)
    return summaries


def main():
    dataset_root = Path("datasets/person_classifier")
    print("=" * 80)
    print("TRACK QUALITY VALIDATOR")
    print("=" * 80)

    # Load data
    print("\n[1] Loading metadata...")
    track_records = load_metadata(dataset_root)
    print(f"    Loaded {sum(len(v) for v in track_records.values())} records across {len(track_records)} tracks")

    print("\n[2] Loading summaries...")
    summaries = load_summaries(dataset_root)
    print(f"    Loaded {len(summaries)} track summaries")

    # Run validator
    print("\n[3] Running track validation...")
    validator = TrackValidator(
        dataset_root=dataset_root,
        hist_threshold=0.5,
        embedding_threshold=0.4,
        bbox_change_threshold=0.5,
        min_frames=5,
        identity_score_threshold=0.5,
    )

    results = validator.validate_all(track_records, summaries)

    # Print summary
    accepted = [r for r in results if not r.rejected]
    rejected = [r for r in results if r.rejected]
    print(f"\n    Total tracks: {len(results)}")
    print(f"    Accepted: {len(accepted)}")
    print(f"    Rejected: {len(rejected)}")

    if rejected:
        print("\n    Rejected tracks:")
        for r in sorted(rejected, key=lambda x: x.track_id):
            print(f"      Track {r.track_id:04d}: score={r.identity_score:.3f}, reason={r.reason}")

    # Write quality report
    print("\n[4] Writing quality report...")
    report_path = dataset_root / "track_quality_report.csv"
    validator.write_quality_report(results, report_path)
    print(f"    Report saved to: {report_path}")

    # Generate contact sheets for rejected tracks
    print("\n[5] Generating contact sheets for rejected tracks...")
    validator.generate_rejected_contact_sheets(results, track_records)

    # Move rejected tracks
    print("\n[6] Moving rejected tracks...")
    validator.move_rejected_tracks(results)

    print("\n" + "=" * 80)
    print("VALIDATION COMPLETE")
    print("=" * 80)
    print(f"\nAccepted tracks: {len(accepted)} → ready for training")
    print(f"Rejected tracks: {len(rejected)} → moved to rejected_tracks/")
    print(f"Quality report: {report_path}")
    print(f"Contact sheets: {dataset_root / 'rejected_tracks'}/track_XXXX_contact.jpg")


if __name__ == "__main__":
    main()