"""Dataset analysis and quality report generator.

This script analyzes the person classification dataset and generates:
- dataset_analysis.md
- dataset_quality_report.md
"""
from __future__ import annotations

import csv
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

import cv2
import numpy as np

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.classification.config import CLASS_NAMES

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def compute_image_hash(image_path: Path) -> str:
    """Compute perceptual hash of an image for duplicate detection."""
    try:
        img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            return ""
        img = cv2.resize(img, (8, 8))
        avg = np.mean(img)
        bits = img > avg
        return "".join("1" if b else "0" for b in bits.flatten())
    except Exception:
        return ""


def analyze_dataset(
    dataset_root: Path = Path("datasets/person_classifier"),
    output_dir: Path = Path("datasets/person_classifier/metadata"),
) -> None:
    """Run full dataset analysis and generate reports."""

    raw_dir = dataset_root / "raw"
    rejected_dir = dataset_root / "rejected_tracks"

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load labels
    labels_csv = dataset_root / "labels.csv"
    track_labels: Dict[str, str] = {}
    track_status: Dict[str, str] = {}
    if labels_csv.exists():
        with open(labels_csv, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                track_id = row.get("track_id", "").strip()
                label = row.get("label", "").strip()
                status = row.get("status", "").strip()
                if track_id:
                    track_labels[track_id] = label if label else "UNLABELED"
                    track_status[track_id] = status if status else "UNLABELED"

    logger.info(f"Loaded {len(track_labels)} track labels")

    # Gather all track folders
    track_folders = sorted(
        [d for d in raw_dir.iterdir() if d.is_dir() and d.name.startswith("track_")]
    )

    # Rejected tracks
    rejected_tracks = []
    if rejected_dir.exists():
        rejected_tracks = [
            d.name for d in rejected_dir.iterdir() if d.is_dir() and d.name.startswith("track_")
        ]

    # Analysis
    total_tracks = len(track_folders)
    labeled_tracks = {tid: lbl for tid, lbl in track_labels.items() if lbl != "UNLABELED"}
    unlabeled_tracks = {tid: lbl for tid, lbl in track_labels.items() if lbl == "UNLABELED"}

    class_track_counts: Dict[str, int] = defaultdict(int)
    for tid, lbl in labeled_tracks.items():
        class_track_counts[lbl] += 1

    # Image statistics
    all_images: List[Path] = []
    resolutions: List[Tuple[int, int]] = []
    file_sizes: List[int] = []
    corrupt_images: List[str] = []
    empty_files: List[str] = []
    wrong_format: List[str] = []
    tiny_images: List[str] = []
    duplicate_hashes: Dict[str, List[str]] = defaultdict(list)

    for track_folder in track_folders:
        track_id = track_folder.name
        if track_id in rejected_tracks:
            continue

        track_id_num = track_id.replace("track_", "")
        label = track_labels.get(track_id_num, "UNLABELED")
        if label == "UNLABELED":
            continue

        for img_path in track_folder.glob("*.jpg"):
            all_images.append(img_path)
            try:
                size = img_path.stat().st_size
                file_sizes.append(size)

                if size == 0:
                    empty_files.append(str(img_path))
                    continue

                img = cv2.imread(str(img_path))
                if img is None:
                    corrupt_images.append(str(img_path))
                    continue

                h, w = img.shape[:2]
                resolutions.append((w, h))

                if w < 32 or h < 64:
                    tiny_images.append(str(img_path))

                img_hash = compute_image_hash(img_path)
                if img_hash:
                    duplicate_hashes[img_hash].append(str(img_path))

            except Exception as e:
                corrupt_images.append(f"{img_path} ({e})")

    # Duplicates
    true_duplicates = {h: paths for h, paths in duplicate_hashes.items() if len(paths) > 1}

    # Build analysis lines
    analysis_lines = []
    analysis_lines.append("# Dataset Analysis")
    analysis_lines.append("")
    analysis_lines.append("## Overview")
    analysis_lines.append("")
    analysis_lines.append(f"- **Total tracks (raw)**: {total_tracks}")
    analysis_lines.append(f"- **Rejected tracks**: {len(rejected_tracks)}")
    analysis_lines.append(f"- **Labeled tracks**: {len(labeled_tracks)}")
    analysis_lines.append(f"- **Unlabeled tracks**: {len(unlabeled_tracks)}")
    analysis_lines.append(f"- **Total images analyzed**: {len(all_images)}")
    analysis_lines.append("")
    analysis_lines.append("## Per-Class Track Count")
    analysis_lines.append("")
    analysis_lines.append("| Class | Tracks |")
    analysis_lines.append("|-------|--------|")
    for cls in CLASS_NAMES:
        analysis_lines.append(f"| {cls} | {class_track_counts.get(cls, 0)} |")
    analysis_lines.append(f"| UNLABELED | {len(unlabeled_tracks)} |")

    analysis_lines.append("")
    analysis_lines.append("## Image Statistics")
    analysis_lines.append("")
    analysis_lines.append(f"- **Total images**: {len(all_images)}")
    analysis_lines.append(f"- **Corrupted images**: {len(corrupt_images)}")
    analysis_lines.append(f"- **Empty files**: {len(empty_files)}")
    analysis_lines.append(f"- **Wrong format**: {len(wrong_format)}")
    analysis_lines.append(f"- **Very small images**: {len(tiny_images)}")
    analysis_lines.append(f"- **Duplicate images**: {sum(len(v) - 1 for v in true_duplicates.values())}")
    analysis_lines.append("")

    if resolutions:
        widths, heights = zip(*resolutions)
        avg_w = float(np.mean(widths))
        avg_h = float(np.mean(heights))
        min_w = int(np.min(widths))
        min_h = int(np.min(heights))
        max_w = int(np.max(widths))
        max_h = int(np.max(heights))

        analysis_lines.append("## Resolution Statistics")
        analysis_lines.append("")
        analysis_lines.append("| Metric | Width | Height |")
        analysis_lines.append("|--------|-------|--------|")
        analysis_lines.append(f"| Average | {avg_w:.1f} | {avg_h:.1f} |")
        analysis_lines.append(f"| Minimum | {min_w} | {min_h} |")
        analysis_lines.append(f"| Maximum | {max_w} | {max_h} |")
        analysis_lines.append("")

    if file_sizes:
        analysis_lines.append("## File Size Statistics")
        analysis_lines.append("")
        analysis_lines.append("| Metric | Size (KB) |")
        analysis_lines.append("|--------|-----------|")
        analysis_lines.append(f"| Average | {np.mean(file_sizes) / 1024:.1f} |")
        analysis_lines.append(f"| Minimum | {np.min(file_sizes) / 1024:.1f} |")
        analysis_lines.append(f"| Maximum | {np.max(file_sizes) / 1024:.1f} |")
        analysis_lines.append("")

    # Class imbalance
    analysis_lines.append("## Class Imbalance Analysis")
    analysis_lines.append("")
    analysis_lines.append("| Class | Tracks | Percentage |")
    analysis_lines.append("|-------|--------|------------|")
    total_labeled = sum(class_track_counts.values())
    for cls in CLASS_NAMES:
        count = class_track_counts.get(cls, 0)
        pct = count / max(total_labeled, 1) * 100
        analysis_lines.append(f"| {cls} | {count} | {pct:.1f}% |")

    analysis_lines.append("")
    analysis_lines.append("## Dataset Split Recommendations")
    analysis_lines.append("")
    analysis_lines.append("Based on current labeled tracks:")
    analysis_lines.append("")

    # Calculate recommended splits
    for cls in CLASS_NAMES:
        count = class_track_counts.get(cls, 0)
        if count == 0:
            continue
        n_train = max(1, int(count * 0.7))
        n_val = max(1, int(count * 0.15))
        n_test = count - n_train - n_val
        if n_test < 0:
            n_test = 0
        analysis_lines.append(f"- **{cls}**: {n_train} train / {n_val} val / {n_test} test")

    analysis_lines.append("")
    analysis_path = output_dir / "dataset_analysis.md"
    with open(analysis_path, "w") as f:
        f.write("\n".join(analysis_lines))
    logger.info(f"Dataset analysis saved to {analysis_path}")

    # Quality report
    quality_lines = []
    quality_lines.append("# Dataset Quality Report")
    quality_lines.append("")
    quality_lines.append("## Summary")
    quality_lines.append("")
    quality_lines.append(f"- **Total raw tracks**: {total_tracks}")
    quality_lines.append(f"- **Labeled tracks**: {len(labeled_tracks)}")
    quality_lines.append(f"- **Unlabeled tracks**: {len(unlabeled_tracks)}")
    quality_lines.append(f"- **Rejected tracks**: {len(rejected_tracks)}")
    quality_lines.append(f"- **Total images examined**: {len(all_images)}")
    quality_lines.append("")
    quality_lines.append("## Quality Issues")
    quality_lines.append("")

    if corrupt_images:
        quality_lines.append("### Corrupted Images")
        quality_lines.append("")
        quality_lines.append(f"Found {len(corrupt_images)} corrupted images:")
        quality_lines.append("")
        for p in corrupt_images[:20]:
            quality_lines.append(f"- `{p}`")
        if len(corrupt_images) > 20:
            quality_lines.append(f"- ... and {len(corrupt_images) - 20} more")

    if empty_files:
        quality_lines.append("")
        quality_lines.append("### Empty Files")
        quality_lines.append("")
        quality_lines.append(f"Found {len(empty_files)} empty files:")
        quality_lines.append("")
        for p in empty_files[:20]:
            quality_lines.append(f"- `{p}`")
        if len(empty_files) > 20:
            quality_lines.append(f"- ... and {len(empty_files) - 20} more")

    if tiny_images:
        quality_lines.append("")
        quality_lines.append("### Very Small Images")
        quality_lines.append("")
        quality_lines.append(f"Found {len(tiny_images)} images smaller than 32x64:")
        quality_lines.append("")
        for p in tiny_images[:20]:
            quality_lines.append(f"- `{p}`")
        if len(tiny_images) > 20:
            quality_lines.append(f"- ... and {len(tiny_images) - 20} more")

    if wrong_format:
        quality_lines.append("")
        quality_lines.append("### Wrong Format")
        quality_lines.append("")
        quality_lines.append(f"Found {len(wrong_format)} files with wrong format:")
        quality_lines.append("")
        for p in wrong_format[:20]:
            quality_lines.append(f"- `{p}`")
        if len(wrong_format) > 20:
            quality_lines.append(f"- ... and {len(wrong_format) - 20} more")

    if true_duplicates:
        dup_count = sum(len(v) - 1 for v in true_duplicates.values())
        quality_lines.append("")
        quality_lines.append("### Duplicate Images")
        quality_lines.append("")
        quality_lines.append(f"Found {dup_count} duplicate images in {len(true_duplicates)} groups:")
        quality_lines.append("")
        for h, paths in list(true_duplicates.items())[:10]:
            quality_lines.append(f"- Hash `{h[:16]}...`: {len(paths)} copies")
            for p in paths:
                quality_lines.append(f"  - `{p}`")
        if len(true_duplicates) > 10:
            quality_lines.append(f"- ... and {len(true_duplicates) - 10} more groups")

    quality_lines.append("")
    quality_lines.append("## Recommendations")
    quality_lines.append("")
    quality_lines.append("1. Remove or relabel corrupted images")
    quality_lines.append("2. Delete empty files")
    quality_lines.append("3. Remove very small images")
    quality_lines.append("4. Remove duplicate images, keeping only one copy per group")
    quality_lines.append("5. Verify labels for all tracks before training")
    quality_lines.append("")

    quality_path = output_dir / "dataset_quality_report.md"
    with open(quality_path, "w") as f:
        f.write("\n".join(quality_lines))
    logger.info(f"Quality report saved to {quality_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("DATASET ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"Total tracks: {total_tracks}")
    print(f"Labeled: {len(labeled_tracks)}, Unlabeled: {len(unlabeled_tracks)}")
    print("Class distribution:")
    for cls in CLASS_NAMES:
        print(f"  {cls}: {class_track_counts.get(cls, 0)} tracks")
    print(f"\nTotal images: {len(all_images)}")
    print(f"Corrupted: {len(corrupt_images)}")
    print(f"Empty: {len(empty_files)}")
    print(f"Tiny: {len(tiny_images)}")
    print(f"Duplicates: {sum(len(v) - 1 for v in true_duplicates.values())}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    analyze_dataset()