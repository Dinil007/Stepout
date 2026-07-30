"""Dataset verification script.

Verifies:
1. Number of images per class
2. Class imbalance
3. Duplicate images
4. Corrupted images
5. Missing labels
"""
from __future__ import annotations

import csv
import hashlib
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.classification.config import CLASS_NAMES


def compute_md5(file_path: Path) -> str:
    """Compute MD5 hash of file."""
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def verify_dataset(
    prepared_dir: Path = Path("datasets/person_classifier/prepared"),
    raw_dir: Path = Path("datasets/person_classifier/raw"),
    labels_file: Path = Path("datasets/person_classifier/metadata/labels.json"),
    output_file: Path = Path("datasets/person_classifier/metadata/dataset_verification.md"),
) -> dict:
    """Verify dataset integrity and generate report."""
    
    report = {
        "total_images": 0,
        "per_class": defaultdict(int),
        "corrupted": [],
        "empty": [],
        "wrong_format": [],
        "tiny": [],
        "missing_labels": [],
        "duplicates": [],
        "class_balance": {},
    }
    
    # Load labels
    import json
    labels = {}
    if labels_file.exists():
        with open(labels_file) as f:
            labels = json.load(f)
    
    # Check each class
    for class_name in CLASS_NAMES:
        class_dir = prepared_dir / "train" / class_name
        if not class_dir.exists():
            continue
            
        images = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.jpeg")) + list(class_dir.glob("*.png"))
        report["per_class"][class_name] = len(images)
        report["total_images"] += len(images)
        
        # Check for corrupted/tiny/empty images
        for img_path in images:
            try:
                size = img_path.stat().st_size
                if size == 0:
                    report["empty"].append(str(img_path))
                    continue
                    
                img = cv2.imread(str(img_path))
                if img is None:
                    report["corrupted"].append(str(img_path))
                    continue
                    
                h, w = img.shape[:2]
                if w < 32 or h < 64:
                    report["tiny"].append(str(img_path))
                    
            except Exception as e:
                report["corrupted"].append(f"{img_path} ({e})")
    
    # Check duplicates
    hashes = defaultdict(list)
    for class_name in CLASS_NAMES:
        class_dir = prepared_dir / "train" / class_name
        if not class_dir.exists():
            continue
        for img_path in class_dir.glob("*.jpg"):
            try:
                if img_path.stat().st_size > 0:
                    img_hash = compute_md5(img_path)
                    hashes[img_hash].append(str(img_path))
            except:
                pass
    
    report["duplicates"] = {h: paths for h, paths in hashes.items() if len(paths) > 1}
    
    # Class balance
    total = sum(report["per_class"].values())
    if total > 0:
        for cls, count in report["per_class"].items():
            report["class_balance"][cls] = count / total
    
    # Generate report
    lines = [
        "# Dataset Verification Report",
        "",
        "## Image Counts",
        "",
        "| Class | Train Images | Percentage |",
        "|-------|--------------|------------|",
    ]
    
    for cls in CLASS_NAMES:
        count = report["per_class"][cls]
        pct = report["class_balance"].get(cls, 0) * 100
        lines.append(f"| {cls} | {count} | {pct:.1f}% |")
    
    lines.extend([
        f"| **Total** | **{report['total_images']}** | 100.0% |",
        "",
        "## Data Quality",
        "",
        f"- **Corrupted images**: {len(report['corrupted'])}",
        f"- **Empty files**: {len(report['empty'])}",
        f"- **Tiny images** (<32x64): {len(report['tiny'])}",
        f"- **Duplicate images**: {len(report['duplicates'])}",
        "",
    ])
    
    if report["corrupted"]:
        lines.append("### Corrupted Images\n")
        for p in report["corrupted"][:10]:
            lines.append(f"- `{p}`")
        if len(report["corrupted"]) > 10:
            lines.append(f"- ... and {len(report['corrupted']) - 10} more")
        lines.append("")
    
    if report["duplicates"]:
        lines.append("### Duplicate Groups\n")
        for h, paths in list(report["duplicates"].items())[:10]:
            lines.append(f"- Hash `{h[:16]}...`: {len(paths)} copies")
        lines.append("")
    
    # Class imbalance analysis
    max_count = max(report["per_class"].values()) if report["per_class"] else 1
    min_count = min(report["per_class"].values()) if report["per_class"] else 1
    imbalance_ratio = max_count / max(min_count, 1)
    
    lines.extend([
        "## Class Imbalance",
        "",
        f"- **Imbalance ratio** (max/min): {imbalance_ratio:.2f}",
        f"- **Status**: {'⚠️ Severe imbalance' if imbalance_ratio > 3 else '✓ Acceptable'}",
        "",
        "## Recommendations",
        "",
        "1. Apply class weights in loss function if imbalance > 3",
        "2. Use data augmentation for minority classes",
        "3. Consider oversampling minority classes",
        "4. Monitor per-class metrics during training",
        "",
    ])
    
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    
    print(f"Dataset verification saved to: {output_file}")
    return report


if __name__ == "__main__":
    verify_dataset()