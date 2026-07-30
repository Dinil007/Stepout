#!/usr/bin/env python
"""Interactive labeling tool for person classification dataset.

Opens each track folder, displays preview.jpg, and lets the user assign a class.

Key bindings:
    1 → TEAM_A
    2 → TEAM_B
    3 → REFEREE
    4 → COACH
    s → Skip (leave unlabeled)
    q → Quit (save progress and exit)
    ← → Previous track
    → → Next track
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Dict, Optional

import cv2
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.classification.config import CLASS_NAMES, LABEL_KEYS, DatasetConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("label_tracks")


def load_labels(labels_path: Path) -> Dict[str, str]:
    """Load existing labels from JSON file."""
    if labels_path.exists():
        with open(labels_path, "r") as f:
            return json.load(f)
    return {}


def save_labels(labels: Dict[str, str], labels_path: Path) -> None:
    """Save labels to JSON file."""
    labels_path.parent.mkdir(parents=True, exist_ok=True)
    with open(labels_path, "w") as f:
        json.dump(labels, f, indent=2)
    logger.info(f"Saved {len(labels)} labels to {labels_path}")


def get_track_folders(raw_dir: Path) -> list[Path]:
    """Get sorted list of track folders."""
    if not raw_dir.exists():
        logger.error(f"Raw directory not found: {raw_dir}")
        return []
    folders = sorted(
        [d for d in raw_dir.iterdir() if d.is_dir() and d.name.startswith("track_")]
    )
    logger.info(f"Found {len(folders)} track folders")
    return folders


def get_preview_image(track_path: Path) -> Optional[np.ndarray]:
    """Get preview image for a track folder.

    Tries preview.jpg first, then the first available image.
    """
    preview_path = track_path / "preview.jpg"
    if preview_path.exists():
        img = cv2.imread(str(preview_path))
        if img is not None:
            return img

    # Fall back to first image
    images = sorted(track_path.glob("*.jpg")) + sorted(track_path.glob("*.png"))
    if images:
        img = cv2.imread(str(images[0]))
        if img is not None:
            return img

    return None


def get_image_count(track_path: Path) -> int:
    """Count images in a track folder."""
    return len(list(track_path.glob("*.jpg")) + list(track_path.glob("*.png")))


def create_info_overlay(
    image: np.ndarray,
    track_id: str,
    image_count: int,
    current_label: Optional[str],
    total_tracks: int,
    current_idx: int,
) -> np.ndarray:
    """Create an overlay with track information."""
    overlay = image.copy()
    h, w = image.shape[:2]

    # Semi-transparent overlay at top
    cv2.rectangle(overlay, (0, 0), (w, 120), (0, 0, 0), -1)
    alpha = 0.6
    image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

    # Track info
    label_text = f"Label: {current_label}" if current_label else "Label: UNLABELED"
    label_color = (0, 255, 0) if current_label else (0, 0, 255)

    cv2.putText(
        image, f"Track: {track_id} ({current_idx + 1}/{total_tracks})",
        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2,
    )
    cv2.putText(
        image, f"Images: {image_count}",
        (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2,
    )
    cv2.putText(
        image, label_text,
        (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, label_color, 2,
    )

    # Key bindings at bottom
    cv2.rectangle(image, (0, h - 40), (w, h), (0, 0, 0), -1)
    cv2.putText(
        image,
        "1:TEAM_A  2:TEAM_B  3:REFEREE  4:COACH  s:Skip  q:Quit  ←/→:Navigate",
        (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1,
    )

    return image


def main():
    """Run the interactive labeling tool."""
    config = DatasetConfig()

    # Load existing labels
    labels = load_labels(config.labels_file)
    logger.info(f"Loaded {len(labels)} existing labels")

    # Get track folders
    track_folders = get_track_folders(config.raw_dir)
    if not track_folders:
        logger.error("No track folders found. Run dataset generation first.")
        sys.exit(1)

    total_tracks = len(track_folders)
    current_idx = 0

    # Find first unlabeled track to start
    for i, folder in enumerate(track_folders):
        if folder.name not in labels:
            current_idx = i
            break

    logger.info("Starting interactive labeling. Press 'q' to quit at any time.")
    print("\n" + "=" * 60)
    print("  PERSON CLASSIFICATION LABELING TOOL")
    print("=" * 60)
    print("  1 → TEAM_A")
    print("  2 → TEAM_B")
    print("  3 → REFEREE")
    print("  4 → COACH")
    print("  s → Skip")
    print("  q → Quit (saves progress)")
    print("  ← → Previous track")
    print("  → → Next track")
    print("=" * 60 + "\n")

    cv2.namedWindow("Label Tracks", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Label Tracks", 800, 600)

    while 0 <= current_idx < total_tracks:
        track_path = track_folders[current_idx]
        track_id = track_path.name
        current_label = labels.get(track_id)

        # Get preview image
        preview = get_preview_image(track_path)
        if preview is None:
            logger.warning(f"No preview image for {track_id}, skipping")
            current_idx += 1
            continue

        # Resize preview for display
        h, w = preview.shape[:2]
        max_display_h = 500
        if h > max_display_h:
            scale = max_display_h / h
            new_w = int(w * scale)
            preview = cv2.resize(preview, (new_w, max_display_h))

        image_count = get_image_count(track_path)

        # Create overlay
        display = create_info_overlay(
            preview, track_id, image_count,
            current_label, total_tracks, current_idx,
        )

        cv2.imshow("Label Tracks", display)
        key = cv2.waitKey(0) & 0xFF

        # Process key press
        if key == ord("q") or key == ord("Q"):
            logger.info("Quit requested. Saving labels...")
            break

        elif key == 81:  # Left arrow
            current_idx = max(0, current_idx - 1)

        elif key == 83:  # Right arrow
            current_idx = min(total_tracks - 1, current_idx + 1)

        elif key == ord("s") or key == ord("S"):
            # Skip - remove label if exists
            if track_id in labels:
                del labels[track_id]
                logger.info(f"Skipped {track_id}")
            current_idx += 1

        elif chr(key) in ("1", "2", "3", "4"):
            class_name = LABEL_KEYS[chr(key)]
            labels[track_id] = class_name
            logger.info(f"Labeled {track_id} → {class_name}")
            current_idx += 1

        else:
            logger.debug(f"Unknown key: {key}")

        # Auto-save every 10 labels
        if len(labels) % 10 == 0:
            save_labels(labels, config.labels_file)

    # Final save
    save_labels(labels, config.labels_file)
    cv2.destroyAllWindows()

    # Summary
    class_counts = {cls: 0 for cls in CLASS_NAMES}
    skipped = 0
    for tid, label in labels.items():
        if label in class_counts:
            class_counts[label] += 1
        else:
            skipped += 1

    print("\n" + "=" * 60)
    print("  LABELING SUMMARY")
    print("=" * 60)
    for cls, count in class_counts.items():
        print(f"  {cls}: {count} tracks")
    print(f"  Skipped: {skipped}")
    print(f"  Total labeled: {len(labels)}")
    print(f"  Total tracks: {total_tracks}")
    print("=" * 60)


if __name__ == "__main__":
    main()