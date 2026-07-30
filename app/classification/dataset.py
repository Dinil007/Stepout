"""Dataset preparation, quality filtering, splitting, and augmentation.

This module handles:
- Organizing raw track folders into train/val/test splits
- Quality filtering (blur, size, brightness, duplicates)
- Track-aware splitting (never split one track across splits)
- Data augmentation for training
"""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import cv2
import numpy as np

from app.classification.config import CLASS_NAMES, DatasetConfig

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# STEP 1: Dataset Preparation
# ──────────────────────────────────────────────


def prepare_dataset_structure(config: DatasetConfig) -> None:
    """Create the prepared directory structure for train/val/test splits."""
    splits = ["train", "val", "test"]
    for split in splits:
        for class_name in CLASS_NAMES:
            path = config.prepared_dir / split / class_name
            path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created directory: {path}")

    config.rejected_dir.mkdir(parents=True, exist_ok=True)
    config.metadata_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Dataset directory structure created.")


def get_track_folders(raw_dir: Path) -> List[Path]:
    """Get sorted list of track folders from the raw directory."""
    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw directory not found: {raw_dir}")
    folders = sorted(
        [d for d in raw_dir.iterdir() if d.is_dir() and d.name.startswith("track_")]
    )
    logger.info(f"Found {len(folders)} track folders in {raw_dir}")
    return folders


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


# ──────────────────────────────────────────────
# STEP 3: Dataset Quality Filter
# ──────────────────────────────────────────────


class QualityFilter:
    """Filters low-quality crops from the dataset."""

    def __init__(self, config: DatasetConfig):
        self.config = config
        self.stats: Dict[str, int] = {
            "total": 0,
            "rejected_blurry": 0,
            "rejected_tiny": 0,
            "rejected_low_confidence": 0,
            "rejected_dark": 0,
            "rejected_aspect_ratio": 0,
            "rejected_duplicate": 0,
            "accepted": 0,
        }

    def check_blurry(self, image: np.ndarray) -> bool:
        """Check if image is blurry using Laplacian variance."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        return laplacian_var < self.config.max_blur_threshold

    def check_size(self, image: np.ndarray) -> bool:
        """Check if image meets minimum size requirements."""
        h, w = image.shape[:2]
        return w >= self.config.min_width and h >= self.config.min_height

    def check_brightness(self, image: np.ndarray) -> bool:
        """Check if image is sufficiently bright."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        mean_brightness = np.mean(gray)
        return mean_brightness >= self.config.min_brightness

    def check_aspect_ratio(self, image: np.ndarray) -> bool:
        """Check if aspect ratio is within acceptable range."""
        h, w = image.shape[:2]
        if h == 0:
            return False
        ratio = w / h
        return self.config.min_aspect_ratio <= ratio <= self.config.max_aspect_ratio

    def check_duplicate(
        self, image: np.ndarray, existing_hashes: Set[int]
    ) -> bool:
        """Check for near-duplicate images using perceptual hashing (dHash)."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (9, 8))
        diff = resized[:, 1:] > resized[:, :-1]
        hash_val = hash(diff.tobytes())

        # Simple exact match on hash
        if hash_val in existing_hashes:
            return True
        return False

    def filter_crop(
        self,
        image: np.ndarray,
        confidence: float = 1.0,
        existing_hashes: Optional[Set[int]] = None,
    ) -> Tuple[bool, str]:
        """Filter a single crop. Returns (is_accepted, reason)."""
        self.stats["total"] += 1

        # Size check
        if not self.check_size(image):
            self.stats["rejected_tiny"] += 1
            return False, "tiny"

        # Aspect ratio check
        if not self.check_aspect_ratio(image):
            self.stats["rejected_aspect_ratio"] += 1
            return False, "aspect_ratio"

        # Brightness check
        if not self.check_brightness(image):
            self.stats["rejected_dark"] += 1
            return False, "dark"

        # Blur check
        if self.check_blurry(image):
            self.stats["rejected_blurry"] += 1
            return False, "blurry"

        # Confidence check
        if confidence < self.config.min_confidence:
            self.stats["rejected_low_confidence"] += 1
            return False, "low_confidence"

        # Duplicate check
        if existing_hashes is not None:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(gray, (9, 8))
            diff = resized[:, 1:] > resized[:, :-1]
            hash_val = hash(diff.tobytes())
            if hash_val in existing_hashes:
                self.stats["rejected_duplicate"] += 1
                return False, "duplicate"
            existing_hashes.add(hash_val)

        self.stats["accepted"] += 1
        return True, "accepted"

    def filter_track(
        self,
        track_path: Path,
        track_id: str,
        rejected_dir: Path,
        existing_hashes: Optional[Set[int]] = None,
    ) -> List[Path]:
        """Filter all images in a track folder. Returns list of accepted paths."""
        accepted_paths: List[Path] = []
        images = sorted(track_path.glob("*.jpg")) + sorted(track_path.glob("*.png"))

        if not images:
            logger.warning(f"Track {track_id}: no images found")
            return accepted_paths

        for img_path in images:
            image = cv2.imread(str(img_path))
            if image is None:
                logger.warning(f"Could not read {img_path}, skipping")
                continue

            accepted, reason = self.filter_crop(image, existing_hashes=existing_hashes)
            if accepted:
                accepted_paths.append(img_path)
            else:
                # Move rejected to rejected directory
                reject_subdir = rejected_dir / reason
                reject_subdir.mkdir(parents=True, exist_ok=True)
                dest = reject_subdir / f"{track_id}_{img_path.name}"
                shutil.move(str(img_path), str(dest))
                logger.debug(f"Rejected {img_path.name}: {reason}")

        logger.info(
            f"Track {track_id}: {len(accepted_paths)}/{len(images)} accepted"
        )
        return accepted_paths

    def generate_report(self) -> str:
        """Generate quality filter report."""
        lines = [
            "# Quality Filter Report",
            "",
            "## Summary",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Total crops examined | {self.stats['total']} |",
            f"| Accepted | {self.stats['accepted']} |",
            f"| Rejected (blurry) | {self.stats['rejected_blurry']} |",
            f"| Rejected (tiny) | {self.stats['rejected_tiny']} |",
            f"| Rejected (low confidence) | {self.stats['rejected_low_confidence']} |",
            f"| Rejected (dark) | {self.stats['rejected_dark']} |",
            f"| Rejected (aspect ratio) | {self.stats['rejected_aspect_ratio']} |",
            f"| Rejected (duplicate) | {self.stats['rejected_duplicate']} |",
            "",
            "## Rejection Rate",
            f"{(self.stats['total'] - self.stats['accepted']) / max(self.stats['total'], 1) * 100:.1f}%",
        ]
        return "\n".join(lines)


# ──────────────────────────────────────────────
# STEP 4: Dataset Split (Track-Aware)
# ──────────────────────────────────────────────


def split_tracks(
    track_ids: List[str],
    labels: Dict[str, str],
    config: DatasetConfig,
    seed: int = 42,
) -> Dict[str, List[str]]:
    """Split tracks into train/val/test while keeping each track intact.

    Args:
        track_ids: List of track folder names (e.g., ['track_0001', ...])
        labels: Dict mapping track_id -> class_name
        config: Dataset configuration with split ratios
        seed: Random seed for reproducibility

    Returns:
        Dict with keys 'train', 'val', 'test' mapping to lists of track IDs
    """
    import random

    rng = random.Random(seed)

    # Group tracks by class
    class_tracks: Dict[str, List[str]] = {cls: [] for cls in CLASS_NAMES}
    for track_id in track_ids:
        label = labels.get(track_id, "SKIP")
        if label in class_tracks:
            class_tracks[label].append(track_id)

    splits: Dict[str, List[str]] = {"train": [], "val": [], "test": []}

    for class_name, tracks in class_tracks.items():
        if not tracks:
            continue

        rng.shuffle(tracks)
        n = len(tracks)
        n_train = max(1, int(round(n * config.train_ratio)))
        n_val = max(0, int(round(n * config.val_ratio)))
        n_test = n - n_train - n_val

        # Adjust to ensure all tracks are assigned
        if n_test < 0:
            n_test = 0
            n_train = n - n_val

        splits["train"].extend(tracks[:n_train])
        splits["val"].extend(tracks[n_train : n_train + n_val])
        splits["test"].extend(tracks[n_train + n_val :])

    # Log split sizes
    for split_name, split_tracks in splits.items():
        class_counts = {cls: 0 for cls in CLASS_NAMES}
        for tid in split_tracks:
            label = labels.get(tid, "SKIP")
            if label in class_counts:
                class_counts[label] += 1
        logger.info(
            f"Split '{split_name}': {len(split_tracks)} tracks, "
            f"class distribution: {class_counts}"
        )

    return splits


def copy_tracks_to_split(
    split: str,
    track_ids: List[str],
    raw_dir: Path,
    prepared_dir: Path,
    labels: Dict[str, str],
    quality_filter: QualityFilter,
    rejected_dir: Path,
) -> None:
    """Copy filtered crops from track folders to the prepared split directory.

    Each image is renamed as {track_id}_{original_filename} to preserve origin.
    """
    existing_hashes: Set[int] = set()

    for track_id in track_ids:
        track_path = raw_dir / track_id
        if not track_path.exists():
            logger.warning(f"Track folder not found: {track_path}")
            continue

        label = labels.get(track_id)
        if label not in CLASS_NAMES:
            logger.warning(f"Track {track_id} has invalid label '{label}', skipping")
            continue

        dest_dir = prepared_dir / split / label

        images = sorted(track_path.glob("*.jpg")) + sorted(track_path.glob("*.png"))
        for img_path in images:
            # Skip non-image files like preview.jpg, contact_sheet etc.
            if img_path.stem in ("preview", "contact_sheet"):
                continue

            image = cv2.imread(str(img_path))
            if image is None:
                continue

            accepted, reason = quality_filter.filter_crop(
                image, existing_hashes=existing_hashes
            )
            if accepted:
                new_name = f"{track_id}_{img_path.name}"
                dest_path = dest_dir / new_name
                shutil.copy2(str(img_path), str(dest_path))
            else:
                reject_subdir = rejected_dir / reason
                reject_subdir.mkdir(parents=True, exist_ok=True)
                dest = reject_subdir / f"{track_id}_{img_path.name}"
                shutil.copy2(str(img_path), str(dest))

        logger.info(
            f"Copied track {track_id} -> {split}/{label} "
            f"({len(list(dest_dir.glob(f'{track_id}_*')))} images)"
        )


# ──────────────────────────────────────────────
# STEP 5: Data Augmentation (Training Only)
# ──────────────────────────────────────────────


class TrainingAugmentation:
    """Data augmentation pipeline for training.

    Applies:
    - Random horizontal flip
    - Random brightness/contrast
    - Random rotation ±8°
    - Random blur
    - Random color jitter
    - Random erasing
    """

    def __init__(self, config: DatasetConfig):
        self.config = config

    def __call__(self, image: np.ndarray) -> np.ndarray:
        """Apply augmentations to a single image.

        Args:
            image: Input BGR image (H, W, 3)

        Returns:
            Augmented BGR image
        """
        import random as rng

        # Random horizontal flip
        if rng.random() < self.config.hflip_prob:
            image = cv2.flip(image, 1)

        # Random brightness and contrast
        if rng.random() < 0.5:
            brightness = rng.uniform(*self.config.brightness_range)
            contrast = rng.uniform(*self.config.contrast_range)
            image = cv2.convertScaleAbs(image, alpha=contrast, beta=(brightness - 1.0) * 127.0)

        # Random rotation
        if rng.random() < 0.5:
            h, w = image.shape[:2]
            angle = rng.uniform(-self.config.rotation_deg, self.config.rotation_deg)
            center = (w // 2, h // 2)
            matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            image = cv2.warpAffine(
                image, matrix, (w, h),
                borderMode=cv2.BORDER_REFLECT,
                flags=cv2.INTER_LINEAR,
            )

        # Random blur
        if rng.random() < 0.3:
            k = self.config.blur_kernel
            if rng.random() < 0.5:
                image = cv2.GaussianBlur(image, (k, k), 0)
            else:
                image = cv2.blur(image, (k, k))

        # Random color jitter (HSV shift)
        if rng.random() < 0.5:
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
            hsv[:, :, 0] += rng.uniform(-10, 10)  # Hue
            hsv[:, :, 1] *= rng.uniform(0.8, 1.2)  # Saturation
            hsv[:, :, 2] *= rng.uniform(0.8, 1.2)  # Value
            hsv = np.clip(hsv, 0, 255).astype(np.uint8)
            image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        # Random erasing
        if rng.random() < 0.3:
            h, w = image.shape[:2]
            erase_h = int(h * rng.uniform(*self.config.erase_scale))
            erase_w = int(erase_h * rng.uniform(*self.config.erase_ratio))
            erase_w = min(erase_w, w)
            erase_h = min(erase_h, h)
            x = rng.randint(0, max(1, w - erase_w))
            y = rng.randint(0, max(1, h - erase_h))
            # Fill with random color
            color = tuple(int(c) for c in rng.randint(0, 255, 3))
            cv2.rectangle(image, (x, y), (x + erase_w, y + erase_h), color, -1)

        return image


# ──────────────────────────────────────────────
# Main Orchestration
# ──────────────────────────────────────────────


def prepare_and_split_dataset(config: Optional[DatasetConfig] = None) -> QualityFilter:
    """Run the full dataset preparation pipeline.

    Args:
        config: Optional DatasetConfig. Uses defaults if not provided.

    Returns:
        QualityFilter instance with statistics.
    """
    if config is None:
        config = DatasetConfig()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    logger.info("Starting dataset preparation pipeline")

    # Create directory structure
    prepare_dataset_structure(config)

    # Load existing labels
    labels = load_labels(config.labels_file)
    logger.info(f"Loaded {len(labels)} labels from {config.labels_file}")

    # Get track folders
    track_folders = get_track_folders(config.raw_dir)
    track_ids = [tf.name for tf in track_folders]
    logger.info(f"Found {len(track_ids)} track folders")

    # Quality filter
    quality_filter = QualityFilter(config)

    # Split tracks (track-aware)
    splits = split_tracks(track_ids, labels, config)

    # Copy filtered images to each split
    for split_name, split_track_ids in splits.items():
        if not split_track_ids:
            logger.warning(f"No tracks for split '{split_name}'")
            continue
        copy_tracks_to_split(
            split=split_name,
            track_ids=split_track_ids,
            raw_dir=config.raw_dir,
            prepared_dir=config.prepared_dir,
            labels=labels,
            quality_filter=quality_filter,
            rejected_dir=config.rejected_dir,
        )

    # Generate and save quality report
    report = quality_filter.generate_report()
    report_path = config.metadata_dir / "quality_report.md"
    with open(report_path, "w") as f:
        f.write(report)
    logger.info(f"Quality report saved to {report_path}")

    # Log final statistics
    for split_name, split_track_ids in splits.items():
        class_counts = {}
        for tid in split_track_ids:
            label = labels.get(tid, "unknown")
            class_counts[label] = class_counts.get(label, 0) + 1
        logger.info(f"Split '{split_name}': {class_counts}")

    return quality_filter