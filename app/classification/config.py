"""Configuration for person classification pipeline."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple


# Class labels
CLASS_NAMES = ["TEAM_A", "TEAM_B", "REFEREE", "COACH"]
CLASS_MAP = {name: idx for idx, name in enumerate(CLASS_NAMES)}
NUM_CLASSES = len(CLASS_NAMES)

# Labeling key bindings
LABEL_KEYS = {
    "1": "TEAM_A",
    "2": "TEAM_B",
    "3": "REFEREE",
    "4": "COACH",
    "s": "SKIP",
    "S": "SKIP",
    "q": "QUIT",
    "Q": "QUIT",
}


@dataclass
class DatasetConfig:
    """Configuration for dataset paths and structure."""

    raw_dir: Path = Path("datasets/person_classifier/raw")
    prepared_dir: Path = Path("datasets/person_classifier/prepared")
    rejected_dir: Path = Path("datasets/person_classifier/rejected")
    metadata_dir: Path = Path("datasets/person_classifier/metadata")
    labels_file: Path = Path("datasets/person_classifier/metadata/labels.json")

    # Split ratios
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15

    # Quality filter thresholds
    min_width: int = 32
    min_height: int = 64
    min_confidence: float = 0.15
    max_blur_threshold: float = 150.0  # Laplacian variance
    min_brightness: float = 20.0  # Mean pixel value
    max_aspect_ratio: float = 3.0
    min_aspect_ratio: float = 0.15
    duplicate_iou_threshold: float = 0.85

    # Augmentation (training only)
    augment: bool = True
    hflip_prob: float = 0.5
    brightness_range: Tuple[float, float] = (0.7, 1.3)
    contrast_range: Tuple[float, float] = (0.7, 1.3)
    rotation_deg: float = 8.0
    blur_kernel: int = 3
    color_jitter_brightness: float = 0.2
    color_jitter_contrast: float = 0.2
    color_jitter_saturation: float = 0.2
    color_jitter_hue: float = 0.1
    erase_scale: Tuple[float, float] = (0.02, 0.2)
    erase_ratio: Tuple[float, float] = (0.3, 3.3)

    # Training
    image_size: Tuple[int, int] = (224, 224)
    batch_size: int = 32
    epochs: int = 100
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    early_stop_patience: int = 10
    gradient_clip_val: float = 1.0
    mixed_precision: bool = True
    num_workers: int = 4
    seed: int = 42

    # Model
    model_name: str = "efficientnet_b0"
    pretrained: bool = True
    dropout: float = 0.3

    # Paths
    checkpoint_dir: Path = Path("models/classifier")
    log_dir: Path = Path("logs/classifier")


@dataclass
class InferenceConfig:
    """Configuration for inference."""

    model_path: Path = Path("models/classifier/best.pth")
    model_name: str = "efficientnet_b0"
    image_size: Tuple[int, int] = (224, 224)
    device: str = "cuda"
    confidence_threshold: float = 0.0