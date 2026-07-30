"""PyTorch dataset and data loader for person classification."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from app.classification.config import CLASS_NAMES, DatasetConfig

logger = logging.getLogger(__name__)


class PersonDataset(Dataset):
    """Dataset for person classification from prepared directory structure."""

    def __init__(
        self,
        root_dir: Path,
        split: str,
        transform: Optional[Callable] = None,
        augment: Optional[Callable] = None,
        image_size: Tuple[int, int] = (224, 224),
    ):
        """Initialize dataset.

        Args:
            root_dir: Root directory of prepared dataset (e.g., datasets/person_classifier/prepared)
            split: One of 'train', 'val', 'test'
            transform: PyTorch transform to apply
            augment: Optional augmentation callable (applied before transform)
            image_size: Target image size (H, W)
        """
        self.root_dir = Path(root_dir) / split
        self.split = split
        self.image_size = image_size
        self.augment = augment

        # Default transform if none provided
        if transform is None:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(image_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ])
        else:
            self.transform = transform

        # Build class mapping
        self.class_to_idx: Dict[str, int] = {
            name: idx for idx, name in enumerate(CLASS_NAMES)
        }

        # Collect all image paths and labels
        self.samples: List[Tuple[Path, int]] = []
        for class_name in CLASS_NAMES:
            class_dir = self.root_dir / class_name
            if not class_dir.exists():
                logger.warning(f"Class directory not found: {class_dir}")
                continue

            for img_path in sorted(class_dir.iterdir()):
                if img_path.suffix.lower() in (".jpg", ".jpeg", ".png"):
                    self.samples.append((img_path, self.class_to_idx[class_name]))

        logger.info(
            f"Loaded {split} split: {len(self.samples)} images "
            f"from {len(CLASS_NAMES)} classes"
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]

        # Read image
        image = cv2.imread(str(img_path))
        if image is None:
            logger.warning(f"Could not read {img_path}, returning blank")
            image = np.zeros((*self.image_size, 3), dtype=np.uint8)

        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply augmentation (training only)
        if self.augment is not None and self.split == "train":
            image = self.augment(image)

        # Apply transform
        tensor = self.transform(image)

        return tensor, label


def create_data_loaders(
    config: DatasetConfig,
    augment_fn: Optional[Callable] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train, val, test data loaders.

    Args:
        config: Dataset configuration
        augment_fn: Optional augmentation function (applied to training only)

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Training transform with augmentation
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(config.image_size),
        transforms.RandomHorizontalFlip(p=config.hflip_prob),
        transforms.ColorJitter(
            brightness=config.color_jitter_brightness,
            contrast=config.color_jitter_contrast,
            saturation=config.color_jitter_saturation,
            hue=config.color_jitter_hue,
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        transforms.RandomErasing(
            p=0.3,
            scale=config.erase_scale,
            ratio=config.erase_ratio,
        ),
    ])

    # Validation/test transform (no augmentation)
    eval_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(config.image_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    train_dataset = PersonDataset(
        root_dir=config.prepared_dir,
        split="train",
        transform=train_transform if config.augment else eval_transform,
        augment=augment_fn,
        image_size=config.image_size,
    )

    val_dataset = PersonDataset(
        root_dir=config.prepared_dir,
        split="val",
        transform=eval_transform,
        image_size=config.image_size,
    )

    test_dataset = PersonDataset(
        root_dir=config.prepared_dir,
        split="test",
        transform=eval_transform,
        image_size=config.image_size,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
    )

    logger.info(
        f"Data loaders created: "
        f"train={len(train_dataset)}, "
        f"val={len(val_dataset)}, "
        f"test={len(test_dataset)}"
    )

    return train_loader, val_loader, test_loader