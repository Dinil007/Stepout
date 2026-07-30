"""Prepare dataset for training.

This script:
1. Creates train/val/test splits
2. Applies quality filtering
3. Copies images to prepared directory structure
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.classification.config import DatasetConfig
from app.classification.dataset import prepare_and_split_dataset
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

if __name__ == "__main__":
    config = DatasetConfig()
    quality_filter = prepare_and_split_dataset(config)
    print("\nDataset preparation complete!")
    print(f"Accepted: {quality_filter.stats['accepted']}")
    print(f"Rejected: {sum(v for k, v in quality_filter.stats.items() if 'rejected' in k)}")