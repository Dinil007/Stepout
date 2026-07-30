#!/usr/bin/env python
"""Standalone person classification inference.

Classifies a single crop image or all images in a directory.

Usage:
    python scripts/classify_person.py --image path/to/crop.jpg
    python scripts/classify_person.py --dir path/to/track_folder
    python scripts/classify_person.py --image crop.jpg --model models/classifier/best.pth
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np

from app.classification.config import CLASS_NAMES, InferenceConfig
from app.classification.inference import PersonClassifier

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("classify_person")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Classify a person crop into team/role"
    )
    parser.add_argument("--image", type=str, default=None,
                        help="Path to a single crop image")
    parser.add_argument("--dir", type=str, default=None,
                        help="Path to a directory of crop images")
    parser.add_argument("--model", type=str, default=None,
                        help="Path to model checkpoint")
    parser.add_argument("--model-name", type=str, default="efficientnet_b0",
                        help="Model architecture name")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device (cuda or cpu)")
    parser.add_argument("--threshold", type=float, default=0.0,
                        help="Confidence threshold")

    return parser.parse_args()


def main():
    args = parse_args()

    if args.image is None and args.dir is None:
        logger.error("Either --image or --dir must be provided")
        sys.exit(1)

    # Configuration
    config = InferenceConfig(
        model_name=args.model_name,
        device=args.device,
        confidence_threshold=args.threshold,
    )
    if args.model:
        config.model_path = Path(args.model)

    # Initialize classifier
    classifier = PersonClassifier(config)

    if args.image:
        # Single image
        image_path = Path(args.image)
        if not image_path.exists():
            logger.error(f"Image not found: {image_path}")
            sys.exit(1)

        image = cv2.imread(str(image_path))
        if image is None:
            logger.error(f"Could not read image: {image_path}")
            sys.exit(1)

        class_name, confidence, probs = classifier.predict(image)

        print(f"\n{'='*50}")
        print(f"  Image: {image_path.name}")
        print(f"{'='*50}")
        print(f"  Prediction: {class_name}")
        print(f"  Confidence: {confidence:.4f}")
        print(f"{'='*50}")
        print(f"  Class probabilities:")
        for i, cls in enumerate(CLASS_NAMES):
            print(f"    {cls:15s}: {probs[i]:.4f}")
        print(f"{'='*50}\n")

    if args.dir:
        # Directory of images
        dir_path = Path(args.dir)
        if not dir_path.exists():
            logger.error(f"Directory not found: {dir_path}")
            sys.exit(1)

        image_paths = sorted(dir_path.glob("*.jpg")) + sorted(dir_path.glob("*.png"))
        if not image_paths:
            logger.error(f"No images found in {dir_path}")
            sys.exit(1)

        print(f"\n{'='*60}")
        print(f"  Classifying {len(image_paths)} images in {dir_path}")
        print(f"{'='*60}")

        class_counts = {cls: 0 for cls in CLASS_NAMES}
        total_conf = 0.0

        for img_path in image_paths:
            image = cv2.imread(str(img_path))
            if image is None:
                continue

            class_name, confidence, probs = classifier.predict(image)
            class_counts[class_name] += 1
            total_conf += confidence

            print(
                f"  {img_path.name:30s} → {class_name:10s} "
                f"(conf={confidence:.4f})"
            )

        print(f"{'='*60}")
        print(f"  Summary:")
        for cls, count in class_counts.items():
            pct = count / len(image_paths) * 100
            print(f"    {cls:15s}: {count:4d} ({pct:.1f}%)")
        print(f"  Avg confidence: {total_conf / len(image_paths):.4f}")
        print(f"{'='*60}\n")


if __name__ == "__main__":
    main()