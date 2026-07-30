#!/usr/bin/env python
"""Train a person classification model.

Usage:
    python scripts/train_classifier.py --model efficientnet_b0 --epochs 100 --batch_size 32
    python scripts/train_classifier.py --model resnet18 --epochs 50 --lr 1e-3 --resume
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from app.classification.config import CLASS_NAMES, DatasetConfig
from app.classification.data import create_data_loaders
from app.classification.dataset import prepare_and_split_dataset
from app.classification.models import build_model, count_parameters
from app.classification.trainer import Trainer

import os
os.makedirs("logs/classifier", exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("logs/classifier/training.log", mode="w"),
    ],
)
logger = logging.getLogger("train_classifier")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train person classification model"
    )
    # Model
    parser.add_argument(
        "--model", type=str, default="efficientnet_b0",
        choices=["efficientnet_b0", "efficientnet_b2", "mobilenet_v3",
                 "convnext_tiny", "resnet18"],
        help="Model architecture",
    )
    parser.add_argument("--pretrained", action="store_true", default=True,
                        help="Use pretrained weights")
    parser.add_argument("--no-pretrained", action="store_false", dest="pretrained",
                        help="Train from scratch")
    parser.add_argument("--dropout", type=float, default=0.3,
                        help="Dropout rate")

    # Training
    parser.add_argument("--epochs", type=int, default=100,
                        help="Maximum number of epochs")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4,
                        help="Weight decay")
    parser.add_argument("--early-stop", type=int, default=10,
                        help="Early stopping patience")

    # Data
    parser.add_argument("--image-size", type=int, default=224,
                        help="Input image size (square)")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="Data loader workers")
    parser.add_argument("--prepare", action="store_true",
                        help="Run dataset preparation before training")

    # Resume
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")

    # Misc
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device (cuda or cpu)")

    return parser.parse_args()


def main():
    args = parse_args()

    # Set random seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Configuration
    config = DatasetConfig(
        model_name=args.model,
        pretrained=args.pretrained,
        dropout=args.dropout,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        early_stop_patience=args.early_stop,
        image_size=(args.image_size, args.image_size),
        num_workers=args.num_workers,
        seed=args.seed,
        mixed_precision=(device.type == "cuda"),
    )

    # Step 1: Prepare dataset (if requested)
    if args.prepare:
        logger.info("Running dataset preparation...")
        quality_filter = prepare_and_split_dataset(config)
        logger.info(f"Quality filter stats: {quality_filter.stats}")

    # Step 2: Create data loaders
    logger.info("Creating data loaders...")
    train_loader, val_loader, test_loader = create_data_loaders(config)

    # Step 3: Build model
    logger.info(f"Building model: {args.model}")
    model = build_model(
        model_name=args.model,
        pretrained=args.pretrained,
        dropout=args.dropout,
    )
    logger.info(f"Model parameters: {count_parameters(model):,}")

    # Step 4: Train
    resume_path = Path(args.resume) if args.resume else None
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device,
        resume_path=resume_path,
    )

    metrics = trainer.train()

    # Step 5: Evaluate on test set
    logger.info("Evaluating on test set...")
    from app.classification.trainer import ConfusionMatrix
    from app.classification.config import NUM_CLASSES

    model.eval()
    test_confusion = ConfusionMatrix(NUM_CLASSES)
    test_loss = 0.0
    criterion = torch.nn.CrossEntropyLoss()

    with torch.inference_mode():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            preds = outputs.argmax(dim=1)
            test_confusion.update(preds, labels)

    test_metrics = test_confusion.to_dict()
    test_metrics["test_loss"] = test_loss / len(test_loader)

    logger.info(f"\n{'='*60}")
    logger.info(f"  TEST SET RESULTS")
    logger.info(f"{'='*60}")
    logger.info(f"  Test Accuracy: {test_metrics['accuracy']:.4f}")
    logger.info(f"  Test Macro F1: {test_metrics['macro_f1']:.4f}")
    logger.info(f"  Test Loss: {test_metrics['test_loss']:.4f}")
    logger.info(f"{'='*60}")
    for cls_name in CLASS_NAMES:
        logger.info(
            f"  {cls_name:15s}: "
            f"Acc={test_metrics['per_class_accuracy'][cls_name]:.4f}, "
            f"Prec={test_metrics['precision'][cls_name]:.4f}, "
            f"Rec={test_metrics['recall'][cls_name]:.4f}, "
            f"F1={test_metrics['f1_score'][cls_name]:.4f}"
        )
    logger.info(f"{'='*60}")

    # Save test metrics
    import json
    test_metrics_path = config.checkpoint_dir / "test_metrics.json"
    with open(test_metrics_path, "w") as f:
        json.dump(test_metrics, f, indent=2)
    logger.info(f"Test metrics saved to {test_metrics_path}")

    logger.info("Training complete!")


if __name__ == "__main__":
    main()