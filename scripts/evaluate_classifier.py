"""Evaluate trained classifier on test dataset.

Generates:
- Accuracy metrics
- Confusion matrix
- Classification report
- Misclassified images list
- evaluation_report.md
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.classification.config import CLASS_NAMES, DatasetConfig
from app.classification.data import prepare_classification_datasets
from app.classification.models import create_model
from app.classification.trainer import ConfusionMatrix

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def evaluate(
    model_path: str = "models/classifier/best.pth",
    data_dir: str = "datasets/person_classifier/prepared",
    model_name: str = "efficientnet_b0",
    batch_size: int = 32,
    image_size: int = 224,
    output_dir: str = "models/classifier",
) -> None:
    """Evaluate model on test dataset."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load test dataset
    config = DatasetConfig(
        prepared_dir=Path(data_dir),
        image_size=(image_size, image_size),
        batch_size=batch_size,
        augment=False,
    )

    _, _, test_dataset = prepare_classification_datasets(config)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    logger.info(f"Test samples: {len(test_dataset)}")

    # Load model
    model = create_model(model_name, num_classes=4, pretrained=False)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    # Evaluate
    confusion = ConfusionMatrix(4)
    misclassified: List[Tuple[str, int, int, float]] = []  # (path, true, pred, conf)

    with torch.no_grad():
        for images, labels, paths in tqdm(test_loader, desc="Testing"):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            preds = outputs.argmax(dim=1)
            confidences = probs.max(dim=1)[0]

            confusion.update(preds, labels)

            for i in range(len(paths)):
                if preds[i] != labels[i]:
                    misclassified.append((
                        paths[i],
                        int(labels[i].item()),
                        int(preds[i].item()),
                        float(confidences[i].item()),
                    ))

    # Generate report
    metrics = confusion.to_dict()
    report_lines = [
        "# Evaluation Report",
        "",
        "## Test Results",
        "",
        f"- **Accuracy**: {metrics['accuracy']:.4f}",
        f"- **Macro F1**: {metrics['macro_f1']:.4f}",
        "",
        "## Per-Class Metrics",
        "",
        "| Class | Accuracy | Precision | Recall | F1 |",
        "|-------|----------|-----------|--------|----|",
    ]

    for cls in CLASS_NAMES:
        acc = metrics['per_class_accuracy'][cls]
        prec = metrics['precision'][cls]
        rec = metrics['recall'][cls]
        f1 = metrics['f1_score'][cls]
        report_lines.append(f"| {cls} | {acc:.4f} | {prec:.4f} | {rec:.4f} | {f1:.4f} |")

    report_lines.extend(
        [
            "",
            "## Confusion Matrix",
            "",
            "|        | TEAM_A | TEAM_B | REFEREE | COACH |",
            "|--------|--------|--------|---------|-------|",
        ]
    )

    matrix = metrics["matrix"]
    for i, cls in enumerate(CLASS_NAMES):
        row = " | ".join(str(matrix[i, j]) for j in range(4))
        report_lines.append(f"| {cls} | {row} |")

    report_lines.extend(
        [
            "",
            "## Misclassified Images",
            "",
            f"Total misclassified: {len(misclassified)}",
            "",
        ]
    )

    if misclassified:
        report_lines.append("| Image | True | Predicted | Confidence |")
        report_lines.append("|-------|------|-----------|------------|")
        for path, true, pred, conf in misclassified[:50]:
            report_lines.append(
                f"| {Path(path).name} | {CLASS_NAMES[true]} | {CLASS_NAMES[pred]} | {conf:.4f} |"
            )
        if len(misclassified) > 50:
            report_lines.append(f"\n... and {len(misclassified) - 50} more")

    report_lines.append("")

    report_path = output_path / "evaluation_report.md"
    with open(report_path, "w") as f:
        f.write("\n".join(report_lines))

    logger.info(f"Evaluation report saved to {report_path}")
    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Macro F1: {metrics['macro_f1']:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/classifier/best.pth")
    parser.add_argument("--data-dir", default="datasets/person_classifier/prepared")
    parser.add_argument("--model-name", default="efficientnet_b0")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--output-dir", default="models/classifier")
    args = parser.parse_args()

    evaluate(
        args.model,
        args.data_dir,
        args.model_name,
        args.batch_size,
        args.image_size,
        args.output_dir,
    )