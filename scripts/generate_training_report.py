"""Generate training_report.md from training metrics.

This script reads training_metrics.json and generates a comprehensive
training report with all required information.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict

import numpy as np

from app.classification.config import CLASS_NAMES

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def generate_training_report(
    metrics_path: Path = Path("models/classifier/training_metrics.json"),
    output_path: Path = Path("models/classifier/training_report.md"),
) -> None:
    """Generate training report from metrics JSON."""
    if not metrics_path.exists():
        logger.error(f"Metrics file not found: {metrics_path}")
        return

    with open(metrics_path, "r") as f:
        metrics: Dict = json.load(f)

    lines = [
        "# Training Report",
        "",
        "## Model Information",
        "",
        f"- **Model Architecture**: {metrics.get('config', {}).get('model_name', 'N/A')}",
        f"- **Number of Classes**: {metrics.get('config', {}).get('num_classes', 'N/A')}",
        f"- **Image Size**: {metrics.get('config', {}).get('image_size', 'N/A')}",
        f"- **Classes**: {', '.join(metrics.get('class_names', CLASS_NAMES))}",
        "",
        "## Training Configuration",
        "",
        f"- **Best Epoch**: {metrics.get('best_epoch', 'N/A')}",
        f"- **Total Training Time**: {metrics.get('total_training_time_s', 0):.2f} seconds",
        f"- **Final Validation Accuracy**: {metrics.get('best_val_acc', 0):.4f}",
        "",
        "## Final Performance Metrics",
        "",
        f"- **Accuracy**: {metrics.get('accuracy', 0):.4f}",
        f"- **Macro F1 Score**: {metrics.get('macro_f1', 0):.4f}",
        "",
        "## Per-Class Performance",
        "",
        "| Class | Accuracy | Precision | Recall | F1 Score |",
        "|-------|----------|-----------|--------|----------|",
    ]

    for cls in CLASS_NAMES:
        acc = metrics.get("per_class_accuracy", {}).get(cls, 0)
        prec = metrics.get("precision", {}).get(cls, 0)
        rec = metrics.get("recall", {}).get(cls, 0)
        f1 = metrics.get("f1_score", {}).get(cls, 0)
        lines.append(f"| {cls} | {acc:.4f} | {prec:.4f} | {rec:.4f} | {f1:.4f} |")

    lines.extend(
        [
            "",
            "## Confusion Matrix",
            "",
            "|        | TEAM_A | TEAM_B | REFEREE | COACH |",
            "|--------|--------|--------|---------|-------|",
        ]
    )

    matrix = metrics.get("matrix", [])
    for i, cls in enumerate(CLASS_NAMES):
        if i < len(matrix):
            row = " | ".join(str(matrix[i][j]) for j in range(4))
            lines.append(f"| {cls} | {row} |")
        else:
            lines.append(f"| {cls} | N/A | N/A | N/A | N/A |")

    lines.extend(
        [
            "",
            "## Learning Curves",
            "",
            "### Training and Validation Loss",
            "",
            "| Epoch | Train Loss | Val Loss |",
            "|-------|------------|----------|",
        ]
    )

    train_losses = metrics.get("train_losses", [])
    val_losses = metrics.get("val_losses", [])
    for i, (train_loss, val_loss) in enumerate(zip(train_losses, val_losses)):
        lines.append(f"| {i+1} | {train_loss:.4f} | {val_loss:.4f} |")

    lines.extend(
        [
            "",
            "### Validation Accuracy per Epoch",
            "",
            "| Epoch | Val Accuracy |",
            "|-------|--------------|",
        ]
    )

    val_accuracies = metrics.get("val_accuracies", [])
    for i, val_acc in enumerate(val_accuracies):
        lines.append(f"| {i+1} | {val_acc:.4f} |")

    lines.append("")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write("\n".join(lines))

    logger.info(f"Training report saved to {output_path}")


if __name__ == "__main__":
    generate_training_report()