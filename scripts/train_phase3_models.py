"""Train all 5 model architectures with class weights for Phase 3."""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.classification.config import CLASS_NAMES, DatasetConfig, NUM_CLASSES
from app.classification.data import create_data_loaders
from app.classification.models import build_model, count_parameters
from app.classification.trainer import Trainer
from app.classification.data import PersonDataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("train_phase3")


class WeightedTrainer(Trainer):
    """Trainer with weighted loss for class imbalance."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Compute class weights from training data
        train_labels = []
        for _, label in self.train_loader.dataset:
            train_labels.append(label)
        counts = np.bincount(train_labels, minlength=NUM_CLASSES)
        weights = 1.0 / (counts + 1e-6)
        weights = weights / weights.sum() * NUM_CLASSES  # Normalize
        weights_tensor = torch.tensor(weights, dtype=torch.float32).to(self.device)
        self.criterion = torch.nn.CrossEntropyLoss(
            weight=weights_tensor, label_smoothing=0.05
        )
        logger.info(f"Class weights: {weights}")


def train_model(
    model_name: str,
    config: DatasetConfig,
    device: torch.device,
    output_dir: Path,
) -> Dict:
    """Train a single model architecture with class weights."""
    logger.info(f"\n{'='*60}")
    logger.info(f"Training {model_name}")
    logger.info(f"{'='*60}\n")

    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    train_loader, val_loader, test_loader = create_data_loaders(config)

    model = build_model(
        model_name=model_name,
        pretrained=config.pretrained,
        dropout=max(config.dropout, 0.3),
    )

    num_params = count_parameters(model)
    logger.info(f"Model: {model_name}, Parameters: {num_params:,}")

    # Use weighted trainer
    trainer = WeightedTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device,
    )

    metrics = trainer.train()
    metrics["model_name"] = model_name
    metrics["num_parameters"] = num_params

    # Save model checkpoint
    checkpoint_path = output_dir / f"{model_name}_best.pth"
    if hasattr(trainer, 'best_model_state') and trainer.best_model_state is not None:
        torch.save({
            "model_state_dict": trainer.best_model_state,
            "metrics": {k: v for k, v in metrics.items() if k in [
                "accuracy", "macro_f1", "per_class_accuracy", "f1_score"
            ]},
            "model_name": model_name,
            "num_classes": NUM_CLASSES,
            "class_names": CLASS_NAMES,
            "image_size": config.image_size,
        }, checkpoint_path)
        logger.info(f"Model saved to {checkpoint_path}")

    return metrics


def generate_comparison_report(results: Dict, output_path: Path) -> None:
    """Generate model comparison report."""
    lines = [
        "# Model Comparison Report",
        "",
        "## Performance Comparison",
        "",
        "| Model | Val Accuracy | Macro F1 | Params (M) | Training Time (s) |",
        "|-------|--------------|----------|------------|-------------------|",
    ]

    for model_name, metrics in results.items():
        if "error" in metrics:
            lines.append(f"| {model_name} | ERROR | ERROR | - | - |")
            continue
        val_acc = metrics.get("best_val_acc", 0)
        macro_f1 = metrics.get("macro_f1", 0)
        params = metrics.get("num_parameters", 0) / 1e6
        train_time = metrics.get("total_training_time_s", 0)
        lines.append(f"| {model_name} | {val_acc:.4f} | {macro_f1:.4f} | {params:.2f} | {train_time:.0f}s |")

    lines.append("")

    valid_models = {k: v for k, v in results.items() if "error" not in v}
    if valid_models:
        best_model = max(valid_models.keys(), key=lambda k: valid_models[k].get("best_val_acc", 0))
        best = valid_models[best_model]
        lines.extend([
            "## Best Model",
            "",
            f"**{best_model}** with validation accuracy: {best.get('best_val_acc', 0):.4f}",
            f"Macro F1: {best.get('macro_f1', 0):.4f}",
            f"Parameters: {best.get('num_parameters', 0):,}",
            "",
            "### Per-Class Performance of Best Model",
            "",
            "| Class | Accuracy | F1 Score |",
            "|-------|----------|----------|",
        ])
        for cls in CLASS_NAMES:
            acc = best.get("per_class_accuracy", {}).get(cls, 0)
            f1 = best.get("f1_score", {}).get(cls, 0)
            lines.append(f"| {cls} | {acc:.4f} | {f1:.4f} |")

        lines.append("")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    logger.info(f"Comparison report saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=160)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output-dir", type=str, default="models/classifier")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = DatasetConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        image_size=(args.image_size, args.image_size),
        mixed_precision=(device.type == "cuda"),
        learning_rate=5e-4,
        weight_decay=1e-3,
        early_stop_patience=8,
        dropout=0.4,
    )

    models_to_train = [
        "efficientnet_b0",
        "efficientnet_b2",
        "mobilenet_v3",
        "resnet18",
        "convnext_tiny",
    ]

    results = {}
    for model_name in models_to_train:
        try:
            t0 = time.time()
            metrics = train_model(model_name, config, device, output_dir)
            t1 = time.time()
            metrics["wall_time_s"] = round(t1 - t0, 1)
            results[model_name] = metrics

            serializable = {
                k: v for k, v in metrics.items()
                if isinstance(v, (int, float, str, list, dict, bool))
            }
            path = output_dir / f"{model_name}_metrics.json"
            with open(path, "w") as f:
                json.dump(serializable, f, indent=2)

        except Exception as e:
            logger.error(f"Error training {model_name}: {e}")
            results[model_name] = {"error": str(e)}

    generate_comparison_report(results, output_dir / "model_comparison.md")
    logger.info("All training complete!")


if __name__ == "__main__":
    main()