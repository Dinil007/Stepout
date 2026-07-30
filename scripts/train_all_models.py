"""Train all model architectures and generate comparison report."""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.classification.config import CLASS_NAMES, DatasetConfig
from app.classification.data import create_data_loaders
from app.classification.models import build_model, count_parameters
from app.classification.trainer import Trainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("train_all_models")


def train_model(
    model_name: str,
    config: DatasetConfig,
    device: torch.device,
) -> Dict:
    """Train a single model architecture."""
    logger.info(f"\n{'='*60}")
    logger.info(f"Training {model_name}")
    logger.info(f"{'='*60}\n")

    # Set seed for reproducibility
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    # Create data loaders (same split for all models)
    train_loader, val_loader, test_loader = create_data_loaders(config)

    # Build model
    model = build_model(
        model_name=model_name,
        pretrained=config.pretrained,
        dropout=config.dropout,
    )

    # Count parameters
    num_params = count_parameters(model)
    logger.info(f"Model: {model_name}, Parameters: {num_params:,}")

    # Train
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device,
    )

    metrics = trainer.train()

    # Add model info
    metrics["model_name"] = model_name
    metrics["num_parameters"] = num_params

    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output-dir", type=str, default="models/classifier")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Base config
    config = DatasetConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        image_size=(args.image_size, args.image_size),
        mixed_precision=(device.type == "cuda"),
    )

    models = [
        "efficientnet_b0",
        "efficientnet_b2",
        "mobilenet_v3",
        "resnet18",
        "convnext_tiny",
    ]

    results = {}
    for model_name in models:
        try:
            metrics = train_model(model_name, config, device)
            results[model_name] = metrics

            # Save individual results
            output_path = Path(args.output_dir) / f"{model_name}_metrics.json"
            serializable = {
                k: v for k, v in metrics.items()
                if isinstance(v, (int, float, str, list, dict, bool))
            }
            with open(output_path, "w") as f:
                json.dump(serializable, f, indent=2)

        except Exception as e:
            logger.error(f"Error training {model_name}: {e}")
            results[model_name] = {"error": str(e)}

    # Generate comparison report
    generate_comparison_report(results, Path(args.output_dir) / "model_comparison.md")
    logger.info("Training complete!")


def generate_comparison_report(results: Dict, output_path: Path) -> None:
    """Generate model comparison report."""
    lines = [
        "# Model Comparison Report",
        "",
        "## Performance Comparison",
        "",
        "| Model | Val Accuracy | Val F1 | Params (M) |",
        "|-------|--------------|--------|------------|",
    ]

    for model_name, metrics in results.items():
        if "error" in metrics:
            lines.append(f"| {model_name} | ERROR | ERROR | - |")
            continue

        val_acc = metrics.get("best_val_acc", 0)
        macro_f1 = metrics.get("macro_f1", 0)
        params = metrics.get("num_parameters", 0) / 1e6
        lines.append(f"| {model_name} | {val_acc:.4f} | {macro_f1:.4f} | {params:.2f} |")

    lines.append("")

    best_model = max(
        [k for k in results.keys() if "error" not in results[k]],
        key=lambda k: results[k].get("best_val_acc", 0),
    )

    lines.extend([
        "## Best Model",
        "",
        f"**{best_model}** with validation accuracy: {results[best_model].get('best_val_acc', 0):.4f}",
        "",
    ])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    logger.info(f"Comparison report saved to: {output_path}")


if __name__ == "__main__":
    main()