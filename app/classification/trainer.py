"""Training engine for person classification.

Features:
- Early stopping
- Best checkpoint saving
- Resume training
- Mixed precision (AMP)
- Learning rate scheduler
- TensorBoard logging
- Confusion matrix
- Accuracy, Precision, Recall, F1 Score
- Per-class accuracy
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    SummaryWriter = None

from app.classification.config import CLASS_NAMES, DatasetConfig, NUM_CLASSES

logger = logging.getLogger(__name__)


class AverageMeter:
    """Computes and stores the average and current value."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class ConfusionMatrix:
    """Confusion matrix for multi-class classification."""

    def __init__(self, num_classes: int):
        self.num_classes = num_classes
        self.matrix = np.zeros((num_classes, num_classes), dtype=np.int64)

    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        """Update with batch predictions and targets."""
        for p, t in zip(preds.cpu().numpy(), targets.cpu().numpy()):
            self.matrix[t, p] += 1

    def reset(self):
        self.matrix.fill(0)

    def accuracy(self) -> float:
        return np.trace(self.matrix) / max(self.matrix.sum(), 1)

    def per_class_accuracy(self) -> Dict[str, float]:
        accs = {}
        for i in range(self.num_classes):
            total = self.matrix[i, :].sum()
            accs[CLASS_NAMES[i]] = float(self.matrix[i, i] / max(total, 1))
        return accs

    def precision(self) -> Dict[str, float]:
        precs = {}
        for i in range(self.num_classes):
            tp = self.matrix[i, i]
            fp = self.matrix[:, i].sum() - tp
            precs[CLASS_NAMES[i]] = float(tp / max(tp + fp, 1))
        return precs

    def recall(self) -> Dict[str, float]:
        recs = {}
        for i in range(self.num_classes):
            tp = self.matrix[i, i]
            fn = self.matrix[i, :].sum() - tp
            recs[CLASS_NAMES[i]] = float(tp / max(tp + fn, 1))
        return recs

    def f1_score(self) -> Dict[str, float]:
        precs = self.precision()
        recs = self.recall()
        f1s = {}
        for cls in CLASS_NAMES:
            p, r = precs[cls], recs[cls]
            f1s[cls] = float(2 * p * r / max(p + r, 1e-9))
        return f1s

    def macro_f1(self) -> float:
        f1s = self.f1_score()
        return float(np.mean(list(f1s.values())))

    def to_dict(self) -> Dict:
        return {
            "matrix": self.matrix.tolist(),
            "accuracy": self.accuracy(),
            "per_class_accuracy": self.per_class_accuracy(),
            "precision": self.precision(),
            "recall": self.recall(),
            "f1_score": self.f1_score(),
            "macro_f1": self.macro_f1(),
        }


class Trainer:
    """Training engine with all required features."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: DatasetConfig,
        device: torch.device,
        resume_path: Optional[Path] = None,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device

        # Loss and optimizer
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=config.epochs, eta_min=1e-6
        )

        # Mixed precision
        self.scaler = GradScaler(enabled=config.mixed_precision and device.type == "cuda")

        # TensorBoard (optional)
        self.writer = None
        if TENSORBOARD_AVAILABLE:
            self.writer = SummaryWriter(log_dir=str(config.log_dir))

        # Checkpoint directory
        self.checkpoint_dir = config.checkpoint_dir
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Training state
        self.start_epoch = 0
        self.best_val_acc = 0.0
        self.best_epoch = 0
        self.early_stop_counter = 0
        self.best_model_state = None

        # Metrics
        self.train_losses: List[float] = []
        self.val_losses: List[float] = []
        self.val_accuracies: List[float] = []

        # Resume from checkpoint
        if resume_path is not None and resume_path.exists():
            self._load_checkpoint(resume_path)

    def _load_checkpoint(self, path: Path) -> None:
        """Load checkpoint and resume training."""
        logger.info(f"Resuming from checkpoint: {path}")
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.start_epoch = checkpoint["epoch"] + 1
        self.best_val_acc = checkpoint.get("best_val_acc", 0.0)
        self.best_epoch = checkpoint.get("best_epoch", 0)
        self.train_losses = checkpoint.get("train_losses", [])
        self.val_losses = checkpoint.get("val_losses", [])
        self.val_accuracies = checkpoint.get("val_accuracies", [])
        logger.info(
            f"Resumed at epoch {self.start_epoch}, "
            f"best val acc: {self.best_val_acc:.4f}"
        )

    def _save_checkpoint(self, epoch: int, is_best: bool = False) -> None:
        """Save training checkpoint."""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_val_acc": self.best_val_acc,
            "best_epoch": self.best_epoch,
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "val_accuracies": self.val_accuracies,
            "config": {
                "model_name": self.config.model_name,
                "num_classes": NUM_CLASSES,
                "image_size": self.config.image_size,
                "class_names": CLASS_NAMES,
            },
        }

        # Save latest checkpoint
        latest_path = self.checkpoint_dir / "latest.pth"
        torch.save(checkpoint, latest_path)
        logger.debug(f"Saved latest checkpoint: {latest_path}")

        # Save best checkpoint
        if is_best:
            best_path = self.checkpoint_dir / "best.pth"
            torch.save(checkpoint, best_path)
            logger.info(f"Saved best checkpoint (acc={self.best_val_acc:.4f}): {best_path}")

    def train_epoch(self) -> float:
        """Train for one epoch. Returns average loss."""
        self.model.train()
        loss_meter = AverageMeter()

        for images, labels in self.train_loader:
            images = images.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()

            # Mixed precision forward
            with autocast(enabled=self.config.mixed_precision and self.device.type == "cuda"):
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

            # Backward with gradient scaling
            self.scaler.scale(loss).backward()
            if self.config.gradient_clip_val > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.gradient_clip_val
                )
            self.scaler.step(self.optimizer)
            self.scaler.update()

            loss_meter.update(loss.item(), images.size(0))

        return loss_meter.avg

    @torch.inference_mode()
    def validate(self) -> Tuple[float, ConfusionMatrix]:
        """Validate the model. Returns (avg_loss, confusion_matrix)."""
        self.model.eval()
        loss_meter = AverageMeter()
        confusion = ConfusionMatrix(NUM_CLASSES)

        for images, labels in self.val_loader:
            images = images.to(self.device)
            labels = labels.to(self.device)

            with autocast(enabled=self.config.mixed_precision and self.device.type == "cuda"):
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

            preds = outputs.argmax(dim=1)
            confusion.update(preds, labels)
            loss_meter.update(loss.item(), images.size(0))

        return loss_meter.avg, confusion

    def train(self) -> Dict:
        """Run full training loop.

        Returns:
            Dict with training results and metrics.
        """
        logger.info(
            f"Starting training: {self.config.epochs} epochs, "
            f"batch_size={self.config.batch_size}, "
            f"lr={self.config.learning_rate}, "
            f"device={self.device}"
        )
        logger.info(
            f"Train samples: {len(self.train_loader.dataset)}, "
            f"Val samples: {len(self.val_loader.dataset)}"
        )

        total_start = time.time()

        for epoch in range(self.start_epoch, self.config.epochs):
            epoch_start = time.time()

            # Train
            train_loss = self.train_epoch()
            self.train_losses.append(train_loss)

            # Validate
            val_loss, confusion = self.validate()
            self.val_losses.append(val_loss)
            val_acc = confusion.accuracy()
            self.val_accuracies.append(val_acc)

            # Step scheduler
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]["lr"]

            # Logging
            epoch_time = time.time() - epoch_start
            logger.info(
                f"Epoch {epoch+1}/{self.config.epochs} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val Acc: {val_acc:.4f} | "
                f"LR: {current_lr:.2e} | "
                f"Time: {epoch_time:.1f}s"
            )

            # TensorBoard
            if self.writer is not None:
                self.writer.add_scalar("Loss/train", train_loss, epoch)
                self.writer.add_scalar("Loss/val", val_loss, epoch)
                self.writer.add_scalar("Accuracy/val", val_acc, epoch)
                self.writer.add_scalar("LR", current_lr, epoch)

                # Per-class metrics
                per_class_acc = confusion.per_class_accuracy()
                for cls_name, acc in per_class_acc.items():
                    self.writer.add_scalar(f"PerClassAcc/{cls_name}", acc, epoch)

            # Checkpoint if best
            is_best = val_acc > self.best_val_acc
            if is_best:
                self.best_val_acc = val_acc
                self.best_epoch = epoch
                self.early_stop_counter = 0
                self.best_model_state = {
                    k: v.cpu().clone() for k, v in self.model.state_dict().items()
                }
                self._save_checkpoint(epoch, is_best=True)
            else:
                self.early_stop_counter += 1

            # Save latest checkpoint every 10 epochs
            if (epoch + 1) % 10 == 0:
                self._save_checkpoint(epoch)

            # Early stopping
            if self.early_stop_counter >= self.config.early_stop_patience:
                logger.info(
                    f"Early stopping triggered after {epoch+1} epochs "
                    f"(no improvement for {self.early_stop_counter} epochs)"
                )
                break

        total_time = time.time() - total_start
        logger.info(
            f"Training completed in {total_time:.1f}s. "
            f"Best val acc: {self.best_val_acc:.4f} at epoch {self.best_epoch+1}"
        )

        # Final validation with best model
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
        _, final_confusion = self.validate()

        # Log final metrics
        metrics = final_confusion.to_dict()
        logger.info(f"Final validation metrics:")
        logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"  Macro F1: {metrics['macro_f1']:.4f}")
        for cls_name in CLASS_NAMES:
            logger.info(
                f"  {cls_name}: "
                f"Acc={metrics['per_class_accuracy'][cls_name]:.4f}, "
                f"Prec={metrics['precision'][cls_name]:.4f}, "
                f"Rec={metrics['recall'][cls_name]:.4f}, "
                f"F1={metrics['f1_score'][cls_name]:.4f}"
            )

        # Save metrics to JSON
        metrics["train_losses"] = self.train_losses
        metrics["val_losses"] = self.val_losses
        metrics["val_accuracies"] = self.val_accuracies
        metrics["best_epoch"] = self.best_epoch
        metrics["best_val_acc"] = self.best_val_acc
        metrics["total_training_time_s"] = round(total_time, 2)
        metrics["class_names"] = CLASS_NAMES

        metrics_path = self.checkpoint_dir / "training_metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"Training metrics saved to {metrics_path}")

        # Close TensorBoard writer
        if self.writer is not None:
            self.writer.close()

        return metrics
