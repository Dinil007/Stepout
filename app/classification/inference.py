"""Inference engine for person classification.

Used by:
- classify_person.py (standalone inference)
- Pipeline integration (real-time classification during tracking)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms

from app.classification.config import CLASS_NAMES, InferenceConfig, NUM_CLASSES
from app.classification.models import build_model

logger = logging.getLogger(__name__)


class PersonClassifier:
    """Person classification inference engine.

    Classifies a cropped person image into:
    - TEAM_A
    - TEAM_B
    - REFEREE
    - COACH
    """

    def __init__(self, config: Optional[InferenceConfig] = None):
        """Initialize the classifier.

        Args:
            config: Inference configuration. Uses defaults if not provided.
        """
        if config is None:
            config = InferenceConfig()

        self.config = config
        self.device = torch.device(
            config.device if torch.cuda.is_available() else "cpu"
        )

        # Image transform
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(config.image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

        # Load model
        self.model = self._load_model(config.model_path)
        self.model.eval()
        logger.info(
            f"PersonClassifier initialized: model={config.model_name}, "
            f"device={self.device}, classes={CLASS_NAMES}"
        )

    def _load_model(self, model_path: Path) -> nn.Module:
        """Load model from checkpoint."""
        if not model_path.exists():
            logger.warning(
                f"Model checkpoint not found: {model_path}. "
                f"Building untrained model instead."
            )
            model = build_model(
                model_name=self.config.model_name,
                pretrained=False,
            )
            return model

        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)

        # Get model config from checkpoint
        ckpt_config = checkpoint.get("config", {})
        model_name = ckpt_config.get("model_name", self.config.model_name)

        model = build_model(
            model_name=model_name,
            pretrained=False,
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model = model.to(self.device)

        logger.info(f"Loaded model from {model_path}")
        return model

    @torch.inference_mode()
    def predict(self, image: np.ndarray) -> Tuple[str, float, np.ndarray]:
        """Classify a single person crop.

        Args:
            image: BGR image crop (H, W, 3)

        Returns:
            Tuple of (class_name, confidence, probabilities)
        """
        # Convert BGR to RGB
        if image.shape[2] == 3:
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            rgb = image

        # Transform
        tensor = self.transform(rgb).unsqueeze(0).to(self.device)

        # Inference
        outputs = self.model(tensor)
        probs = torch.softmax(outputs, dim=1)
        confidence, pred_idx = torch.max(probs, dim=1)

        class_name = CLASS_NAMES[pred_idx.item()]
        confidence_val = confidence.item()
        prob_array = probs.squeeze(0).cpu().numpy()

        return class_name, confidence_val, prob_array

    @torch.inference_mode()
    def predict_batch(self, images: List[np.ndarray]) -> List[Tuple[str, float, np.ndarray]]:
        """Classify a batch of person crops.

        Args:
            images: List of BGR image crops

        Returns:
            List of (class_name, confidence, probabilities) tuples
        """
        if not images:
            return []

        batch = []
        for img in images:
            if img.shape[2] == 3:
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            else:
                rgb = img
            tensor = self.transform(rgb)
            batch.append(tensor)

        batch_tensor = torch.stack(batch).to(self.device)
        outputs = self.model(batch_tensor)
        probs = torch.softmax(outputs, dim=1)
        confidences, pred_indices = torch.max(probs, dim=1)

        results = []
        for i in range(len(images)):
            results.append((
                CLASS_NAMES[pred_indices[i].item()],
                confidences[i].item(),
                probs[i].cpu().numpy(),
            ))

        return results

    def get_class_index(self, class_name: str) -> int:
        """Get index for a class name."""
        return CLASS_NAMES.index(class_name)

    def get_class_name(self, class_idx: int) -> str:
        """Get class name for an index."""
        return CLASS_NAMES[class_idx]