"""Model factory for person classification.

Supports:
- EfficientNet-B0, EfficientNet-B2
- MobileNetV3
- ConvNeXt-Tiny
- ResNet18
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn

from app.classification.config import NUM_CLASSES

logger = logging.getLogger(__name__)

# Registry of supported models
MODEL_REGISTRY = {
    "efficientnet_b0": {
        "module": "torchvision.models",
        "builder": "efficientnet_b0",
        "weights": "EfficientNet_B0_Weights.IMAGENET1K_V1",
        "num_features": 1280,
    },
    "efficientnet_b2": {
        "module": "torchvision.models",
        "builder": "efficientnet_b2",
        "weights": "EfficientNet_B2_Weights.IMAGENET1K_V1",
        "num_features": 1408,
    },
    "mobilenet_v3": {
        "module": "torchvision.models",
        "builder": "mobilenet_v3_small",
        "weights": "MobileNet_V3_Small_Weights.IMAGENET1K_V1",
        "num_features": 576,
    },
    "convnext_tiny": {
        "module": "torchvision.models",
        "builder": "convnext_tiny",
        "weights": "ConvNeXt_Tiny_Weights.IMAGENET1K_V1",
        "num_features": 768,
    },
    "resnet18": {
        "module": "torchvision.models",
        "builder": "resnet18",
        "weights": "ResNet18_Weights.IMAGENET1K_V1",
        "num_features": 512,
    },
}


def build_model(
    model_name: str = "efficientnet_b0",
    num_classes: int = NUM_CLASSES,
    pretrained: bool = True,
    dropout: float = 0.3,
) -> nn.Module:
    """Build a classification model.

    Args:
        model_name: Name of the model architecture.
        num_classes: Number of output classes.
        pretrained: Whether to load ImageNet pretrained weights.
        dropout: Dropout rate for the classifier head.

    Returns:
        PyTorch model with a custom classifier head.

    Raises:
        ValueError: If model_name is not in the registry.
    """
    if model_name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{model_name}'. "
            f"Available: {list(MODEL_REGISTRY.keys())}"
        )

    import importlib

    registry = MODEL_REGISTRY[model_name]
    module = importlib.import_module(registry["module"])
    builder_fn = getattr(module, registry["builder"])
    num_features = registry["num_features"]

    # Build with or without pretrained weights
    if pretrained:
        try:
            weights_enum = getattr(module, registry["weights"].split(".")[0])
            weights = getattr(weights_enum, registry["weights"].split(".")[1])
            model = builder_fn(weights=weights)
            logger.info(f"Loaded pretrained weights for {model_name}")
        except (AttributeError, ImportError):
            model = builder_fn(weights=None)
            logger.warning(f"Could not load pretrained weights, using random init")
    else:
        model = builder_fn(weights=None)

    # Replace classifier head
    if "efficientnet" in model_name:
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes),
        )
    elif "mobilenet" in model_name:
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes),
        )
    elif "convnext" in model_name:
        in_features = model.classifier[2].in_features
        model.classifier = nn.Sequential(
            nn.Flatten(1),
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes),
        )
    elif "resnet" in model_name:
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes),
        )

    logger.info(
        f"Built {model_name} with {num_classes} classes, "
        f"dropout={dropout}, pretrained={pretrained}"
    )
    return model


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)