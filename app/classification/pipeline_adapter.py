"""Pipeline adapter for integrating person classification into the tracking pipeline.

This module bridges the person classifier with the existing pipeline stages
without modifying any existing code.

Usage:
    from app.classification.pipeline_adapter import classify_tracked_players

    # After tracking
    tracked_dets = player_tracker.update(detections, frame_shape, frame_no, frame)
    # Add classification
    for det in tracked_dets:
        det.team, det.confidence, _ = classifier.predict(crop)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from app.classification.config import CLASS_NAMES, InferenceConfig
from app.classification.inference import PersonClassifier

logger = logging.getLogger(__name__)

# Singleton classifier instance
_classifier: Optional[PersonClassifier] = None


def get_classifier(config: Optional[InferenceConfig] = None) -> PersonClassifier:
    """Get or create the singleton person classifier.

    Args:
        config: Optional inference configuration.

    Returns:
        PersonClassifier instance.
    """
    global _classifier
    if _classifier is None:
        _classifier = PersonClassifier(config=config)
    return _classifier


def classify_player_crop(
    frame: np.ndarray,
    bbox: Tuple[int, int, int, int],
    classifier: Optional[PersonClassifier] = None,
) -> Tuple[str, float, np.ndarray]:
    """Classify a single player crop from a video frame.

    Args:
        frame: Full video frame (BGR).
        bbox: Bounding box (x1, y1, x2, y2).
        classifier: PersonClassifier instance. Uses singleton if None.

    Returns:
        Tuple of (class_name, confidence, probabilities).
    """
    if classifier is None:
        classifier = get_classifier()

    x1, y1, x2, y2 = bbox
    # Ensure crop is within frame bounds
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(frame.shape[1] - 1, x2)
    y2 = min(frame.shape[0] - 1, y2)

    if x2 <= x1 or y2 <= y1:
        return "UNKNOWN", 0.0, np.zeros(len(CLASS_NAMES))

    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return "UNKNOWN", 0.0, np.zeros(len(CLASS_NAMES))

    return classifier.predict(crop)


def classify_tracked_players(
    frame: np.ndarray,
    tracked_detections: List,
    classifier: Optional[PersonClassifier] = None,
) -> None:
    """Add team/role classification to tracked detections IN PLACE.

    Modifies each detection object by setting 'team', 'classifier_confidence',
    and 'classifier_probs' attributes.

    Args:
        frame: Full video frame (BGR).
        tracked_detections: List of Detection objects from PlayerTracker.
        classifier: PersonClassifier instance. Uses singleton if None.
    """
    if classifier is None:
        classifier = get_classifier()

    for det in tracked_detections:
        bbox = getattr(det, "bbox", None)
        if bbox is None:
            continue

        team, conf, probs = classify_player_crop(frame, bbox, classifier)

        # Set attributes on the detection object
        det.team = team
        det.classifier_confidence = conf
        det.classifier_probs = probs.tolist() if hasattr(probs, "tolist") else list(probs)


def reset_classifier() -> None:
    """Reset the singleton classifier (useful for testing/reloading)."""
    global _classifier
    _classifier = None