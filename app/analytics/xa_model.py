"""Expected assists model abstractions."""
from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import List, Optional, Protocol

from app.analytics.xa_features import XAFeatures

LOGGER = logging.getLogger(__name__)


class XAModel(Protocol):
    """xA model boundary."""
    name: str

    def predict_proba(self, features: XAFeatures, vector: List[float]) -> float:
        """Return assist probability from 0.0 to 1.0."""


class RuleBasedXAModel:
    """Interpretable xA fallback used when no trained model exists."""
    name = "rule_based"

    def predict_proba(self, features: XAFeatures, vector: List[float]) -> float:
        shot_xg = max(features.shot_xg, 0.001)
        pass_len = min(features.pass_length_m / 40.0, 1.0)
        forward = max(0.0, min(features.forward_distance_m / 30.0, 1.0))
        receiver_dist = 1.0 / (1.0 + math.exp((features.receiver_distance_to_goal_m - 20.0) / 6.0))
        receiver_angle = min(features.receiver_angle_to_goal_deg / 50.0, 1.0)
        pressure = features.defensive_pressure
        progression = min(features.ball_progression_m / 25.0, 1.0)

        value = 0.02 + 0.30 * shot_xg + 0.10 * forward + 0.08 * pass_len
        value += 0.20 * receiver_dist + 0.15 * receiver_angle
        value += 0.10 * progression
        value -= 0.25 * pressure

        if features.pass_type and "Through Ball" in features.pass_type:
            value = min(value + 0.08, 0.95)
        if features.pass_length_m > 25.0:
            value = min(value + 0.05, 0.95)
        if shot_xg > 0.3:
            value = max(value, 0.15)

        return round(max(0.01, min(0.95, value)), 3)


class SklearnXAModel:
    """Loads a joblib sklearn model and optional scaler for xA."""
    name = "sklearn"

    def __init__(self, model_path: Path, scaler_path: Optional[Path] = None) -> None:
        try:
            import joblib
        except ImportError as exc:
            raise RuntimeError("joblib is required to load xA models") from exc

        self.model_path = model_path
        self.scaler_path = scaler_path
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path) if scaler_path and scaler_path.exists() else None

    def predict_proba(self, features: XAFeatures, vector: List[float]) -> float:
        rows = [vector]
        if self.scaler is not None:
            rows = self.scaler.transform(rows)
        if hasattr(self.model, "predict_proba"):
            probability = float(self.model.predict_proba(rows)[0][1])
        else:
            probability = float(self.model.predict(rows)[0])
        return round(max(0.0, min(1.0, probability)), 3)


class XAModelFactory:
    """Chooses the best available xA model."""

    def load(
        self,
        model_path: Path = Path("models/xa_model.pkl"),
        scaler_path: Path = Path("models/xa_scaler.pkl"),
        force_rule_based: bool = False,
    ) -> XAModel:
        if force_rule_based or not model_path.exists():
            LOGGER.info("Using rule-based xA model")
            return RuleBasedXAModel()
        try:
            LOGGER.info("Loading trained xA model from %s", model_path)
            return SklearnXAModel(model_path=model_path, scaler_path=scaler_path)
        except Exception as exc:
            LOGGER.warning("Falling back to rule-based xA model: %s", exc)
            return RuleBasedXAModel()