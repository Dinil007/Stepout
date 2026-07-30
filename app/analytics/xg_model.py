"""Expected goals model abstractions."""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Iterable, List, Optional, Protocol

from app.analytics.xg_features import XGFeatures

LOGGER = logging.getLogger(__name__)


class XGModel(Protocol):
    """xG model boundary."""

    name: str

    def predict_proba(self, features: XGFeatures, vector: List[float]) -> float:
        """Return a goal probability from 0.0 to 1.0."""


class RuleBasedXGModel:
    """Interpretable xG fallback used when no trained model exists."""

    name = "rule_based"

    def predict_proba(self, features: XGFeatures, vector: List[float]) -> float:
        distance = max(features.distance_m, 0.1)
        angle = max(features.angle_deg, 0.0)
        visibility = features.goal_mouth_visibility
        pressure = features.pressure_score
        speed_bonus = min(features.ball_speed_mps / 35.0, 1.0) * 0.05

        distance_component = 1.0 / (1.0 + math.exp((distance - 15.0) / 5.5))
        angle_component = min(angle / 55.0, 1.0)
        value = 0.03 + 0.52 * distance_component + 0.28 * angle_component
        value += 0.12 * visibility + speed_bonus
        value -= 0.22 * pressure

        if distance <= 10.0 and angle > 40.0:
            value = max(value, 0.45)
        if distance > 25.0:
            value = min(value, 0.12)
        return round(max(0.01, min(0.95, value)), 3)


class SklearnXGModel:
    """Loads a joblib sklearn model and optional scaler."""

    name = "sklearn"

    def __init__(self, model_path: Path, scaler_path: Optional[Path] = None) -> None:
        try:
            import joblib
        except ImportError as exc:  # pragma: no cover - dependency declared
            raise RuntimeError("joblib is required to load xG models") from exc

        self.model_path = model_path
        self.scaler_path = scaler_path
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path) if scaler_path and scaler_path.exists() else None

    def predict_proba(self, features: XGFeatures, vector: List[float]) -> float:
        rows = [vector]
        if self.scaler is not None:
            rows = self.scaler.transform(rows)
        if hasattr(self.model, "predict_proba"):
            probability = float(self.model.predict_proba(rows)[0][1])
        else:
            probability = float(self.model.predict(rows)[0])
        return round(max(0.0, min(1.0, probability)), 3)


class XGModelFactory:
    """Chooses the best available xG model."""

    def load(
        self,
        model_path: Path = Path("models/xg_model.pkl"),
        scaler_path: Path = Path("models/scaler.pkl"),
        force_rule_based: bool = False,
    ) -> XGModel:
        if force_rule_based or not model_path.exists():
            LOGGER.info("Using rule-based xG model")
            return RuleBasedXGModel()
        try:
            LOGGER.info("Loading trained xG model from %s", model_path)
            return SklearnXGModel(model_path=model_path, scaler_path=scaler_path)
        except Exception as exc:
            LOGGER.warning("Falling back to rule-based xG model: %s", exc)
            return RuleBasedXGModel()
