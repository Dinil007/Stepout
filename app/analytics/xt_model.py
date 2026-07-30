"""Expected Threat model abstraction — supports rule-based and future learned grids."""
from __future__ import annotations

import logging
from typing import Protocol

from app.analytics.xt_grid import XTGrid
from app.analytics.xt_features import XTFeatures

LOGGER = logging.getLogger(__name__)


class XTModel(Protocol):
    """xT model boundary."""
    name: str

    def compute_xt_added(self, features: XTFeatures) -> float:
        """Return xT added for a single action."""


class RuleBasedXTModel:
    """Rule-based xT model using a predefined grid."""
    name = "rule_based"

    def __init__(self, grid: XTGrid) -> None:
        self.grid = grid

    def compute_xt_added(self, features: XTFeatures) -> float:
        return features.xt_added


class XTModelFactory:
    """Chooses the best available xT model."""

    @staticmethod
    def load(grid_key: str = "12x8") -> RuleBasedXTModel:
        grid = XTGrid(grid_key=grid_key)
        LOGGER.info("Using rule-based xT model with %s grid", grid_key)
        return RuleBasedXTModel(grid=grid)