"""Rule-based recommendation engine grounded in measurable analytics."""

from __future__ import annotations

from typing import List

from app.ai.schemas import JsonDict


class RecommendationEngine:
    """Generates measurable and clearly labelled heuristic recommendations."""

    def generate(self, context: JsonDict) -> JsonDict:
        return {
            "team": self._team_recommendations(context),
            "player": self._player_recommendations(context),
        }

    def _team_recommendations(self, context: JsonDict) -> JsonDict:
        recommendations: JsonDict = {}
        for team, data in context.get("teams", {}).items():
            items: List[JsonDict] = []
            accuracy = data.get("pass_accuracy_pct")
            possession = data.get("possession_pct")
            shape = data.get("tactical_shape", {})
            width = shape.get("width_m")
            compactness = shape.get("compactness")

            if accuracy is not None and accuracy < 70:
                items.append({
                    "recommendation": "Improve pass security in build-up phases.",
                    "basis": f"Measured pass accuracy is {accuracy}%.",
                    "support": "measured",
                })
            if possession is not None and possession < 40:
                items.append({
                    "recommendation": "Increase ball retention through shorter support angles.",
                    "basis": f"Measured possession share is {possession}%.",
                    "support": "measured",
                })
            if width is not None and width < 45:
                items.append({
                    "recommendation": "Increase width during possession.",
                    "basis": f"Measured team width is {width} m.",
                    "support": "measured",
                })
            if compactness is not None and compactness > 1.4:
                items.append({
                    "recommendation": "Maintain a more compact defensive shape.",
                    "basis": f"Measured compactness is {compactness}.",
                    "support": "measured",
                })
            if not items:
                items.append({
                    "recommendation": "Review transition speed and flank exploitation in video follow-up.",
                    "basis": "No threshold breach found in exported summaries.",
                    "support": "heuristic",
                })
            recommendations[team] = items
        return recommendations

    def _player_recommendations(self, context: JsonDict) -> JsonDict:
        recommendations: JsonDict = {}
        for player_id, data in context.get("players", {}).items():
            items: List[JsonDict] = []
            attempts = int(data.get("passes_attempted") or 0)
            accuracy = data.get("pass_accuracy_pct")
            shots = int(data.get("shots") or 0)
            movement_radius = data.get("movement_radius")

            if attempts > 0 and accuracy is not None and accuracy < 65:
                items.append({
                    "recommendation": "Reduce risky passes and improve passing selection.",
                    "basis": f"Measured pass accuracy is {accuracy}% from {attempts} attempts.",
                    "support": "measured",
                })
            if attempts == 0:
                items.append({
                    "recommendation": "Increase involvement as a passing option.",
                    "basis": "No pass attempts were exported for this player.",
                    "support": "measured",
                })
            if shots == 0:
                items.append({
                    "recommendation": "Increase attacking involvement where role permits.",
                    "basis": "No shots were exported for this player.",
                    "support": "heuristic",
                })
            if movement_radius is not None and movement_radius < 8:
                items.append({
                    "recommendation": "Create more dynamic movement to receive between lines.",
                    "basis": f"Measured movement radius is {movement_radius} m.",
                    "support": "measured",
                })
            recommendations[player_id] = items
        return recommendations
