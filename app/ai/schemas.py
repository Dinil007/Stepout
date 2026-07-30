"""Shared schemas and provider abstractions for the AI analyst layer."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol


JsonDict = Dict[str, Any]


@dataclass(frozen=True)
class AnalyticsArtifacts:
    """Structured analytics loaded from exported pipeline artifacts."""

    analytics: JsonDict = field(default_factory=dict)
    pass_summary: JsonDict = field(default_factory=dict)
    pass_events: List[JsonDict] = field(default_factory=list)
    shot_summary: JsonDict = field(default_factory=dict)
    shot_events: List[JsonDict] = field(default_factory=list)
    team_possession_summary: JsonDict = field(default_factory=dict)
    team_passing_summary: JsonDict = field(default_factory=dict)
    average_positions: JsonDict = field(default_factory=dict)
    xg_shots: List[JsonDict] = field(default_factory=list)
    team_xg_summary: JsonDict = field(default_factory=dict)
    player_xg_summary: JsonDict = field(default_factory=dict)
    xg_summary: JsonDict = field(default_factory=dict)
    xa_passes: List[JsonDict] = field(default_factory=list)
    team_xa_summary: JsonDict = field(default_factory=dict)
    player_xa_summary: JsonDict = field(default_factory=dict)
    xa_summary: JsonDict = field(default_factory=dict)
    xt_actions: List[JsonDict] = field(default_factory=list)
    team_xt_summary: JsonDict = field(default_factory=dict)
    player_xt_summary: JsonDict = field(default_factory=dict)
    xt_summary: JsonDict = field(default_factory=dict)
    formation_windows: List[JsonDict] = field(default_factory=list)
    formation_transitions: List[JsonDict] = field(default_factory=list)
    formation_metrics: JsonDict = field(default_factory=dict)
    pressing_events: List[JsonDict] = field(default_factory=list)
    pressing_sequences: List[JsonDict] = field(default_factory=list)
    pressing_metrics: JsonDict = field(default_factory=dict)
    pressing_detection: JsonDict = field(default_factory=dict)
    pressing_timeline: List[JsonDict] = field(default_factory=list)


@dataclass(frozen=True)
class MatchContext:
    """Clean context object consumed by prompts, reports, and validation."""

    match_id: str
    match: JsonDict
    teams: JsonDict
    players: JsonDict
    events: JsonDict
    tactical: JsonDict
    insights: JsonDict
    formation: JsonDict = field(default_factory=dict)
    pressing: JsonDict = field(default_factory=dict)

    def as_dict(self) -> JsonDict:
        return {
            "match_id": self.match_id,
            "match": self.match,
            "teams": self.teams,
            "players": self.players,
            "events": self.events,
            "tactical": self.tactical,
            "insights": self.insights,
            "formation": self.formation,
            "pressing": self.pressing,
        }


@dataclass(frozen=True)
class LLMResponse:
    """Normalized LLM response."""

    text: str
    provider: str
    model: str
    raw: Optional[JsonDict] = None


class LLMProvider(Protocol):
    """Provider boundary used by MatchAnalyst."""

    name: str
    model: str

    def generate(self, prompt: str, context: JsonDict) -> LLMResponse:
        """Generate an analysis response from a structured prompt and context."""


@dataclass(frozen=True)
class QueryRequest:
    """Simple request object for dashboard/internal callers."""

    question: str
