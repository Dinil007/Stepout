from __future__ import annotations

import time
from datetime import datetime, timezone

import pytest

from app.analytics.formation_engine import FormationEngine, FormationAnalysisResult
from app.analytics.formation_types import PlayerPosition


def _players(positions, team_id: int = 1, start_id: int = 1) -> list[PlayerPosition]:
    players = []
    now = datetime.now(timezone.utc)
    for idx, (x, y) in enumerate(positions):
        players.append(
            PlayerPosition(
                player_id=start_id + idx,
                team_id=team_id,
                team_name="Home" if team_id == 1 else "Away",
                jersey_number=idx + 1,
                x=x,
                y=y,
                frame_number=1,
                timestamp=now,
                confidence=1.0,
                is_goalkeeper=False,
                is_visible=True,
            )
        )
    return players


def test_analyze_returns_valid_result():
    engine = FormationEngine()
    players = _players([(0.2, 0.25), (0.4, 0.25), (0.6, 0.25), (0.8, 0.25), (0.3, 0.5), (0.5, 0.5), (0.7, 0.5), (0.25, 0.75), (0.5, 0.75), (0.75, 0.75)])
    result = engine.analyze(players, frame_number=10)
    assert isinstance(result, FormationAnalysisResult)
    assert result.detected_formation in engine.detector.registry.list_templates()
    assert 0.0 <= result.confidence <= 1.0
    assert result.frame_number == 10
    assert result.analysis_duration_seconds >= 0


def test_analyze_preserves_frame_number():
    engine = FormationEngine()
    players = _players([(0.2, 0.25), (0.4, 0.25), (0.6, 0.25), (0.8, 0.25), (0.3, 0.5), (0.5, 0.5), (0.7, 0.5), (0.25, 0.75), (0.5, 0.75), (0.75, 0.75)])
    result = engine.analyze(players, frame_number=42)
    assert result.frame_number == 42


def test_analyze_team_selects_correct_team():
    engine = FormationEngine()
    home = _players([(0.2, 0.25), (0.4, 0.25), (0.6, 0.25), (0.8, 0.25), (0.3, 0.5), (0.5, 0.5), (0.7, 0.5), (0.25, 0.75), (0.5, 0.75), (0.75, 0.75)], team_id=1)
    away = _players([(0.2, 0.25), (0.4, 0.25), (0.6, 0.25), (0.8, 0.25), (0.2, 0.5), (0.4, 0.5), (0.6, 0.5), (0.8, 0.5), (0.35, 0.78), (0.65, 0.78)], team_id=2, start_id=20)
    result = engine.analyze_team(home + away, team_id=2)
    assert result.team_id == 2
    assert result.detected_formation == "4-4-2"


def test_analyze_team_unknown_team_raises():
    engine = FormationEngine()
    players = _players([(0.2, 0.25), (0.4, 0.25), (0.6, 0.25), (0.8, 0.25), (0.3, 0.5), (0.5, 0.5), (0.7, 0.5), (0.25, 0.75), (0.5, 0.75), (0.75, 0.75)])
    with pytest.raises(ValueError):
        engine.analyze_team(players, team_id=99)


def test_analyze_match_independent_teams():
    engine = FormationEngine()
    home = _players([(0.2, 0.25), (0.4, 0.25), (0.6, 0.25), (0.8, 0.25), (0.3, 0.5), (0.5, 0.5), (0.7, 0.5), (0.25, 0.75), (0.5, 0.75), (0.75, 0.75)], team_id=1)
    away = _players([(0.2, 0.25), (0.4, 0.25), (0.6, 0.25), (0.8, 0.25), (0.2, 0.5), (0.4, 0.5), (0.6, 0.5), (0.8, 0.5), (0.35, 0.78), (0.65, 0.78)], team_id=2, start_id=20)
    results = engine.analyze_match(home + away)
    assert 1 in results and 2 in results
    assert results[1].detected_formation == "4-3-3"
    assert results[2].detected_formation == "4-4-2"


def test_batch_analyze_preserves_order():
    engine = FormationEngine()
    frames = [
        _players([(0.2, 0.25), (0.4, 0.25), (0.6, 0.25), (0.8, 0.25), (0.3, 0.5), (0.5, 0.5), (0.7, 0.5), (0.25, 0.75), (0.5, 0.75), (0.75, 0.75)]),
        _players([(0.2, 0.25), (0.4, 0.25), (0.6, 0.25), (0.8, 0.25), (0.2, 0.5), (0.4, 0.5), (0.6, 0.5), (0.8, 0.5), (0.35, 0.78), (0.65, 0.78)]),
    ]
    results = engine.batch_analyze(frames, frame_numbers=[0, 1])
    assert len(results) == 2
    assert results[0].detected_formation == "4-3-3"
    assert results[1].detected_formation == "4-4-2"
    assert results[0].frame_number == 0
    assert results[1].frame_number == 1


def test_batch_analyze_isolates_failures():
    engine = FormationEngine()
    good = _players([(0.2, 0.25), (0.4, 0.25), (0.6, 0.25), (0.8, 0.25), (0.3, 0.5), (0.5, 0.5), (0.7, 0.5), (0.25, 0.75), (0.5, 0.75), (0.75, 0.75)])
    bad = _players([(0.2, 0.25), (0.4, 0.25)])
    frames = [good, bad]
    results = engine.batch_analyze(frames)
    assert len(results) == 1


def test_reset_clears_results():
    engine = FormationEngine()
    players = _players([(0.2, 0.25), (0.4, 0.25), (0.6, 0.25), (0.8, 0.25), (0.3, 0.5), (0.5, 0.5), (0.7, 0.5), (0.25, 0.75), (0.5, 0.75), (0.75, 0.75)])
    engine.analyze(players)
    engine.reset()
    assert engine.summary() == {"count": 0, "formations": {}}


def test_summary_reports_counts():
    engine = FormationEngine()
    players = _players([(0.2, 0.25), (0.4, 0.25), (0.6, 0.25), (0.8, 0.25), (0.3, 0.5), (0.5, 0.5), (0.7, 0.5), (0.25, 0.75), (0.5, 0.75), (0.75, 0.75)])
    engine.analyze(players)
    stats = engine.summary()
    assert stats["count"] == 1
    assert "4-3-3" in stats["formations"]
    assert stats["formations"]["4-3-3"] == 1


def test_analyze_does_not_mutate_input():
    engine = FormationEngine()
    players = _players([(0.2, 0.25), (0.4, 0.25), (0.6, 0.25), (0.8, 0.25), (0.3, 0.5), (0.5, 0.5), (0.7, 0.5), (0.25, 0.75), (0.5, 0.75), (0.75, 0.75)])
    original = [(p.player_id, p.x, p.y) for p in players]
    engine.analyze(players)
    after = [(p.player_id, p.x, p.y) for p in players]
    assert original == after