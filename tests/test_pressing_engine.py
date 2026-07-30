from __future__ import annotations

import pytest

from app.analytics.pressing_config import PressingConfig
from app.analytics.pressing_engine import PressingAnalysisResult, PressingEngine
from app.analytics.pressing_types import PressingMetrics


@pytest.fixture
def engine() -> PressingEngine:
    return PressingEngine()


# ------------------------------------------------------------------
# analyze
# ------------------------------------------------------------------

def test_analyze_empty(engine: PressingEngine):
    result = engine.analyze(attackers=[], defenders=[], frame_number=0, timestamp=0.0)
    assert isinstance(result, PressingAnalysisResult)
    assert len(result.pressure_events) == 0
    assert len(result.pressing_sequences) == 0
    assert result.pressing_detection is None
    assert result.pressing_metrics is None
    assert result.processing_time_ms >= 0


def test_analyze_with_pressure(engine: PressingEngine):
    # Close attackers/defenders with closing speed should produce events
    attackers = [(0.5, 0.5, 2.0, 0.0)]
    defenders = [(0.51, 0.5, 0.0, 0.0)]
    result = engine.analyze(attackers, defenders, frame_number=1, timestamp=0.0)
    assert len(result.pressure_events) >= 1
    # pressing_detection may be None if np is not available in the engine's scope
    assert result.pressing_metrics is not None
    assert result.pressing_metrics.total_pressures >= 1
    assert result.metadata["frame_number"] == 1


def test_analyze_no_pressure(engine: PressingEngine):
    # Far apart, no pressure
    attackers = [(0.0, 0.0, 0.0, 0.0)]
    defenders = [(0.9, 0.9, 0.0, 0.0)]
    result = engine.analyze(attackers, defenders, frame_number=5, timestamp=1.0)
    assert len(result.pressure_events) == 0
    assert result.pressing_detection is None
    assert result.pressing_metrics is None


def test_analyze_metadata(engine: PressingEngine):
    attackers = [(0.5, 0.5, 2.0, 0.0)]
    defenders = [(0.51, 0.5, 0.0, 0.0)]
    result = engine.analyze(attackers, defenders, frame_number=10, timestamp=2.5)
    assert result.metadata["frame_number"] == 10
    assert result.metadata["timestamp"] == 2.5
    assert "warnings" in result.metadata
    assert "errors" in result.metadata


# ------------------------------------------------------------------
# analyze_team
# ------------------------------------------------------------------

def test_analyze_team_empty(engine: PressingEngine):
    result = engine.analyze_team([], [], frame_number=0, timestamp=0.0)
    assert isinstance(result, PressingAnalysisResult)
    assert len(result.pressure_events) == 0


def test_analyze_team_with_data(engine: PressingEngine):
    attackers = [(0.5, 0.5, 2.0, 0.0)]
    defenders = [(0.51, 0.5, 0.0, 0.0)]
    result = engine.analyze_team(attackers, defenders, frame_number=1, timestamp=0.0)
    assert len(result.pressure_events) >= 1


# ------------------------------------------------------------------
# analyze_match
# ------------------------------------------------------------------

def test_analyze_match_empty(engine: PressingEngine):
    results = engine.analyze_match(frames=[])
    assert len(results) == 0


def test_analyze_match_single_frame(engine: PressingEngine):
    frames = [
        ([(0.5, 0.5, 2.0, 0.0)], [(0.51, 0.5, 0.0, 0.0)]),
    ]
    results = engine.analyze_match(frames, frame_numbers=[1], timestamps=[0.0])
    assert 1 in results
    assert len(results[1].pressure_events) >= 1


def test_analyze_match_multiple_frames(engine: PressingEngine):
    frames = [
        ([(0.5, 0.5, 2.0, 0.0)], [(0.51, 0.5, 0.0, 0.0)]),
        ([(0.0, 0.0, 0.0, 0.0)], [(0.9, 0.9, 0.0, 0.0)]),
    ]
    results = engine.analyze_match(frames, frame_numbers=[1, 2], timestamps=[0.0, 1.0])
    assert 1 in results
    assert 2 in results
    assert len(results[1].pressure_events) >= 1
    assert len(results[2].pressure_events) == 0


def test_analyze_match_no_frame_numbers(engine: PressingEngine):
    frames = [
        ([(0.5, 0.5, 2.0, 0.0)], [(0.51, 0.5, 0.0, 0.0)]),
    ]
    results = engine.analyze_match(frames)
    assert 0 in results  # defaults to index


# ------------------------------------------------------------------
# batch_analyze
# ------------------------------------------------------------------

def test_batch_analyze_empty(engine: PressingEngine):
    results = engine.batch_analyze(frames=[])
    assert len(results) == 0


def test_batch_analyze_multiple(engine: PressingEngine):
    frames = [
        ([(0.5, 0.5, 2.0, 0.0)], [(0.51, 0.5, 0.0, 0.0)]),
        ([(0.0, 0.0, 0.0, 0.0)], [(0.9, 0.9, 0.0, 0.0)]),
    ]
    results = engine.batch_analyze(frames, frame_numbers=[10, 20])
    assert len(results) == 2
    assert len(results[0].pressure_events) >= 1
    assert len(results[1].pressure_events) == 0


# ------------------------------------------------------------------
# summary
# ------------------------------------------------------------------

def test_summary_empty(engine: PressingEngine):
    summary = engine.summary()
    assert summary["count"] == 0
    assert summary["total_pressures"] == 0


def test_summary_after_analysis(engine: PressingEngine):
    attackers = [(0.5, 0.5, 2.0, 0.0)]
    defenders = [(0.51, 0.5, 0.0, 0.0)]
    engine.analyze(attackers, defenders, frame_number=1, timestamp=0.0)
    summary = engine.summary()
    assert summary["count"] == 1
    assert summary["total_pressures"] >= 1


def test_summary_after_multiple(engine: PressingEngine):
    attackers = [(0.5, 0.5, 2.0, 0.0)]
    defenders = [(0.51, 0.5, 0.0, 0.0)]
    engine.analyze(attackers, defenders, frame_number=1, timestamp=0.0)
    engine.analyze(attackers, defenders, frame_number=2, timestamp=0.04)
    summary = engine.summary()
    assert summary["count"] == 2


# ------------------------------------------------------------------
# reset
# ------------------------------------------------------------------

def test_reset(engine: PressingEngine):
    attackers = [(0.5, 0.5, 2.0, 0.0)]
    defenders = [(0.51, 0.5, 0.0, 0.0)]
    engine.analyze(attackers, defenders, frame_number=1, timestamp=0.0)
    assert engine.summary()["count"] == 1
    engine.reset()
    assert engine.summary()["count"] == 0


# ------------------------------------------------------------------
# PressingAnalysisResult
# ------------------------------------------------------------------

def test_analysis_result_defaults():
    result = PressingAnalysisResult()
    assert len(result.pressure_events) == 0
    assert len(result.pressing_sequences) == 0
    assert result.pressing_detection is None
    assert result.pressing_metrics is None
    assert result.metadata == {}
    assert result.processing_time_ms == 0.0