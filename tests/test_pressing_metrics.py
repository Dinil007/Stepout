from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pytest

from app.analytics.pressing_config import PressingConfig
from app.analytics.pressing_metrics import PressingMetricsEngine
from app.analytics.pressing_types import (
    PPDAWindow,
    PressingMetrics,
    PressingSequence,
    PressingZone,
    PressureEvent,
)


@pytest.fixture
def engine() -> PressingMetricsEngine:
    return PressingMetricsEngine(config=PressingConfig())


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _make_event(
    attacker_id: int = 0,
    defender_id: int = 0,
    distance: float = 3.0,
    speed: float = 2.0,
    success: bool = False,
) -> PressureEvent:
    return PressureEvent(
        attacker_id=attacker_id,
        defender_id=defender_id,
        team_id=1,
        frame_number=0,
        timestamp=_now(),
        distance=distance,
        closing_speed=speed,
        pressure_angle=0.0,
        successful=success,
    )


def _make_sequence(events: list[PressureEvent] | None = None, duration: float = 2.0) -> PressingSequence:
    ts = _now()
    return PressingSequence(
        sequence_id=0,
        team_id=1,
        start_frame=0,
        end_frame=50,
        start_time=ts,
        end_time=ts,
        pressure_events=events or [],
        duration_seconds=duration,
    )


# ------------------------------------------------------------------
# calculate_metrics (full integration)
# ------------------------------------------------------------------

def test_calculate_metrics_empty(engine: PressingMetricsEngine):
    metrics = engine.calculate_metrics([], [])
    assert metrics.total_pressures == 0
    assert metrics.successful_pressures == 0
    assert metrics.pressure_success_rate == 0.0
    assert metrics.average_pressure_time == 0.0
    assert metrics.average_closing_speed == 0.0
    assert metrics.ppda == 0.0
    assert metrics.high_press_count == 0
    assert metrics.mid_block_count == 0
    assert metrics.low_block_count == 0
    assert metrics.is_valid() is True


def test_calculate_metrics_with_events(engine: PressingMetricsEngine):
    events = [
        _make_event(attacker_id=0, defender_id=0, distance=1.0, speed=2.0, success=True),
        _make_event(attacker_id=1, defender_id=1, distance=3.0, speed=1.5, success=False),
        _make_event(attacker_id=2, defender_id=2, distance=5.0, speed=3.0, success=True),
    ]
    seq = _make_sequence(events=events, duration=2.0)
    ppda = PPDAWindow(team_id=1, start_time=_now(), end_time=_now(),
                      passes_allowed=50, defensive_actions=10, ppda=5.0)
    metrics = engine.calculate_metrics(events, [seq], ppda_window=ppda)
    assert metrics.total_pressures == 3
    assert metrics.successful_pressures == 2
    assert metrics.pressure_success_rate == pytest.approx(2.0 / 3.0)
    assert metrics.average_pressure_time == 2.0
    assert metrics.average_closing_speed == pytest.approx((2.0 + 1.5 + 3.0) / 3.0)
    assert metrics.ppda == 5.0
    assert metrics.is_valid() is True


# ------------------------------------------------------------------
# calculate_ppda
# ------------------------------------------------------------------

def test_calculate_ppda_zero_actions(engine: PressingMetricsEngine):
    ppda = PPDAWindow(team_id=1, start_time=_now(), end_time=_now(),
                      passes_allowed=100, defensive_actions=0, ppda=0.0)
    assert engine.calculate_ppda(ppda) == 0.0


def test_calculate_ppda_normal(engine: PressingMetricsEngine):
    ppda = PPDAWindow(team_id=1, start_time=_now(), end_time=_now(),
                      passes_allowed=100, defensive_actions=20, ppda=5.0)
    assert engine.calculate_ppda(ppda) == 5.0


def test_calculate_ppda_from_scratch(engine: PressingMetricsEngine):
    # Recalculate even if window.ppda is different
    ppda = PPDAWindow(team_id=1, start_time=_now(), end_time=_now(),
                      passes_allowed=80, defensive_actions=16, ppda=99.0)
    assert engine.calculate_ppda(ppda) == 5.0  # 80/16


# ------------------------------------------------------------------
# calculate_pressure_success_rate
# ------------------------------------------------------------------

def test_success_rate_empty(engine: PressingMetricsEngine):
    assert engine.calculate_pressure_success_rate([]) == 0.0


def test_success_rate_all_fail(engine: PressingMetricsEngine):
    events = [
        _make_event(success=False),
        _make_event(success=False),
    ]
    assert engine.calculate_pressure_success_rate(events) == 0.0


def test_success_rate_all_success(engine: PressingMetricsEngine):
    events = [
        _make_event(success=True),
        _make_event(success=True),
        _make_event(success=True),
    ]
    assert engine.calculate_pressure_success_rate(events) == 1.0


def test_success_rate_mixed(engine: PressingMetricsEngine):
    events = [
        _make_event(success=True),
        _make_event(success=False),
        _make_event(success=True),
        _make_event(success=False),
    ]
    assert engine.calculate_pressure_success_rate(events) == 0.5


# ------------------------------------------------------------------
# calculate_average_closing_speed
# ------------------------------------------------------------------

def test_avg_closing_speed_empty(engine: PressingMetricsEngine):
    assert engine.calculate_average_closing_speed([]) == 0.0


def test_avg_closing_speed_single(engine: PressingMetricsEngine):
    events = [_make_event(speed=2.5)]
    assert engine.calculate_average_closing_speed(events) == 2.5


def test_avg_closing_speed_multiple(engine: PressingMetricsEngine):
    events = [
        _make_event(speed=1.0),
        _make_event(speed=2.0),
        _make_event(speed=3.0),
    ]
    assert engine.calculate_average_closing_speed(events) == 2.0


# ------------------------------------------------------------------
# calculate_average_pressure_time
# ------------------------------------------------------------------

def test_avg_pressure_time_empty(engine: PressingMetricsEngine):
    assert engine.calculate_average_pressure_time([]) == 0.0


def test_avg_pressure_time_single(engine: PressingMetricsEngine):
    seqs = [_make_sequence(duration=3.0)]
    assert engine.calculate_average_pressure_time(seqs) == 3.0


def test_avg_pressure_time_multiple(engine: PressingMetricsEngine):
    seqs = [
        _make_sequence(duration=1.0),
        _make_sequence(duration=2.0),
        _make_sequence(duration=3.0),
    ]
    assert engine.calculate_average_pressure_time(seqs) == 2.0


# ------------------------------------------------------------------
# calculate_zone_counts
# ------------------------------------------------------------------

def test_zone_counts_empty(engine: PressingMetricsEngine):
    counts = engine.calculate_zone_counts([])
    assert all(v == 0 for v in counts.values())
    assert set(counts.keys()) == {PressingZone.HIGH_PRESS, PressingZone.MID_BLOCK, PressingZone.LOW_BLOCK}


def test_zone_counts_distribution(engine: PressingMetricsEngine):
    events = [
        _make_event(distance=1.0),   # HIGH_PRESS (<=2.0)
        _make_event(distance=1.5),   # HIGH_PRESS
        _make_event(distance=3.0),   # MID_BLOCK (<=4.0)
        _make_event(distance=4.0),   # MID_BLOCK
        _make_event(distance=5.0),   # LOW_BLOCK (>4.0)
        _make_event(distance=10.0),  # LOW_BLOCK
    ]
    counts = engine.calculate_zone_counts(events)
    assert counts[PressingZone.HIGH_PRESS] == 2
    assert counts[PressingZone.MID_BLOCK] == 2
    assert counts[PressingZone.LOW_BLOCK] == 2


# ------------------------------------------------------------------
# classify_pressing_zone (metrics version)
# ------------------------------------------------------------------

def test_classify_high_press(engine: PressingMetricsEngine):
    assert engine.classify_pressing_zone(0.0) == PressingZone.HIGH_PRESS
    assert engine.classify_pressing_zone(2.0) == PressingZone.HIGH_PRESS


def test_classify_mid_block(engine: PressingMetricsEngine):
    assert engine.classify_pressing_zone(2.1) == PressingZone.MID_BLOCK
    assert engine.classify_pressing_zone(4.0) == PressingZone.MID_BLOCK


def test_classify_low_block(engine: PressingMetricsEngine):
    assert engine.classify_pressing_zone(4.1) == PressingZone.LOW_BLOCK
    assert engine.classify_pressing_zone(100.0) == PressingZone.LOW_BLOCK


# ------------------------------------------------------------------
# calculate_team_metrics
# ------------------------------------------------------------------

def test_calculate_team_metrics(engine: PressingMetricsEngine):
    events = [_make_event(success=True)]
    seq = _make_sequence(events=[events[0]], duration=1.0)
    metrics = engine.calculate_team_metrics(events, [seq])
    assert isinstance(metrics, PressingMetrics)
    assert metrics.total_pressures == 1


# ------------------------------------------------------------------
# reset
# ------------------------------------------------------------------

def test_reset(engine: PressingMetricsEngine):
    engine.reset()