from __future__ import annotations

from datetime import datetime, timezone

import pytest

from app.analytics.pressing_types import (
    PPDAWindow,
    PressingDetection,
    PressingMetrics,
    PressingSequence,
    PressingZone,
    PressureEvent,
)


def _now() -> datetime:
    return datetime.now(timezone.utc)


# ------------------------------------------------------------------
# PressureEvent
# ------------------------------------------------------------------

def test_pressure_event_valid():
    ts = _now()
    e = PressureEvent(
        attacker_id=1, defender_id=2, team_id=1,
        frame_number=10, timestamp=ts,
        distance=3.0, closing_speed=2.5,
        pressure_angle=0.5, successful=True,
    )
    assert e.is_valid() is True
    assert e.within_bounds() is True


def test_pressure_event_invalid_negative_frame():
    ts = _now()
    e = PressureEvent(
        attacker_id=1, defender_id=2, team_id=1,
        frame_number=-1, timestamp=ts,
        distance=3.0, closing_speed=2.5,
        pressure_angle=0.5, successful=True,
    )
    assert e.is_valid() is False


def test_pressure_event_invalid_negative_distance():
    ts = _now()
    e = PressureEvent(
        attacker_id=1, defender_id=2, team_id=1,
        frame_number=10, timestamp=ts,
        distance=-1.0, closing_speed=2.5,
        pressure_angle=0.5, successful=True,
    )
    assert e.is_valid() is False


def test_pressure_event_invalid_negative_speed():
    ts = _now()
    e = PressureEvent(
        attacker_id=1, defender_id=2, team_id=1,
        frame_number=10, timestamp=ts,
        distance=3.0, closing_speed=-0.5,
        pressure_angle=0.5, successful=True,
    )
    assert e.is_valid() is False


def test_pressure_event_negative_ids():
    ts = _now()
    e = PressureEvent(
        attacker_id=-1, defender_id=2, team_id=1,
        frame_number=10, timestamp=ts,
        distance=3.0, closing_speed=2.5,
        pressure_angle=0.5, successful=True,
    )
    assert e.is_valid() is False


def test_pressure_event_within_bounds_exceeds():
    ts = _now()
    e = PressureEvent(
        attacker_id=1, defender_id=2, team_id=1,
        frame_number=10, timestamp=ts,
        distance=150.0, closing_speed=2.5,
        pressure_angle=0.5, successful=True,
    )
    assert e.within_bounds(max_distance=100.0) is False


def test_pressure_event_within_bounds_speed_exceeds():
    ts = _now()
    e = PressureEvent(
        attacker_id=1, defender_id=2, team_id=1,
        frame_number=10, timestamp=ts,
        distance=3.0, closing_speed=15.0,
        pressure_angle=0.5, successful=True,
    )
    assert e.within_bounds(max_speed=10.0) is False


# ------------------------------------------------------------------
# PressingSequence
# ------------------------------------------------------------------

def test_pressing_sequence_valid():
    ts = _now()
    seq = PressingSequence(
        sequence_id=1, team_id=1,
        start_frame=0, end_frame=50,
        start_time=ts, end_time=ts,
        pressure_events=[], duration_seconds=2.0,
    )
    assert seq.is_valid() is True


def test_pressing_sequence_invalid_negative_frame():
    ts = _now()
    seq = PressingSequence(
        sequence_id=1, team_id=1,
        start_frame=-1, end_frame=50,
        start_time=ts, end_time=ts,
        pressure_events=[], duration_seconds=2.0,
    )
    assert seq.is_valid() is False


def test_pressing_sequence_invalid_end_before_start():
    ts = _now()
    seq = PressingSequence(
        sequence_id=1, team_id=1,
        start_frame=100, end_frame=50,
        start_time=ts, end_time=ts,
        pressure_events=[], duration_seconds=2.0,
    )
    assert seq.is_valid() is False


def test_pressing_sequence_invalid_duration():
    ts = _now()
    seq = PressingSequence(
        sequence_id=1, team_id=1,
        start_frame=0, end_frame=50,
        start_time=ts, end_time=ts,
        pressure_events=[], duration_seconds=-1.0,
    )
    assert seq.is_valid() is False


def test_pressing_sequence_invalid_time_order():
    ts = _now()
    later = datetime(2025, 1, 2, tzinfo=timezone.utc)
    earlier = datetime(2025, 1, 1, tzinfo=timezone.utc)
    seq = PressingSequence(
        sequence_id=1, team_id=1,
        start_frame=0, end_frame=50,
        start_time=later, end_time=earlier,
        pressure_events=[], duration_seconds=2.0,
    )
    assert seq.is_valid() is False


def test_pressing_sequence_event_count():
    ts = _now()
    events = [
        PressureEvent(attacker_id=1, defender_id=2, team_id=1, frame_number=10,
                      timestamp=ts, distance=3.0, closing_speed=2.0,
                      pressure_angle=0.0, successful=False),
    ]
    seq = PressingSequence(
        sequence_id=1, team_id=1,
        start_frame=0, end_frame=50,
        start_time=ts, end_time=ts,
        pressure_events=events, duration_seconds=2.0,
    )
    assert seq.event_count() == 1


def test_pressing_sequence_event_count_empty():
    ts = _now()
    seq = PressingSequence(
        sequence_id=1, team_id=1,
        start_frame=0, end_frame=50,
        start_time=ts, end_time=ts,
        pressure_events=[], duration_seconds=2.0,
    )
    assert seq.event_count() == 0


# ------------------------------------------------------------------
# PPDAWindow
# ------------------------------------------------------------------

def test_ppda_window_valid():
    ts = _now()
    w = PPDAWindow(
        team_id=1,
        start_time=ts, end_time=ts,
        passes_allowed=100, defensive_actions=20, ppda=5.0,
    )
    assert w.is_valid() is True


def test_ppda_window_negative_passes():
    ts = _now()
    w = PPDAWindow(
        team_id=1,
        start_time=ts, end_time=ts,
        passes_allowed=-1, defensive_actions=20, ppda=5.0,
    )
    assert w.is_valid() is False


def test_ppda_window_negative_actions():
    ts = _now()
    w = PPDAWindow(
        team_id=1,
        start_time=ts, end_time=ts,
        passes_allowed=100, defensive_actions=-5, ppda=5.0,
    )
    assert w.is_valid() is False


def test_ppda_window_negative_ppda():
    ts = _now()
    w = PPDAWindow(
        team_id=1,
        start_time=ts, end_time=ts,
        passes_allowed=100, defensive_actions=20, ppda=-1.0,
    )
    assert w.is_valid() is False


def test_ppda_window_invalid_time_order():
    ts = _now()
    later = datetime(2025, 1, 2, tzinfo=timezone.utc)
    earlier = datetime(2025, 1, 1, tzinfo=timezone.utc)
    w = PPDAWindow(
        team_id=1,
        start_time=later, end_time=earlier,
        passes_allowed=100, defensive_actions=20, ppda=5.0,
    )
    assert w.is_valid() is False


# ------------------------------------------------------------------
# PressingMetrics
# ------------------------------------------------------------------

def test_pressing_metrics_valid():
    m = PressingMetrics(
        total_pressures=50,
        successful_pressures=25,
        pressure_success_rate=0.5,
        average_pressure_time=2.0,
        average_closing_speed=1.5,
        ppda=8.0,
        high_press_count=20,
        mid_block_count=15,
        low_block_count=15,
    )
    assert m.is_valid() is True


def test_pressing_metrics_invalid_negative_total():
    m = PressingMetrics(
        total_pressures=-1,
        successful_pressures=0,
        pressure_success_rate=0.0,
        average_pressure_time=0.0,
        average_closing_speed=0.0,
        ppda=0.0,
        high_press_count=0,
        mid_block_count=0,
        low_block_count=0,
    )
    assert m.is_valid() is False


def test_pressing_metrics_invalid_success_exceeds_total():
    m = PressingMetrics(
        total_pressures=10,
        successful_pressures=20,
        pressure_success_rate=2.0,
        average_pressure_time=0.0,
        average_closing_speed=0.0,
        ppda=0.0,
        high_press_count=0,
        mid_block_count=0,
        low_block_count=0,
    )
    assert m.is_valid() is False


def test_pressing_metrics_invalid_success_rate_out_of_range():
    m = PressingMetrics(
        total_pressures=10,
        successful_pressures=5,
        pressure_success_rate=2.0,
        average_pressure_time=1.0,
        average_closing_speed=1.0,
        ppda=5.0,
        high_press_count=5,
        mid_block_count=3,
        low_block_count=2,
    )
    assert m.is_valid() is False


def test_pressing_metrics_negative_averages():
    m = PressingMetrics(
        total_pressures=10,
        successful_pressures=5,
        pressure_success_rate=0.5,
        average_pressure_time=-1.0,
        average_closing_speed=1.0,
        ppda=5.0,
        high_press_count=5,
        mid_block_count=3,
        low_block_count=2,
    )
    assert m.is_valid() is False


def test_pressing_metrics_negative_zone_counts():
    m = PressingMetrics(
        total_pressures=10,
        successful_pressures=5,
        pressure_success_rate=0.5,
        average_pressure_time=1.0,
        average_closing_speed=1.0,
        ppda=5.0,
        high_press_count=-1,
        mid_block_count=0,
        low_block_count=0,
    )
    assert m.is_valid() is False


def test_pressing_metrics_pressure_distribution():
    m = PressingMetrics(
        total_pressures=50,
        successful_pressures=25,
        pressure_success_rate=0.5,
        average_pressure_time=2.0,
        average_closing_speed=1.5,
        ppda=8.0,
        high_press_count=20,
        mid_block_count=15,
        low_block_count=15,
    )
    dist = m.pressure_distribution()
    assert dist[PressingZone.HIGH_PRESS] == 20
    assert dist[PressingZone.MID_BLOCK] == 15
    assert dist[PressingZone.LOW_BLOCK] == 15


# ------------------------------------------------------------------
# PressingDetection
# ------------------------------------------------------------------

def test_pressing_detection_valid():
    ts = _now()
    d = PressingDetection(
        pressing_style=PressingZone.HIGH_PRESS,
        confidence=0.9,
        frame_number=10,
        timestamp=ts,
    )
    assert d.is_valid() is True


def test_pressing_detection_invalid_confidence_above():
    ts = _now()
    d = PressingDetection(
        pressing_style=PressingZone.HIGH_PRESS,
        confidence=1.5,
        frame_number=10,
        timestamp=ts,
    )
    assert d.is_valid() is False


def test_pressing_detection_invalid_confidence_below():
    ts = _now()
    d = PressingDetection(
        pressing_style=PressingZone.HIGH_PRESS,
        confidence=-0.1,
        frame_number=10,
        timestamp=ts,
    )
    assert d.is_valid() is False


def test_pressing_detection_invalid_negative_frame():
    ts = _now()
    d = PressingDetection(
        pressing_style=PressingZone.HIGH_PRESS,
        confidence=0.8,
        frame_number=-5,
        timestamp=ts,
    )
    assert d.is_valid() is False


def test_pressing_detection_invalid_zone_type():
    ts = _now()
    d = PressingDetection(
        pressing_style="invalid_zone",  # type: ignore[arg-type]
        confidence=0.8,
        frame_number=10,
        timestamp=ts,
    )
    assert d.is_valid() is False


# ------------------------------------------------------------------
# PressingZone enum
# ------------------------------------------------------------------

def test_pressing_zone_values():
    assert PressingZone.HIGH_PRESS.value == "high_press"
    assert PressingZone.MID_BLOCK.value == "mid_block"
    assert PressingZone.LOW_BLOCK.value == "low_block"


def test_pressing_zone_distinct():
    zones = {PressingZone.HIGH_PRESS, PressingZone.MID_BLOCK, PressingZone.LOW_BLOCK}
    assert len(zones) == 3