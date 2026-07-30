from __future__ import annotations

from datetime import datetime, timezone

import pytest

from app.analytics.pressing_engine import PressingAnalysisResult
from app.analytics.pressing_types import (
    PPDAWindow,
    PressingDetection,
    PressingMetrics,
    PressingSequence,
    PressingZone,
    PressureEvent,
)
from app.analytics.pressing_validation import PressingValidator, ValidationReport


@pytest.fixture
def validator() -> PressingValidator:
    return PressingValidator()


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _make_event(
    attacker_id: int = 1,
    defender_id: int = 2,
    distance: float = 3.0,
    speed: float = 2.0,
    success: bool = False,
    frame: int = 10,
) -> PressureEvent:
    return PressureEvent(
        attacker_id=attacker_id,
        defender_id=defender_id,
        team_id=1,
        frame_number=frame,
        timestamp=_now(),
        distance=distance,
        closing_speed=speed,
        pressure_angle=0.0,
        successful=success,
    )


def _make_sequence(events: list[PressureEvent] | None = None) -> PressingSequence:
    ts = _now()
    return PressingSequence(
        sequence_id=0,
        team_id=1,
        start_frame=0,
        end_frame=50,
        start_time=ts,
        end_time=ts,
        pressure_events=events or [],
        duration_seconds=2.0,
    )


# ------------------------------------------------------------------
# ValidationReport
# ------------------------------------------------------------------

def test_validation_report_defaults():
    r = ValidationReport()
    assert r.overall_valid is True
    assert r.errors == []
    assert r.warnings == []
    assert r.checked_items == 0
    assert r.passed_items == 0


def test_validation_report_add_error():
    r = ValidationReport()
    r.add_error("Something went wrong")
    assert r.overall_valid is False
    assert len(r.errors) == 1
    assert r.checked_items == 1
    assert r.passed_items == 0


def test_validation_report_add_warning():
    r = ValidationReport()
    r.add_warning("Something suspicious")
    assert r.overall_valid is True  # warnings don't invalidate
    assert len(r.warnings) == 1
    assert r.checked_items == 1
    assert r.passed_items == 0


def test_validation_report_add_pass():
    r = ValidationReport()
    r.add_pass()
    assert r.checked_items == 1
    assert r.passed_items == 1


# ------------------------------------------------------------------
# validate_pressure_event
# ------------------------------------------------------------------

def test_validate_pressure_event_valid(validator: PressingValidator):
    event = _make_event()
    report = validator.validate_pressure_event(event)
    assert report.overall_valid is True


def test_validate_pressure_event_invalid_frame(validator: PressingValidator):
    event = _make_event(frame=-1)
    report = validator.validate_pressure_event(event)
    assert report.overall_valid is False
    assert any("frame_number" in e for e in report.errors)


def test_validate_pressure_event_negative_distance(validator: PressingValidator):
    event = _make_event(distance=-1.0)
    report = validator.validate_pressure_event(event)
    assert report.overall_valid is False


def test_validate_pressure_event_negative_speed(validator: PressingValidator):
    event = _make_event(speed=-0.5)
    report = validator.validate_pressure_event(event)
    assert report.overall_valid is False


# ------------------------------------------------------------------
# validate_sequence
# ------------------------------------------------------------------

def test_validate_sequence_valid(validator: PressingValidator):
    seq = _make_sequence()
    report = validator.validate_sequence(seq)
    assert report.overall_valid is True


def test_validate_sequence_invalid_order(validator: PressingValidator):
    ts = _now()
    seq = PressingSequence(
        sequence_id=1, team_id=1,
        start_frame=100, end_frame=50,
        start_time=ts, end_time=ts,
        pressure_events=[], duration_seconds=2.0,
    )
    report = validator.validate_sequence(seq)
    assert report.overall_valid is False
    assert any("start_frame" in e for e in report.errors)


def test_validate_sequence_negative_duration(validator: PressingValidator):
    ts = _now()
    seq = PressingSequence(
        sequence_id=1, team_id=1,
        start_frame=0, end_frame=50,
        start_time=ts, end_time=ts,
        pressure_events=[], duration_seconds=-1.0,
    )
    report = validator.validate_sequence(seq)
    assert report.overall_valid is False


def test_validate_sequence_invalid_time(validator: PressingValidator):
    later = datetime(2025, 1, 2, tzinfo=timezone.utc)
    earlier = datetime(2025, 1, 1, tzinfo=timezone.utc)
    seq = PressingSequence(
        sequence_id=1, team_id=1,
        start_frame=0, end_frame=50,
        start_time=later, end_time=earlier,
        pressure_events=[], duration_seconds=2.0,
    )
    report = validator.validate_sequence(seq)
    assert report.overall_valid is False


# ------------------------------------------------------------------
# validate_ppda_window
# ------------------------------------------------------------------

def test_validate_ppda_window_valid(validator: PressingValidator):
    ts = _now()
    w = PPDAWindow(team_id=1, start_time=ts, end_time=ts,
                   passes_allowed=100, defensive_actions=20, ppda=5.0)
    report = validator.validate_ppda_window(w)
    assert report.overall_valid is True


def test_validate_ppda_window_negative_passes(validator: PressingValidator):
    ts = _now()
    w = PPDAWindow(team_id=1, start_time=ts, end_time=ts,
                   passes_allowed=-1, defensive_actions=20, ppda=5.0)
    report = validator.validate_ppda_window(w)
    assert report.overall_valid is False


def test_validate_ppda_window_negative_actions(validator: PressingValidator):
    ts = _now()
    w = PPDAWindow(team_id=1, start_time=ts, end_time=ts,
                   passes_allowed=100, defensive_actions=-5, ppda=5.0)
    report = validator.validate_ppda_window(w)
    assert report.overall_valid is False


def test_validate_ppda_window_invalid_time(validator: PressingValidator):
    later = datetime(2025, 1, 2, tzinfo=timezone.utc)
    earlier = datetime(2025, 1, 1, tzinfo=timezone.utc)
    w = PPDAWindow(team_id=1, start_time=later, end_time=earlier,
                   passes_allowed=100, defensive_actions=20, ppda=5.0)
    report = validator.validate_ppda_window(w)
    assert report.overall_valid is False


# ------------------------------------------------------------------
# validate_metrics
# ------------------------------------------------------------------

def test_validate_metrics_valid(validator: PressingValidator):
    m = PressingMetrics(
        total_pressures=50, successful_pressures=25,
        pressure_success_rate=0.5, average_pressure_time=2.0,
        average_closing_speed=1.5, ppda=8.0,
        high_press_count=20, mid_block_count=15, low_block_count=15,
    )
    report = validator.validate_metrics(m)
    assert report.overall_valid is True


def test_validate_metrics_negative_total(validator: PressingValidator):
    m = PressingMetrics(
        total_pressures=-1, successful_pressures=0,
        pressure_success_rate=0.0, average_pressure_time=0.0,
        average_closing_speed=0.0, ppda=0.0,
        high_press_count=0, mid_block_count=0, low_block_count=0,
    )
    report = validator.validate_metrics(m)
    assert report.overall_valid is False


def test_validate_metrics_success_exceeds_total(validator: PressingValidator):
    m = PressingMetrics(
        total_pressures=10, successful_pressures=20,
        pressure_success_rate=2.0, average_pressure_time=0.0,
        average_closing_speed=0.0, ppda=0.0,
        high_press_count=0, mid_block_count=0, low_block_count=0,
    )
    report = validator.validate_metrics(m)
    assert report.overall_valid is False


def test_validate_metrics_invalid_success_rate(validator: PressingValidator):
    m = PressingMetrics(
        total_pressures=10, successful_pressures=5,
        pressure_success_rate=1.5, average_pressure_time=1.0,
        average_closing_speed=1.0, ppda=5.0,
        high_press_count=5, mid_block_count=3, low_block_count=2,
    )
    report = validator.validate_metrics(m)
    assert report.overall_valid is False


def test_validate_metrics_negative_zone_counts(validator: PressingValidator):
    m = PressingMetrics(
        total_pressures=10, successful_pressures=5,
        pressure_success_rate=0.5, average_pressure_time=1.0,
        average_closing_speed=1.0, ppda=5.0,
        high_press_count=-1, mid_block_count=0, low_block_count=0,
    )
    report = validator.validate_metrics(m)
    assert report.overall_valid is False


# ------------------------------------------------------------------
# validate_detection
# ------------------------------------------------------------------

def test_validate_detection_valid(validator: PressingValidator):
    d = PressingDetection(
        pressing_style=PressingZone.HIGH_PRESS,
        confidence=0.9, frame_number=10, timestamp=_now(),
    )
    report = validator.validate_detection(d)
    assert report.overall_valid is True


def test_validate_detection_invalid_confidence(validator: PressingValidator):
    d = PressingDetection(
        pressing_style=PressingZone.HIGH_PRESS,
        confidence=1.5, frame_number=10, timestamp=_now(),
    )
    report = validator.validate_detection(d)
    assert report.overall_valid is False


def test_validate_detection_negative_frame(validator: PressingValidator):
    d = PressingDetection(
        pressing_style=PressingZone.HIGH_PRESS,
        confidence=0.8, frame_number=-5, timestamp=_now(),
    )
    report = validator.validate_detection(d)
    assert report.overall_valid is False


# ------------------------------------------------------------------
# validate_analysis
# ------------------------------------------------------------------

def test_validate_analysis_valid(validator: PressingValidator):
    metrics = PressingMetrics(
        total_pressures=10, successful_pressures=5,
        pressure_success_rate=0.5, average_pressure_time=1.0,
        average_closing_speed=1.0, ppda=5.0,
        high_press_count=5, mid_block_count=3, low_block_count=2,
    )
    detection = PressingDetection(
        pressing_style=PressingZone.HIGH_PRESS,
        confidence=0.9, frame_number=10, timestamp=_now(),
    )
    analysis = PressingAnalysisResult(
        pressure_events=[_make_event()],
        pressing_sequences=[_make_sequence()],
        pressing_detection=detection,
        pressing_metrics=metrics,
        metadata={"frame": 10},
        processing_time_ms=5.0,
    )
    report = validator.validate_analysis(analysis)
    assert report.overall_valid is True


def test_validate_analysis_empty_metadata_warning(validator: PressingValidator):
    analysis = PressingAnalysisResult(
        pressure_events=[], pressing_sequences=[],
        metadata={}, processing_time_ms=1.0,
    )
    report = validator.validate_analysis(analysis)
    assert report.overall_valid is True  # warning only
    assert len(report.warnings) >= 1


def test_validate_analysis_negative_time(validator: PressingValidator):
    analysis = PressingAnalysisResult(
        pressure_events=[], pressing_sequences=[],
        metadata={"frame": 1}, processing_time_ms=-1.0,
    )
    report = validator.validate_analysis(analysis)
    assert report.overall_valid is False


def test_validate_analysis_with_invalid_metrics(validator: PressingValidator):
    metrics = PressingMetrics(
        total_pressures=10, successful_pressures=20,
        pressure_success_rate=2.0, average_pressure_time=0.0,
        average_closing_speed=0.0, ppda=0.0,
        high_press_count=0, mid_block_count=0, low_block_count=0,
    )
    analysis = PressingAnalysisResult(
        pressure_events=[_make_event()],
        pressing_sequences=[],
        pressing_metrics=metrics,
        metadata={"frame": 1},
        processing_time_ms=1.0,
    )
    report = validator.validate_analysis(analysis)
    assert report.overall_valid is False


# ------------------------------------------------------------------
# validate_batch
# ------------------------------------------------------------------

def test_validate_batch_empty(validator: PressingValidator):
    report = validator.validate_batch([])
    assert report.overall_valid is True
    assert len(report.warnings) >= 1  # empty batch warning


def test_validate_batch_valid(validator: PressingValidator):
    analysis = PressingAnalysisResult(
        pressure_events=[], pressing_sequences=[],
        metadata={"frame": 1}, processing_time_ms=1.0,
    )
    report = validator.validate_batch([analysis, analysis])
    assert report.overall_valid is True


def test_validate_batch_with_errors(validator: PressingValidator):
    bad = PressingAnalysisResult(
        pressure_events=[], pressing_sequences=[],
        metadata={"frame": 1}, processing_time_ms=-1.0,
    )
    good = PressingAnalysisResult(
        pressure_events=[], pressing_sequences=[],
        metadata={"frame": 2}, processing_time_ms=1.0,
    )
    report = validator.validate_batch([good, bad])
    assert report.overall_valid is False


# ------------------------------------------------------------------
# reset
# ------------------------------------------------------------------

def test_reset(validator: PressingValidator):
    validator.reset()