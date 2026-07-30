from __future__ import annotations

import math

import numpy as np
import pytest

from app.analytics.pressing_config import PressingConfig
from app.analytics.pressing_detector import PressingDetector
from app.analytics.pressing_types import PressingZone


@pytest.fixture
def detector() -> PressingDetector:
    return PressingDetector(config=PressingConfig())


# ------------------------------------------------------------------
# detect_pressure_events
# ------------------------------------------------------------------

def test_detect_pressure_events_empty_attackers(detector: PressingDetector):
    events = detector.detect_pressure_events(
        attackers=[], defenders=[(0.5, 0.5, 0.0, 0.0)],
        frame_number=1, timestamp=0.0,
    )
    assert len(events) == 0


def test_detect_pressure_events_empty_defenders(detector: PressingDetector):
    events = detector.detect_pressure_events(
        attackers=[(0.5, 0.5, 0.0, 0.0)], defenders=[],
        frame_number=1, timestamp=0.0,
    )
    assert len(events) == 0


def test_detect_pressure_events_both_empty(detector: PressingDetector):
    events = detector.detect_pressure_events(
        attackers=[], defenders=[],
        frame_number=1, timestamp=0.0,
    )
    assert len(events) == 0


def test_detect_pressure_events_close_proximity(detector: PressingDetector):
    # attacker and defender very close with closing speed
    events = detector.detect_pressure_events(
        attackers=[(0.5, 0.5, 2.0, 0.0)],
        defenders=[(0.51, 0.5, 0.0, 0.0)],
        frame_number=1, timestamp=0.0,
    )
    assert len(events) == 1
    e = events[0]
    assert e.attacker_id == 0
    assert e.defender_id == 0
    assert e.frame_number == 1
    assert e.distance > 0
    assert e.closing_speed > 0


def test_detect_pressure_events_too_far(detector: PressingDetector):
    # attacker and defender far apart
    events = detector.detect_pressure_events(
        attackers=[(0.0, 0.0, 2.0, 0.0)],
        defenders=[(0.9, 0.9, 0.0, 0.0)],
        frame_number=1, timestamp=0.0,
    )
    assert len(events) == 0


def test_detect_pressure_events_low_closing_speed(detector: PressingDetector):
    # close but no closing speed
    events = detector.detect_pressure_events(
        attackers=[(0.5, 0.5, 0.0, 0.0)],
        defenders=[(0.51, 0.5, 0.0, 0.0)],
        frame_number=1, timestamp=0.0,
    )
    assert len(events) == 0


def test_detect_pressure_events_multiple_pairs(detector: PressingDetector):
    attackers = [(0.5, 0.5, 2.0, 0.0), (0.3, 0.3, 1.5, 0.0)]
    defenders = [(0.51, 0.5, 0.0, 0.0), (0.31, 0.3, 0.0, 0.0)]
    events = detector.detect_pressure_events(
        attackers=attackers, defenders=defenders,
        frame_number=5, timestamp=1.0,
    )
    assert len(events) == 2
    for e in events:
        assert e.frame_number == 5


def test_detect_pressure_events_angle_computation(detector: PressingDetector):
    # defender to the right of attacker -> angle should be ~0
    # distance = 0.01 * 100 = 1.0, well within threshold of 5.0
    events = detector.detect_pressure_events(
        attackers=[(0.5, 0.5, 2.0, 0.0)],
        defenders=[(0.51, 0.5, 0.0, 0.0)],
        frame_number=1, timestamp=0.0,
    )
    assert len(events) == 1
    # dx positive, dy zero -> atan2(0, positive) = 0
    assert abs(events[0].pressure_angle) < 0.01


def test_detect_pressure_events_no_velocity_columns(detector: PressingDetector):
    # Only (x, y) without velocity
    events = detector.detect_pressure_events(
        attackers=[(0.5, 0.5)],
        defenders=[(0.51, 0.5)],
        frame_number=1, timestamp=0.0,
    )
    # Without velocity, closing_speed will be 0 -> filtered out
    assert len(events) == 0


# ------------------------------------------------------------------
# detect_pressing_sequences
# ------------------------------------------------------------------

def test_detect_pressing_sequences_empty(detector: PressingDetector):
    seqs = detector.detect_pressing_sequences({})
    assert len(seqs) == 0


def test_detect_pressing_sequences_single_frame(detector: PressingDetector):
    # Single frame with events, but duration < minimum -> no sequence
    events = [
        _make_event(attacker_id=0, defender_id=0, frame_number=0, timestamp=0.0),
    ]
    seqs = detector.detect_pressing_sequences({0: events}, frame_rate=25.0)
    assert len(seqs) == 0


def test_detect_pressing_sequences_continuous_frames(detector: PressingDetector):
    # Multiple consecutive frames exceeding minimum duration
    events_by_frame: dict[int, list] = {}
    for f in range(50):
        events_by_frame[f] = [
            _make_event(attacker_id=0, defender_id=0, frame_number=f, timestamp=f / 25.0),
        ]
    seqs = detector.detect_pressing_sequences(events_by_frame, frame_rate=25.0)
    # 50 frames at 25 fps = 2.0 seconds >= 1.5 minimum
    assert len(seqs) >= 1
    assert seqs[0].duration_seconds >= 1.5


def test_detect_pressing_sequences_gap_breaks_sequence(detector: PressingDetector):
    events_by_frame: dict[int, list] = {}
    for f in range(10):
        events_by_frame[f] = [
            _make_event(attacker_id=0, defender_id=0, frame_number=f, timestamp=f / 25.0),
        ]
    # gap
    for f in range(20, 30):
        events_by_frame[f] = [
            _make_event(attacker_id=0, defender_id=0, frame_number=f, timestamp=f / 25.0),
        ]
    seqs = detector.detect_pressing_sequences(events_by_frame, frame_rate=25.0)
    # Each block is 10 frames = 0.4s < 1.5s minimum -> no sequences
    assert len(seqs) == 0


def test_detect_pressing_sequences_multiple_sequences(detector: PressingConfig):
    # Use a lower minimum duration to get multiple sequences
    cfg = PressingConfig(minimum_pressure_duration=0.2)
    det = PressingDetector(config=cfg)
    events_by_frame: dict[int, list] = {}
    for f in range(10):
        events_by_frame[f] = [
            _make_event(attacker_id=0, defender_id=0, frame_number=f, timestamp=f / 25.0),
        ]
    for f in range(30, 40):
        events_by_frame[f] = [
            _make_event(attacker_id=1, defender_id=1, frame_number=f, timestamp=f / 25.0),
        ]
    seqs = det.detect_pressing_sequences(events_by_frame, frame_rate=25.0)
    # 10 frames = 0.4s >= 0.2s -> 2 sequences
    assert len(seqs) == 2
    assert seqs[0].sequence_id == 0
    assert seqs[1].sequence_id == 1


def test_detect_pressing_sequences_zero_frame_rate(detector: PressingDetector):
    events_by_frame = {0: [_make_event(attacker_id=0, defender_id=0, frame_number=0, timestamp=0.0)]}
    seqs = detector.detect_pressing_sequences(events_by_frame, frame_rate=0.0)
    assert len(seqs) == 0


# ------------------------------------------------------------------
# classify_pressing_zone
# ------------------------------------------------------------------

def test_classify_pressing_zone_high(detector: PressingDetector):
    assert detector.classify_pressing_zone(0.1) == PressingZone.HIGH_PRESS
    assert detector.classify_pressing_zone(0.35) == PressingZone.HIGH_PRESS


def test_classify_pressing_zone_mid(detector: PressingDetector):
    assert detector.classify_pressing_zone(0.4) == PressingZone.MID_BLOCK
    assert detector.classify_pressing_zone(0.65) == PressingZone.MID_BLOCK


def test_classify_pressing_zone_low(detector: PressingDetector):
    assert detector.classify_pressing_zone(0.7) == PressingZone.LOW_BLOCK
    assert detector.classify_pressing_zone(0.9) == PressingZone.LOW_BLOCK
    assert detector.classify_pressing_zone(1.0) == PressingZone.LOW_BLOCK


# ------------------------------------------------------------------
# calculate_confidence
# ------------------------------------------------------------------

def test_calculate_confidence_empty(detector: PressingDetector):
    assert detector.calculate_confidence([], []) == 0.0


def test_calculate_confidence_no_events(detector: PressingDetector):
    seq = _make_sequence(sequence_id=0, start=0, end=10)
    assert detector.calculate_confidence([], [seq]) == 0.0


def test_calculate_confidence_no_sequences(detector: PressingDetector):
    event = _make_event(attacker_id=0, defender_id=0, frame_number=0, timestamp=0.0)
    assert detector.calculate_confidence([event], []) == 0.0


def test_calculate_confidence_with_data(detector: PressingDetector):
    events = [
        _make_event(attacker_id=0, defender_id=0, frame_number=0, timestamp=0.0, speed=2.0, success=True),
        _make_event(attacker_id=1, defender_id=1, frame_number=1, timestamp=0.04, speed=1.5, success=False),
        _make_event(attacker_id=2, defender_id=2, frame_number=2, timestamp=0.08, speed=3.0, success=True),
    ]
    seq = _make_sequence(sequence_id=0, start=0, end=2)
    conf = detector.calculate_confidence(events, [seq])
    assert 0.0 <= conf <= 1.0


def test_calculate_confidence_all_successful(detector: PressingDetector):
    events = [
        _make_event(attacker_id=0, defender_id=0, frame_number=0, timestamp=0.0, speed=2.0, success=True),
        _make_event(attacker_id=1, defender_id=1, frame_number=1, timestamp=0.04, speed=2.0, success=True),
    ]
    seq = _make_sequence(sequence_id=0, start=0, end=1)
    conf = detector.calculate_confidence(events, [seq])
    assert conf > 0.5


# ------------------------------------------------------------------
# reset
# ------------------------------------------------------------------

def test_reset(detector: PressingDetector):
    # Should not raise
    detector.reset()


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _make_event(
    attacker_id: int = 0,
    defender_id: int = 0,
    frame_number: int = 0,
    timestamp: float = 0.0,
    speed: float = 2.0,
    success: bool = False,
):
    from datetime import datetime, timezone
    from app.analytics.pressing_types import PressureEvent
    return PressureEvent(
        attacker_id=attacker_id,
        defender_id=defender_id,
        team_id=1,
        frame_number=frame_number,
        timestamp=datetime.fromtimestamp(timestamp, tz=timezone.utc),
        distance=3.0,
        closing_speed=speed,
        pressure_angle=0.0,
        successful=success,
    )


def _make_sequence(sequence_id: int = 0, start: int = 0, end: int = 10):
    from datetime import datetime, timezone
    from app.analytics.pressing_types import PressingSequence
    return PressingSequence(
        sequence_id=sequence_id,
        team_id=1,
        start_frame=start,
        end_frame=end,
        start_time=datetime.fromtimestamp(start / 25.0, tz=timezone.utc),
        end_time=datetime.fromtimestamp(end / 25.0, tz=timezone.utc),
        pressure_events=[],
        duration_seconds=(end - start) / 25.0,
    )