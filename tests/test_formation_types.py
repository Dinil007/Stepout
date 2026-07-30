from __future__ import annotations

from datetime import datetime, timezone

import pytest

from app.analytics.formation_types import (
    FormationDetection,
    FormationMetrics,
    FormationTemplate,
    FormationWindow,
    FormationTransition,
    PlayerPosition,
    TeamShape,
)


def test_player_position_defaults():
    now = datetime.now(timezone.utc)
    pos = PlayerPosition(
        player_id=1, team_id=1, team_name="A", jersey_number=10, x=0.5, y=0.5, frame_number=1, timestamp=now
    )
    assert pos.confidence == 1.0
    assert pos.is_goalkeeper is False
    assert pos.is_visible is True


def test_player_position_valid():
    now = datetime.now(timezone.utc)
    pos = PlayerPosition(
        player_id=1, team_id=1, team_name="A", jersey_number=10, x=0.5, y=0.5, frame_number=1, timestamp=now
    )
    assert pos.is_valid() is True
    assert pos.within_pitch_bounds() is True


def test_player_position_invalid_coordinates():
    now = datetime.now(timezone.utc)
    pos = PlayerPosition(
        player_id=1, team_id=1, team_name="A", jersey_number=10, x=-0.1, y=1.2, frame_number=1, timestamp=now
    )
    assert pos.is_valid() is False
    assert pos.within_pitch_bounds() is False


def test_player_position_invalid_confidence():
    now = datetime.now(timezone.utc)
    pos = PlayerPosition(
        player_id=1, team_id=1, team_name="A", jersey_number=10, x=0.5, y=0.5, frame_number=1, timestamp=now, confidence=1.5
    )
    assert pos.is_valid() is False
    pos2 = PlayerPosition(
        player_id=1, team_id=1, team_name="A", jersey_number=10, x=0.5, y=0.5, frame_number=1, timestamp=now, confidence=-0.1
    )
    assert pos2.is_valid() is False


def test_player_position_distance_to():
    now = datetime.now(timezone.utc)
    p1 = PlayerPosition(
        player_id=1, team_id=1, team_name="A", jersey_number=10, x=0.0, y=0.0, frame_number=1, timestamp=now
    )
    p2 = PlayerPosition(
        player_id=2, team_id=1, team_name="A", jersey_number=11, x=0.0, y=1.0, frame_number=1, timestamp=now
    )
    assert abs(p1.distance_to(p2) - 1.0) < 1e-6


def test_formation_template_player_count():
    tpl = FormationTemplate(
        formation_name="4-3-3",
        defenders=4,
        midfielders=3,
        forwards=3,
        normalized_positions=[(0.0, 0.0)] * 10,
    )
    assert tpl.player_count() == 10
    assert tpl.is_valid() is True


def test_formation_template_invalid_counts():
    tpl = FormationTemplate(
        formation_name="BAD",
        defenders=-1,
        midfielders=3,
        forwards=3,
    )
    assert tpl.is_valid() is False


def test_formation_template_mismatch_positions():
    tpl = FormationTemplate(
        formation_name="MISMATCH",
        defenders=4,
        midfielders=3,
        forwards=3,
        normalized_positions=[(0.0, 0.0)] * 9,
    )
    assert tpl.is_valid() is False


def test_formation_detection_valid():
    now = datetime.now(timezone.utc)
    det = FormationDetection(
        detected_formation="4-3-3",
        confidence=0.9,
        frame_number=10,
        timestamp=now,
        matched_template="4-3-3",
        score=0.8,
    )
    assert det.is_valid() is True


def test_formation_detection_invalid():
    now = datetime.now(timezone.utc)
    det = FormationDetection(
        detected_formation="4-3-3",
        confidence=1.2,
        frame_number=10,
        timestamp=now,
        matched_template="4-3-3",
        score=0.8,
    )
    assert det.is_valid() is False


def test_formation_metrics_valid():
    metrics = FormationMetrics(
        team_width=1.0,
        team_length=1.0,
        compactness=0.5,
        centroid_x=0.5,
        centroid_y=0.5,
        convex_hull_area=0.3,
        defensive_line=0.25,
        midfield_line=0.5,
        forward_line=0.75,
        vertical_stretch=0.5,
        horizontal_stretch=0.6,
    )
    assert metrics.is_valid() is True
    assert metrics.within_pitch_bounds() is True


def test_formation_metrics_invalid():
    metrics = FormationMetrics(
        team_width=-0.1,
        team_length=1.0,
        compactness=0.5,
        centroid_x=0.5,
        centroid_y=0.5,
        convex_hull_area=0.3,
        defensive_line=1.5,
        midfield_line=0.5,
        forward_line=0.75,
        vertical_stretch=0.5,
        horizontal_stretch=0.6,
    )
    assert metrics.is_valid() is False


def test_formation_window_valid():
    now = datetime.now(timezone.utc)
    metrics = FormationMetrics(
        team_width=1.0,
        team_length=1.0,
        compactness=0.5,
        centroid_x=0.5,
        centroid_y=0.5,
        convex_hull_area=0.3,
        defensive_line=0.25,
        midfield_line=0.5,
        forward_line=0.75,
        vertical_stretch=0.5,
        horizontal_stretch=0.6,
    )
    window = FormationWindow(
        start_frame=0,
        end_frame=100,
        duration_seconds=4.0,
        formation="4-3-3",
        confidence=0.9,
        metrics=metrics,
    )
    assert window.is_valid() is True


def test_formation_window_invalid():
    now = datetime.now(timezone.utc)
    metrics = FormationMetrics(
        team_width=1.0,
        team_length=1.0,
        compactness=0.5,
        centroid_x=0.5,
        centroid_y=0.5,
        convex_hull_area=0.3,
        defensive_line=0.25,
        midfield_line=0.5,
        forward_line=0.75,
        vertical_stretch=0.5,
        horizontal_stretch=0.6,
    )
    window = FormationWindow(
        start_frame=100,
        end_frame=0,
        duration_seconds=4.0,
        formation="4-3-3",
        confidence=0.9,
        metrics=metrics,
    )
    assert window.is_valid() is False


def test_formation_transition_valid():
    now = datetime.now(timezone.utc)
    tr = FormationTransition(
        previous_formation="4-4-2",
        new_formation="4-3-3",
        timestamp=now,
        frame_number=50,
        confidence=0.95,
    )
    assert tr.is_valid() is True


def test_formation_transition_invalid():
    now = datetime.now(timezone.utc)
    tr = FormationTransition(
        previous_formation="",
        new_formation="4-3-3",
        timestamp=now,
        frame_number=50,
        confidence=0.95,
    )
    assert tr.is_valid() is False


def test_team_shape_valid():
    shape = TeamShape(
        team_name="A",
        average_positions=[(0.5, 0.5)] * 11,
        width=1.0,
        length=1.0,
        compactness=0.5,
        centroid=(0.5, 0.5),
        formation="4-3-3",
    )
    assert shape.is_valid() is True
    assert shape.player_count() == 11


def test_team_shape_invalid():
    shape = TeamShape(
        team_name="",
        average_positions=[(0.5, 0.5)] * 11,
        width=1.0,
        length=1.0,
        compactness=1.2,
        centroid=(0.5, 0.5),
        formation="4-3-3",
    )
    assert shape.is_valid() is False