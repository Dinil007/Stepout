from __future__ import annotations

from datetime import datetime, timezone

import pytest

from app.analytics.formation_engine import FormationEngine
from app.analytics.formation_validation import FormationValidator, ValidationReport
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


# Player Validation


def test_valid_players_pass():
    validator = FormationValidator()
    players = _players([(0.0, 0.0), (0.5, 0.5)])
    report = validator.validate_players(players)
    assert report.overall_valid is True


def test_invalid_player_report():
    validator = FormationValidator()
    players = _players([(1.5, 0.5), (0.2, 0.5)])
    report = validator.validate_players(players)
    assert report.overall_valid is False


def test_empty_player_list_fails():
    validator = FormationValidator()
    report = validator.validate_players([])
    assert report.overall_valid is False


# Detection Validation


def test_valid_detection_passes():
    validator = FormationValidator()
    engine = FormationEngine()
    players = _players([(0.2, 0.25), (0.4, 0.25), (0.6, 0.25), (0.8, 0.25), (0.3, 0.5), (0.5, 0.5), (0.7, 0.5), (0.25, 0.75), (0.5, 0.75), (0.75, 0.75)])
    analysis = engine.analyze(players)
    detections = engine.detector.detect_all_teams(players)
    first_detection = next(iter(detections.values()))
    report = validator.validate_detection(first_detection)
    assert report.overall_valid is True


def test_invalid_confidence_fails():
    from app.analytics.formation_types import FormationDetection

    validator = FormationValidator()
    now = datetime.now(timezone.utc)
    det = FormationDetection(
        detected_formation="4-3-3",
        confidence=1.2,
        frame_number=1,
        timestamp=now,
        matched_template="4-3-3",
        score=0.8,
    )
    report = validator.validate_detection(det)
    assert report.overall_valid is False


def test_unknown_formation_fails():
    from app.analytics.formation_types import FormationDetection

    validator = FormationValidator()
    now = datetime.now(timezone.utc)
    det = FormationDetection(
        detected_formation="unknown-formation",
        confidence=0.9,
        frame_number=1,
        timestamp=now,
        matched_template="unknown-formation",
        score=0.8,
    )
    report = validator.validate_detection(det)
    assert report.overall_valid is False


# Metrics Validation


def test_metrics_negative_width_fails():
    from app.analytics.formation_types import FormationMetrics

    validator = FormationValidator()
    metrics = FormationMetrics(
        team_width=-0.1,
        team_length=1.0,
        compactness=0.5,
        centroid_x=0.5,
        centroid_y=0.5,
        convex_hull_area=0.3,
        defensive_line=0.2,
        midfield_line=0.5,
        forward_line=0.8,
        vertical_stretch=1.0,
        horizontal_stretch=1.0,
    )
    report = validator.validate_metrics(metrics)
    assert report.overall_valid is False


def test_metrics_centroid_out_of_bounds_fails():
    from app.analytics.formation_types import FormationMetrics

    validator = FormationValidator()
    metrics = FormationMetrics(
        team_width=1.0,
        team_length=1.0,
        compactness=0.5,
        centroid_x=1.5,
        centroid_y=-0.1,
        convex_hull_area=0.3,
        defensive_line=0.2,
        midfield_line=0.5,
        forward_line=0.8,
        vertical_stretch=1.0,
        horizontal_stretch=1.0,
    )
    report = validator.validate_metrics(metrics)
    assert report.overall_valid is False


def test_valid_metrics_pass():
    from app.analytics.formation_types import FormationMetrics

    validator = FormationValidator()
    metrics = FormationMetrics(
        team_width=1.0,
        team_length=1.0,
        compactness=0.5,
        centroid_x=0.5,
        centroid_y=0.5,
        convex_hull_area=0.3,
        defensive_line=0.2,
        midfield_line=0.5,
        forward_line=0.8,
        vertical_stretch=1.0,
        horizontal_stretch=1.0,
    )
    report = validator.validate_metrics(metrics)
    assert report.overall_valid is True


# Analysis Validation


def test_valid_analysis_passes():
    validator = FormationValidator()
    engine = FormationEngine()
    players = _players([(0.2, 0.25), (0.4, 0.25), (0.6, 0.25), (0.8, 0.25), (0.3, 0.5), (0.5, 0.5), (0.7, 0.5), (0.25, 0.75), (0.5, 0.75), (0.75, 0.75)])
    result = engine.analyze(players)
    report = validator.validate_analysis(result)
    assert report.overall_valid is True


def test_report_counts():
    report = ValidationReport()
    assert report.checked_items == 0
    assert report.passed_items == 0
    assert report.overall_valid is True
    report.add_pass()
    assert report.checked_items == 1
    assert report.passed_items == 1
    report.add_warning("warn")
    assert report.checked_items == 2
    report.add_error("err")
    assert report.overall_valid is False


def test_validation_logging_does_not_raise():
    validator = FormationValidator()
    players = _players([(0.5, 0.5), (0.6, 0.6)])
    report = validator.validate_players(players)
    assert isinstance(report, ValidationReport)