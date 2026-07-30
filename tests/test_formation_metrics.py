from __future__ import annotations

from datetime import datetime, timezone

import pytest

from app.analytics.formation_config import FormationConfig
from app.analytics.formation_metrics import FormationMetricsEngine
from app.analytics.formation_types import PlayerPosition


def _players(positions, team_id: int = 1, start_id: int = 1) -> list[PlayerPosition]:
    players = []
    now = datetime.now(timezone.utc)
    for idx, (x, y) in enumerate(positions):
        players.append(
            PlayerPosition(
                player_id=start_id + idx,
                team_id=team_id,
                team_name="Home",
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


# 1. Team Width


def test_width_computed_correctly():
    engine = FormationMetricsEngine()
    players = _players([(0.0, 0.5), (0.5, 0.5), (1.0, 0.5)])
    metrics = engine.compute_metrics(players)
    assert abs(metrics.team_width - 1.0) < 1e-6


def test_width_zero_when_same_x():
    engine = FormationMetricsEngine()
    players = _players([(0.5, 0.2), (0.5, 0.5), (0.5, 0.8)])
    metrics = engine.compute_metrics(players)
    assert abs(metrics.team_width - 0.0) < 1e-6


# 2. Team Length


def test_length_computed_correctly():
    engine = FormationMetricsEngine()
    players = _players([(0.5, 0.0), (0.5, 0.5), (0.5, 1.0)])
    metrics = engine.compute_metrics(players)
    assert abs(metrics.team_length - 1.0) < 1e-6


def test_length_zero_when_same_y():
    engine = FormationMetricsEngine()
    players = _players([(0.2, 0.5), (0.5, 0.5), (0.8, 0.5)])
    metrics = engine.compute_metrics(players)
    assert abs(metrics.team_length - 0.0) < 1e-6


# 3. Centroid


def test_centroid_symmetric_formation():
    engine = FormationMetricsEngine()
    players = _players([(0.25, 0.25), (0.75, 0.25), (0.25, 0.75), (0.75, 0.75)])
    metrics = engine.compute_metrics(players)
    assert abs(metrics.centroid_x - 0.5) < 1e-6
    assert abs(metrics.centroid_y - 0.5) < 1e-6


def test_centroid_asymmetric_formation():
    engine = FormationMetricsEngine()
    players = _players([(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)])
    metrics = engine.compute_metrics(players)
    assert abs(metrics.centroid_x - 2.0 / 3.0) < 1e-6
    assert abs(metrics.centroid_y - 1.0 / 3.0) < 1e-6


# 4. Compactness


def test_compactness_non_negative():
    engine = FormationMetricsEngine()
    players = _players([(0.2, 0.2), (0.4, 0.4), (0.6, 0.6)])
    metrics = engine.compute_metrics(players)
    assert metrics.compactness >= 0.0
    assert metrics.compactness <= 1.0


def test_compact_team_has_higher_compactness_than_dispersed():
    engine = FormationMetricsEngine()
    compact = _players([(0.4, 0.4), (0.5, 0.5), (0.6, 0.4)])
    dispersed = _players([(0.0, 0.0), (0.5, 1.0), (1.0, 0.0)])
    compact_metrics = engine.compute_metrics(compact)
    dispersed_metrics = engine.compute_metrics(dispersed)
    assert compact_metrics.compactness > dispersed_metrics.compactness


# 5. Convex Hull Area


def test_convex_hull_positive_for_standard_formation():
    engine = FormationMetricsEngine()
    players = _players([(0.2, 0.2), (0.8, 0.2), (0.5, 0.8)])
    metrics = engine.compute_metrics(players)
    assert metrics.convex_hull_area > 0.0


def test_convex_hull_returns_zero_for_collinear():
    engine = FormationMetricsEngine()
    players = _players([(0.0, 0.0), (0.5, 0.5), (1.0, 1.0)])
    metrics = engine.compute_metrics(players)
    assert abs(metrics.convex_hull_area - 0.0) < 1e-6


def test_convex_hull_returns_zero_for_fewer_than_three():
    engine = FormationMetricsEngine()
    players = _players([(0.0, 0.0), (0.5, 0.5)])
    metrics = engine.compute_metrics(players)
    assert abs(metrics.convex_hull_area - 0.0) < 1e-6


# 6. Line Heights


def test_line_heights_known_layout():
    engine = FormationMetricsEngine()
    ys = [0.2, 0.25, 0.3, 0.5, 0.55, 0.6, 0.8, 0.85, 0.9]
    players = _players([(0.5, y) for y in ys])
    metrics = engine.compute_metrics(players)
    assert metrics.defensive_line < metrics.midfield_line < metrics.forward_line


# 7. Vertical Stretch


def test_vertical_stretch_equals_length():
    engine = FormationMetricsEngine()
    players = _players([(0.5, 0.1), (0.5, 0.3), (0.5, 0.7), (0.5, 0.9)])
    metrics = engine.compute_metrics(players)
    assert abs(metrics.vertical_stretch - metrics.team_length) < 1e-6


# 8. Horizontal Stretch


def test_horizontal_stretch_equals_width():
    engine = FormationMetricsEngine()
    players = _players([(0.1, 0.5), (0.3, 0.5), (0.7, 0.5), (0.9, 0.5)])
    metrics = engine.compute_metrics(players)
    assert abs(metrics.horizontal_stretch - metrics.team_width) < 1e-6


# 9. Average Inter-player Distance


def test_interplayer_distance_positive_for_multiple_players():
    engine = FormationMetricsEngine()
    players = _players([(0.0, 0.0), (0.0, 1.0), (1.0, 0.0)])
    metrics = engine.compute_metrics(players)
    assert metrics is not None
    # engine calculates internally but does not expose it; verify by direct call
    xs = [p.x for p in players]
    ys = [p.y for p in players]
    avg = engine.calculate_interplayer_distance(
        __import__("numpy").array(xs, dtype=float),
        __import__("numpy").array(ys, dtype=float),
    )
    assert avg >= 0.0


# 10. Team Density


def test_density_no_division_by_zero():
    engine = FormationMetricsEngine()
    players = _players([(0.5, 0.5), (0.5, 0.5)])
    metrics = engine.compute_metrics(players)
    assert metrics is not None
    density = engine.calculate_density(len(players), metrics.team_width, metrics.team_length)
    assert density >= 0.0


# 11. Validation


def test_empty_player_list_raises():
    engine = FormationMetricsEngine()
    with pytest.raises(ValueError):
        engine.compute_metrics([])


def test_invalid_coordinates_raise():
    engine = FormationMetricsEngine()
    players = _players([(1.5, 0.5), (0.2, 0.5)])
    with pytest.raises(ValueError):
        engine.compute_metrics(players)


# 12. Numerical Stability


def test_very_close_coordinates_stable():
    engine = FormationMetricsEngine()
    base = [0.5 + 1e-12, 0.5 + 2e-12, 0.5 + 3e-12]
    players = _players([(x, 0.5) for x in base])
    metrics = engine.compute_metrics(players)
    assert metrics is not None
    assert metrics.centroid_x >= 0.0


# 13. Regression - no mutation


def test_metrics_engine_does_not_mutate_input():
    engine = FormationMetricsEngine()
    base = [(0.2, 0.2), (0.4, 0.4), (0.6, 0.6)]
    players = _players(base)
    original = [(p.x, p.y) for p in players]
    engine.compute_metrics(players)
    after = [(p.x, p.y) for p in players]
    assert original == after