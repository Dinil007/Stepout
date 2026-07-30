from __future__ import annotations

import pytest

from app.analytics.formation_config import ClusteringAlgorithm, FormationConfig


def test_default_config_is_valid():
    cfg = FormationConfig()
    cfg.validate()  # should not raise


def test_invalid_confidence_raises():
    cfg = FormationConfig(minimum_confidence=1.5)
    with pytest.raises(ValueError):
        cfg.validate()


def test_invalid_pitch_length_raises():
    cfg = FormationConfig(pitch_length=0)
    with pytest.raises(ValueError):
        cfg.validate()


def test_invalid_pitch_width_raises():
    cfg = FormationConfig(pitch_width=-10)
    with pytest.raises(ValueError):
        cfg.validate()


def test_invalid_player_limits_raises():
    cfg = FormationConfig(minimum_team_players=12, maximum_team_players=11)
    with pytest.raises(ValueError):
        cfg.validate()


def test_window_size_frames_multiple_fps():
    cfg = FormationConfig(analysis_window_seconds=2.0)
    assert cfg.window_size_frames(25) == 50
    assert cfg.window_size_frames(30) == 60
    assert cfg.window_size_frames(60) == 120


def test_window_size_frames_invalid_fps():
    cfg = FormationConfig()
    with pytest.raises(ValueError):
        cfg.window_size_frames(0)


def test_copy_is_independent():
    cfg = FormationConfig(minimum_confidence=0.8, pitch_length=100.0)
    clone = cfg.copy()
    assert clone is not cfg
    assert clone.minimum_confidence == 0.8
    clone.minimum_confidence = 0.5
    assert cfg.minimum_confidence == 0.8


def test_to_dict_from_dict_preserves_values():
    cfg = FormationConfig(
        analysis_window_seconds=5.0,
        frame_stride=2,
        minimum_confidence=0.6,
        pitch_length=100.0,
        pitch_width=64.0,
        clustering_algorithm=ClusteringAlgorithm.DBSCAN,
    )
    data = cfg.to_dict()
    recovered = FormationConfig.from_dict(data)
    assert recovered.analysis_window_seconds == 5.0
    assert recovered.frame_stride == 2
    assert recovered.minimum_confidence == 0.6
    assert recovered.pitch_length == 100.0
    assert recovered.pitch_width == 64.0
    assert recovered.clustering_algorithm == ClusteringAlgorithm.DBSCAN


def test_pitch_dimensions():
    cfg = FormationConfig(pitch_length=110.0, pitch_width=70.0)
    assert cfg.pitch_dimensions() == (110.0, 70.0)