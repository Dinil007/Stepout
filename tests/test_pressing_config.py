from __future__ import annotations

import pytest

from app.analytics.pressing_config import PressingConfig


def test_default_config_is_valid():
    cfg = PressingConfig()
    errors = cfg.validate()
    assert len(errors) == 0
    assert cfg.is_valid() is True


def test_invalid_pressure_distance_threshold():
    cfg = PressingConfig(pressure_distance_threshold=0)
    assert cfg.is_valid() is False
    assert any("pressure_distance_threshold" in e for e in cfg.validate())


def test_invalid_high_press_line_y_below():
    cfg = PressingConfig(high_press_line_y=-0.1)
    assert cfg.is_valid() is False


def test_invalid_high_press_line_y_above():
    cfg = PressingConfig(high_press_line_y=1.5)
    assert cfg.is_valid() is False


def test_invalid_mid_block_line_y():
    cfg = PressingConfig(mid_block_line_y=1.5)
    assert cfg.is_valid() is False


def test_invalid_low_block_line_y():
    cfg = PressingConfig(low_block_line_y=-0.5)
    assert cfg.is_valid() is False


def test_zone_line_order_violation():
    cfg = PressingConfig(high_press_line_y=0.6, mid_block_line_y=0.4)
    assert cfg.is_valid() is False
    assert any("high_press_line_y" in e for e in cfg.validate())


def test_mid_block_greater_than_low_block():
    cfg = PressingConfig(mid_block_line_y=0.7, low_block_line_y=0.5)
    assert cfg.is_valid() is False


def test_invalid_minimum_pressure_duration():
    cfg = PressingConfig(minimum_pressure_duration=0)
    assert cfg.is_valid() is False


def test_invalid_minimum_closing_speed():
    cfg = PressingConfig(minimum_closing_speed=-1.0)
    assert cfg.is_valid() is False


def test_invalid_ppda_window():
    cfg = PressingConfig(ppda_window_seconds=0)
    assert cfg.is_valid() is False


def test_invalid_confidence_threshold_above():
    cfg = PressingConfig(confidence_threshold=1.2)
    assert cfg.is_valid() is False


def test_invalid_confidence_threshold_below():
    cfg = PressingConfig(confidence_threshold=-0.1)
    assert cfg.is_valid() is False


def test_invalid_smoothing_window():
    cfg = PressingConfig(smoothing_window=0)
    assert cfg.is_valid() is False


def test_to_dict_round_trip():
    cfg = PressingConfig(
        pressure_distance_threshold=3.0,
        high_press_line_y=0.3,
        mid_block_line_y=0.6,
        low_block_line_y=0.85,
        minimum_pressure_duration=2.0,
        minimum_closing_speed=1.0,
        ppda_window_seconds=10.0,
        confidence_threshold=0.8,
        smoothing_window=5,
        enable_validation=False,
        enable_logging=False,
    )
    data = cfg.to_dict()
    recovered = PressingConfig.from_dict(data)
    assert recovered.pressure_distance_threshold == 3.0
    assert recovered.high_press_line_y == 0.3
    assert recovered.mid_block_line_y == 0.6
    assert recovered.low_block_line_y == 0.85
    assert recovered.minimum_pressure_duration == 2.0
    assert recovered.minimum_closing_speed == 1.0
    assert recovered.ppda_window_seconds == 10.0
    assert recovered.confidence_threshold == 0.8
    assert recovered.smoothing_window == 5
    assert recovered.enable_validation is False
    assert recovered.enable_logging is False


def test_copy_is_independent():
    cfg = PressingConfig(pressure_distance_threshold=4.0)
    clone = cfg.copy()
    assert clone is not cfg
    assert clone.pressure_distance_threshold == 4.0
    clone.pressure_distance_threshold = 10.0
    assert cfg.pressure_distance_threshold == 4.0


def test_update_returns_new_instance():
    cfg = PressingConfig(pressure_distance_threshold=3.0)
    updated = cfg.update(pressure_distance_threshold=8.0, minimum_closing_speed=2.0)
    assert updated is not cfg
    assert updated.pressure_distance_threshold == 8.0
    assert updated.minimum_closing_speed == 2.0
    assert cfg.pressure_distance_threshold == 3.0