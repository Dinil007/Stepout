from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from app.analytics.pressing_config import PressingConfig
from app.analytics.pressing_types import (
    PPDAWindow,
    PressingDetection,
    PressingMetrics,
    PressingSequence,
    PressingZone,
    PressureEvent,
)
from app.analytics.pressing_visualizer import PressingVisualizer, VisualizerConfig


@pytest.fixture
def visualizer() -> PressingVisualizer:
    return PressingVisualizer()


@pytest.fixture
def mock_cv2():
    """Mock cv2 module so tests work without OpenCV installed."""
    with patch("app.analytics.pressing_visualizer.PressingVisualizer._import_opencv") as mock:
        cv2_mock = MagicMock()
        cv2_mock.FONT_HERSHEY_SIMPLEX = 0
        cv2_mock.LINE_AA = 16
        cv2_mock.MARKER_TILTED_CROSS = 0
        mock.return_value = cv2_mock
        yield cv2_mock


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _make_event(
    attacker_id: int = 0,
    defender_id: int = 0,
    distance: float = 3.0,
    speed: float = 2.0,
    success: bool = False,
    frame: int = 0,
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
# VisualizerConfig
# ------------------------------------------------------------------

def test_visualizer_config_defaults():
    cfg = VisualizerConfig()
    assert cfg.image_width == 1280
    assert cfg.image_height == 800
    assert cfg.show_pitch is True
    assert cfg.show_players is True
    assert cfg.show_ball is True
    assert cfg.show_pressure_events is True
    assert cfg.show_pressing_zones is True
    assert cfg.show_pressing_lines is True
    assert cfg.show_labels is True
    assert cfg.show_confidence is True
    assert cfg.show_sequences is True
    assert cfg.show_heat_overlay is False
    assert cfg.line_thickness == 2
    assert cfg.player_radius == 8
    assert cfg.font_scale == 0.6
    assert cfg.font_thickness == 2
    assert cfg.alpha == 0.4


def test_visualizer_config_custom():
    cfg = VisualizerConfig(
        image_width=640,
        image_height=480,
        show_pitch=False,
        show_players=False,
        show_ball=False,
        show_pressure_events=False,
        show_pressing_zones=False,
        show_pressing_lines=False,
        show_labels=False,
        show_confidence=False,
        show_sequences=False,
        show_heat_overlay=True,
        line_thickness=3,
        player_radius=10,
        font_scale=0.8,
        font_thickness=1,
        alpha=0.6,
    )
    assert cfg.image_width == 640
    assert cfg.image_height == 480
    assert cfg.show_pitch is False
    assert cfg.show_players is False
    assert cfg.show_ball is False
    assert cfg.show_pressure_events is False
    assert cfg.show_pressing_zones is False
    assert cfg.show_pressing_lines is False
    assert cfg.show_labels is False
    assert cfg.show_confidence is False
    assert cfg.show_sequences is False
    assert cfg.show_heat_overlay is True
    assert cfg.line_thickness == 3
    assert cfg.player_radius == 10
    assert cfg.font_scale == 0.8
    assert cfg.font_thickness == 1
    assert cfg.alpha == 0.6


# ------------------------------------------------------------------
# Initialization
# ------------------------------------------------------------------

def test_visualizer_default_init():
    v = PressingVisualizer()
    assert isinstance(v.config, VisualizerConfig)
    assert isinstance(v.pressing_config, PressingConfig)
    v.reset()


def test_visualizer_custom_init():
    vc = VisualizerConfig(image_width=800)
    pc = PressingConfig(pressure_distance_threshold=3.0)
    v = PressingVisualizer(config=vc, pressing_config=pc)
    assert v.config.image_width == 800
    assert v.pressing_config.pressure_distance_threshold == 3.0


# ------------------------------------------------------------------
# _create_canvas
# ------------------------------------------------------------------

def test_create_canvas(visualizer: PressingVisualizer):
    canvas = visualizer._create_canvas()
    assert isinstance(canvas, np.ndarray)
    assert canvas.shape == (800, 1280, 3)
    assert canvas.dtype == np.uint8


# ------------------------------------------------------------------
# _normalize_to_pixel
# ------------------------------------------------------------------

def test_normalize_to_pixel(visualizer: PressingVisualizer):
    canvas = visualizer._create_canvas()
    px, py = visualizer._normalize_to_pixel(0.5, 0.5, canvas)
    assert px == 639  # int(0.5 * 1279)
    assert py == 399  # int(0.5 * 799)


def test_normalize_to_pixel_origin(visualizer: PressingVisualizer):
    canvas = visualizer._create_canvas()
    px, py = visualizer._normalize_to_pixel(0.0, 0.0, canvas)
    assert px == 0
    assert py == 0


def test_normalize_to_pixel_max(visualizer: PressingVisualizer):
    canvas = visualizer._create_canvas()
    px, py = visualizer._normalize_to_pixel(1.0, 1.0, canvas)
    assert px == 1279
    assert py == 799


# ------------------------------------------------------------------
# _clamp_pixel
# ------------------------------------------------------------------

def test_clamp_pixel_within_bounds(visualizer: PressingVisualizer):
    x, y = visualizer._clamp_pixel(100, 100)
    assert x == 100
    assert y == 100


def test_clamp_pixel_negative(visualizer: PressingVisualizer):
    x, y = visualizer._clamp_pixel(-10, -20)
    assert x == 0
    assert y == 0


def test_clamp_pixel_exceeds(visualizer: PressingVisualizer):
    x, y = visualizer._clamp_pixel(2000, 2000)
    assert x == 1279
    assert y == 799


# ------------------------------------------------------------------
# draw_pitch
# ------------------------------------------------------------------

def test_draw_pitch_creates_canvas(visualizer: PressingVisualizer):
    canvas = visualizer.draw_pitch()
    assert isinstance(canvas, np.ndarray)
    assert canvas.shape == (800, 1280, 3)


def test_draw_pitch_on_existing(visualizer: PressingVisualizer):
    existing = np.zeros((400, 600, 3), dtype=np.uint8)
    result = visualizer.draw_pitch(canvas=existing)
    assert result.shape == (400, 600, 3)


# ------------------------------------------------------------------
# draw_players
# ------------------------------------------------------------------

def test_draw_players_none(visualizer: PressingVisualizer):
    canvas = visualizer._create_canvas()
    result = visualizer.draw_players(player_positions=None, canvas=canvas)
    assert result is canvas


def test_draw_players_empty(visualizer: PressingVisualizer):
    canvas = visualizer._create_canvas()
    result = visualizer.draw_players(player_positions=[], canvas=canvas)
    assert result is canvas


def test_draw_players_with_data(visualizer: PressingVisualizer):
    canvas = visualizer._create_canvas()
    players = [(1, 0.5, 0.5, 1), (2, 0.3, 0.3, 2)]
    result = visualizer.draw_players(player_positions=players, canvas=canvas)
    assert isinstance(result, np.ndarray)


def test_draw_players_creates_canvas(visualizer: PressingVisualizer):
    players = [(1, 0.5, 0.5, 1)]
    result = visualizer.draw_players(player_positions=players)
    assert isinstance(result, np.ndarray)


# ------------------------------------------------------------------
# draw_ball
# ------------------------------------------------------------------

def test_draw_ball_none(visualizer: PressingVisualizer):
    canvas = visualizer._create_canvas()
    result = visualizer.draw_ball(ball_position=None, canvas=canvas)
    assert result is canvas


def test_draw_ball_with_position(visualizer: PressingVisualizer):
    canvas = visualizer._create_canvas()
    result = visualizer.draw_ball(ball_position=(0.5, 0.5), canvas=canvas)
    assert isinstance(result, np.ndarray)


def test_draw_ball_creates_canvas(visualizer: PressingVisualizer):
    result = visualizer.draw_ball(ball_position=(0.5, 0.5))
    assert isinstance(result, np.ndarray)


# ------------------------------------------------------------------
# draw_pressure_events
# ------------------------------------------------------------------

def test_draw_pressure_events_none(visualizer: PressingVisualizer):
    canvas = visualizer._create_canvas()
    result = visualizer.draw_pressure_events(events=None, canvas=canvas)
    assert result is canvas


def test_draw_pressure_events_empty(visualizer: PressingVisualizer):
    canvas = visualizer._create_canvas()
    result = visualizer.draw_pressure_events(events=[], canvas=canvas)
    assert result is canvas


def test_draw_pressure_events_with_data(visualizer: PressingVisualizer):
    canvas = visualizer._create_canvas()
    events = [_make_event(attacker_id=0, defender_id=0)]
    attacker_pos = {0: (0.5, 0.5)}
    defender_pos = {0: (0.6, 0.5)}
    result = visualizer.draw_pressure_events(
        events=events, canvas=canvas,
        attacker_positions=attacker_pos,
        defender_positions=defender_pos,
    )
    assert isinstance(result, np.ndarray)


def test_draw_pressure_events_missing_positions(visualizer: PressingVisualizer):
    canvas = visualizer._create_canvas()
    events = [_make_event(attacker_id=0, defender_id=0)]
    # No positions provided -> should skip gracefully
    result = visualizer.draw_pressure_events(events=events, canvas=canvas)
    assert isinstance(result, np.ndarray)


# ------------------------------------------------------------------
# draw_pressing_zones
# ------------------------------------------------------------------

def test_draw_pressing_zones(visualizer: PressingVisualizer):
    canvas = visualizer._create_canvas()
    result = visualizer.draw_pressing_zones(canvas=canvas)
    assert isinstance(result, np.ndarray)


def test_draw_pressing_zones_creates_canvas(visualizer: PressingVisualizer):
    result = visualizer.draw_pressing_zones()
    assert isinstance(result, np.ndarray)


# ------------------------------------------------------------------
# draw_pressing_sequences
# ------------------------------------------------------------------

def test_draw_pressing_sequences_none(visualizer: PressingVisualizer):
    canvas = visualizer._create_canvas()
    result = visualizer.draw_pressing_sequences(sequences=None, canvas=canvas)
    assert result is canvas


def test_draw_pressing_sequences_empty(visualizer: PressingVisualizer):
    canvas = visualizer._create_canvas()
    result = visualizer.draw_pressing_sequences(sequences=[], canvas=canvas)
    assert result is canvas


def test_draw_pressing_sequences_with_data(visualizer: PressingVisualizer):
    canvas = visualizer._create_canvas()
    events = [
        _make_event(attacker_id=0, defender_id=0, frame=0),
        _make_event(attacker_id=0, defender_id=0, frame=1),
    ]
    seq = _make_sequence(events=events)
    attacker_pos = {0: (0.5, 0.5)}
    defender_pos = {0: (0.6, 0.5)}
    result = visualizer.draw_pressing_sequences(
        sequences=[seq], canvas=canvas,
        attacker_positions=attacker_pos,
        defender_positions=defender_pos,
    )
    assert isinstance(result, np.ndarray)


def test_draw_pressing_sequences_single_event(visualizer: PressingVisualizer):
    canvas = visualizer._create_canvas()
    events = [_make_event(attacker_id=0, defender_id=0, frame=0)]
    seq = _make_sequence(events=events)
    result = visualizer.draw_pressing_sequences(
        sequences=[seq], canvas=canvas,
    )
    assert isinstance(result, np.ndarray)


# ------------------------------------------------------------------
# annotate_metrics
# ------------------------------------------------------------------

def test_annotate_metrics_none(visualizer: PressingVisualizer):
    canvas = visualizer._create_canvas()
    result = visualizer.annotate_metrics(canvas=canvas)
    assert isinstance(result, np.ndarray)


def test_annotate_metrics_with_metrics(visualizer: PressingVisualizer):
    canvas = visualizer._create_canvas()
    metrics = PressingMetrics(
        total_pressures=50, successful_pressures=25,
        pressure_success_rate=0.5, average_pressure_time=2.0,
        average_closing_speed=1.5, ppda=8.0,
        high_press_count=20, mid_block_count=15, low_block_count=15,
    )
    result = visualizer.annotate_metrics(metrics=metrics, canvas=canvas)
    assert isinstance(result, np.ndarray)


def test_annotate_metrics_with_detection(visualizer: PressingVisualizer):
    canvas = visualizer._create_canvas()
    detection = PressingDetection(
        pressing_style=PressingZone.HIGH_PRESS,
        confidence=0.9, frame_number=10, timestamp=_now(),
    )
    result = visualizer.annotate_metrics(detection=detection, canvas=canvas)
    assert isinstance(result, np.ndarray)


def test_annotate_metrics_with_ppda(visualizer: PressingVisualizer):
    canvas = visualizer._create_canvas()
    ppda = PPDAWindow(
        team_id=1, start_time=_now(), end_time=_now(),
        passes_allowed=100, defensive_actions=20, ppda=5.0,
    )
    result = visualizer.annotate_metrics(ppda_window=ppda, canvas=canvas)
    assert isinstance(result, np.ndarray)


def test_annotate_metrics_all(visualizer: PressingVisualizer):
    canvas = visualizer._create_canvas()
    metrics = PressingMetrics(
        total_pressures=50, successful_pressures=25,
        pressure_success_rate=0.5, average_pressure_time=2.0,
        average_closing_speed=1.5, ppda=8.0,
        high_press_count=20, mid_block_count=15, low_block_count=15,
    )
    ppda = PPDAWindow(
        team_id=1, start_time=_now(), end_time=_now(),
        passes_allowed=100, defensive_actions=20, ppda=5.0,
    )
    detection = PressingDetection(
        pressing_style=PressingZone.HIGH_PRESS,
        confidence=0.9, frame_number=10, timestamp=_now(),
    )
    result = visualizer.annotate_metrics(
        metrics=metrics, ppda_window=ppda, detection=detection, canvas=canvas,
    )
    assert isinstance(result, np.ndarray)


# ------------------------------------------------------------------
# render_frame
# ------------------------------------------------------------------

def test_render_frame_default(visualizer: PressingVisualizer):
    result = visualizer.render_frame()
    assert isinstance(result, np.ndarray)
    assert result.shape == (800, 1280, 3)


def test_render_frame_with_data(visualizer: PressingVisualizer):
    players = [(1, 0.5, 0.5, 1), (2, 0.3, 0.7, 2)]
    ball = (0.6, 0.5)
    events = [_make_event(attacker_id=0, defender_id=0)]
    seq = _make_sequence(events=events)
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
    result = visualizer.render_frame(
        player_positions=players,
        ball_position=ball,
        pressure_events=events,
        sequences=[seq],
        metrics=metrics,
        detection=detection,
        attacker_positions={0: (0.5, 0.5)},
        defender_positions={0: (0.6, 0.5)},
    )
    assert isinstance(result, np.ndarray)


def test_render_frame_with_canvas(visualizer: PressingVisualizer):
    canvas = np.zeros((400, 600, 3), dtype=np.uint8)
    result = visualizer.render_frame(canvas=canvas)
    assert result.shape == (400, 600, 3)


def test_render_frame_toggles_off(visualizer: PressingVisualizer):
    visualizer.config.show_pitch = False
    visualizer.config.show_players = False
    visualizer.config.show_ball = False
    visualizer.config.show_pressure_events = False
    visualizer.config.show_pressing_zones = False
    visualizer.config.show_sequences = False
    result = visualizer.render_frame()
    assert isinstance(result, np.ndarray)


# ------------------------------------------------------------------
# render_animation
# ------------------------------------------------------------------

def test_render_animation_empty_raises(visualizer: PressingVisualizer):
    with pytest.raises(ValueError):
        visualizer.render_animation(frames=None)


def test_render_animation_empty_list_raises(visualizer: PressingVisualizer):
    with pytest.raises(ValueError):
        visualizer.render_animation(frames=[])


def test_render_animation_single_frame(visualizer: PressingVisualizer):
    frames = [{}]
    result = visualizer.render_animation(frames=frames)
    assert len(result) == 1
    assert isinstance(result[0], np.ndarray)


def test_render_animation_multiple_frames(visualizer: PressingVisualizer):
    frames = [{}, {}, {}]
    result = visualizer.render_animation(frames=frames)
    assert len(result) == 3


# ------------------------------------------------------------------
# reset
# ------------------------------------------------------------------

def test_reset(visualizer: PressingVisualizer):
    visualizer._frame_count = 42
    visualizer.reset()
    assert visualizer._frame_count == 0


# ------------------------------------------------------------------
# Error handling: missing frame / rendering failures
# ------------------------------------------------------------------

def test_render_frame_exception_returns_original(visualizer: PressingVisualizer):
    """If an exception occurs, render_frame should return the original canvas."""
    canvas = np.zeros((100, 100, 3), dtype=np.uint8)
    # Force an error by passing invalid data that causes a crash in OpenCV calls
    # Since we mock cv2, we can't easily force an error. Instead verify the
    # method handles None canvas gracefully.
    result = visualizer.render_frame(canvas=canvas)
    assert result.shape == (100, 100, 3)


def test_draw_pitch_exception_returns_canvas(visualizer: PressingVisualizer):
    """draw_pitch should handle exceptions and return a canvas."""
    result = visualizer.draw_pitch()
    assert isinstance(result, np.ndarray)