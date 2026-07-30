from __future__ import annotations

import time
from datetime import datetime, timezone

import pytest

from app.analytics.formation_config import FormationConfig
from app.analytics.formation_detector import FormationDetector
from app.analytics.formation_templates import (
    FormationTemplateRegistry,
    default_registry,
)
from app.analytics.formation_types import PlayerPosition


def _positions(formation_name: str) -> list[tuple[float, float]]:
    template = default_registry.get_template(formation_name)
    return list(template.normalized_positions)


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


# 1. Happy Path


@pytest.mark.parametrize(
    "formation_name",
    [
        "4-3-3",
        "4-2-3-1",
        "4-4-2",
        "3-5-2",
    ],
)
def test_happy_path_detects_expected_formation(formation_name: str):
    detector = FormationDetector()
    positions = _positions(formation_name)
    players = _players(positions)
    result = detector.detect(players)
    assert result.detected_formation == formation_name, (
        f"Expected {formation_name}, got {result.detected_formation}"
    )
    assert result.confidence > 0.99


# 2. Confidence


def test_confidence_perfect_match_near_one():
    detector = FormationDetector()
    positions = _positions("4-3-3")
    players = _players(positions)
    result = detector.detect(players)
    assert abs(result.confidence - 1.0) < 1e-6


def test_confidence_decreases_with_noise():
    detector = FormationDetector()
    base = _positions("4-3-3")
    perfect = _players(base)
    noisy = _players([(min(x + 0.02, 1.0), min(y + 0.02, 1.0)) for x, y in base])
    base_conf = detector.detect(perfect).confidence
    noisy_conf = detector.detect(noisy).confidence
    assert noisy_conf < base_conf


def test_confidence_large_error_decreases_significantly():
    detector = FormationDetector()
    base = _positions("4-3-3")
    noisy = _players([(min(x + 0.15, 1.0), min(y + 0.15, 1.0)) for x, y in base])
    conf = detector.detect(noisy).confidence
    base_conf = detector.detect(_players(base)).confidence
    assert conf < base_conf


def test_confidence_bounded_in_range():
    detector = FormationDetector()
    base = _positions("4-4-2")
    players = _players(base)
    result = detector.detect(players)
    assert 0.0 <= result.confidence <= 1.0


# 3. Goalkeeper Handling


def test_goalkeeper_ignored_when_configured():
    config = FormationConfig(ignore_goalkeeper=True)
    detector = FormationDetector(config=config)
    positions = _positions("4-3-3")
    players = _players(positions)
    players.append(
        PlayerPosition(
            player_id=99,
            team_id=1,
            team_name="Home",
            jersey_number=1,
            x=0.5,
            y=0.1,
            frame_number=1,
            timestamp=datetime.now(timezone.utc),
            is_goalkeeper=True,
        )
    )
    result = detector.detect(players)
    assert result.detected_formation == "4-3-3"


def test_goalkeeper_included_when_configured():
    config = FormationConfig(ignore_goalkeeper=False, minimum_tracked_players=11)
    detector = FormationDetector(config=config)
    positions = _positions("5-4-1")
    players = _players(positions)
    players.append(
        PlayerPosition(
            player_id=99,
            team_id=1,
            team_name="Home",
            jersey_number=1,
            x=0.5,
            y=0.1,
            frame_number=1,
            timestamp=datetime.now(timezone.utc),
            is_goalkeeper=True,
        )
    )
    result = detector.detect(players)
    assert result.detected_formation == "5-4-1"


def test_missing_goalkeeper_handled_gracefully():
    config = FormationConfig(ignore_goalkeeper=True)
    detector = FormationDetector(config=config)
    positions = _positions("4-4-2")
    players = _players(positions)
    result = detector.detect(players)
    assert result.detected_formation == "4-4-2"


# 4. Validation


def test_empty_player_list_raises():
    detector = FormationDetector()
    with pytest.raises(ValueError):
        detector.detect([])


def test_too_few_players_raises():
    detector = FormationDetector()
    positions = _positions("4-3-3")[:3]
    players = _players(positions)
    with pytest.raises(ValueError):
        detector.detect_team(players)


def test_duplicate_player_ids_raises():
    detector = FormationDetector()
    positions = _positions("4-4-2")
    players = _players(positions)
    players[0] = PlayerPosition(
        player_id=players[1].player_id,
        team_id=players[0].team_id,
        team_name=players[0].team_name,
        jersey_number=players[0].jersey_number,
        x=players[0].x,
        y=players[0].y,
        frame_number=players[0].frame_number,
        timestamp=players[0].timestamp,
    )
    with pytest.raises(ValueError):
        detector.validate_players(players)


def test_invalid_coordinates_raise():
    detector = FormationDetector()
    players = _players(_positions("4-4-2"))
    players[0] = PlayerPosition(
        player_id=players[0].player_id,
        team_id=players[0].team_id,
        team_name=players[0].team_name,
        jersey_number=players[0].jersey_number,
        x=1.5,
        y=players[0].y,
        frame_number=players[0].frame_number,
        timestamp=players[0].timestamp,
    )
    with pytest.raises(ValueError):
        detector.validate_players(players)


# 5. Multi-Team


def test_multi_team_detection_groups_correctly():
    detector = FormationDetector()
    home = _players(_positions("4-3-3"), team_id=1)
    away = _players(_positions("4-2-3-1"), team_id=2, start_id=20)
    mixed = home + away
    detections = detector.detect_all_teams(mixed)
    assert 1 in detections
    assert 2 in detections
    assert detections[1].detected_formation == "4-3-3"
    assert detections[2].detected_formation == "4-2-3-1"


def test_detect_selects_highest_confidence():
    detector = FormationDetector()
    home = _players(_positions("4-3-3"), team_id=1)
    noisy = _players(
        [(min(x + 0.05, 1.0), min(y + 0.05, 1.0)) for x, y in _positions("4-4-2")],
        team_id=2,
        start_id=20,
    )
    mixed = home + noisy
    result = detector.detect(mixed)
    assert result.detected_formation == "4-3-3"
    assert result.confidence > 0.8


# 6. Sorting


def test_sorted_positions_produce_same_result():
    detector = FormationDetector()
    positions = _positions("4-2-3-1")
    players = _players(positions)
    reversed_players = list(reversed(players))
    result_original = detector.detect(players)
    result_reversed = detector.detect(reversed_players)
    assert result_original.detected_formation == result_reversed.detected_formation
    assert abs(result_original.confidence - result_reversed.confidence) < 1e-6


# 7. Template Matching


def test_all_registered_templates_matchable():
    detector = FormationDetector()
    for name in default_registry.list_templates():
        assert detector.registry.template_exists(name)


def test_unknown_template_raises_on_lookup():
    registry = FormationTemplateRegistry()
    with pytest.raises(KeyError):
        registry.get_template("unknown-formation")


# 8. Performance


def test_detection_completes_in_reasonable_time():
    detector = FormationDetector()
    positions = _positions("4-3-3")
    players = _players(positions)
    start = time.perf_counter()
    for _ in range(100):
        detector.detect(players)
    elapsed = time.perf_counter() - start
    assert elapsed < 1.0, f"100 detections took too long: {elapsed:.3f}s"


# 9. Edge Cases


def test_exact_minimum_player_count():
    config = FormationConfig(minimum_tracked_players=10)
    detector = FormationDetector(config=config)
    positions = _positions("4-4-2")
    players = _players(positions)
    result = detector.detect(players)
    assert result.detected_formation == "4-4-2"


def test_duplicate_positions_allowed():
    detector = FormationDetector()
    base = _positions("4-3-3")
    duplicate = base + [base[0]]
    players = _players(duplicate)
    result = detector.detect(players)
    assert result.detected_formation in default_registry.list_templates()
    assert 0.0 <= result.confidence <= 1.0


def test_extremely_compact_team():
    detector = FormationDetector()
    compact = [(0.5, 0.5)] * 10
    players = _players(compact)
    result = detector.detect(players)
    assert 0.0 <= result.confidence <= 1.0


def test_extremely_wide_team():
    detector = FormationDetector()
    wide = [(i * 0.1, 0.5) for i in range(10)]
    players = _players(wide)
    result = detector.detect(players)
    assert 0.0 <= result.confidence <= 1.0


# 10. Regression — no mutation


def test_detector_does_not_mutate_input():
    detector = FormationDetector()
    base = _positions("4-4-2")
    players = _players(base)
    original_ids = [p.player_id for p in players]
    detector.detect(players)
    after_ids = [p.player_id for p in players]
    assert original_ids == after_ids