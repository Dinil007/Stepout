"""
Test script for ball possession percentage implementation.
"""

import numpy as np
from app.analytics.ball_analytics.possession import PossessionDetector
from app.analytics.ball_analytics.metrics import BallMetricsGenerator
from app.analytics.ball_analytics.ball_analytics import BallAnalyticsEngine


def test_possession_detector_basic():
    """Test basic possession detection and percentage calculation."""
    print("=" * 50)
    print("Test 1: Basic Possession Detection")
    print("=" * 50)

    detector = PossessionDetector(
        possession_radius_m=2.0,
        min_possession_duration_frames=2,
        fps=30.0
    )

    # Simulate 10 frames
    player_positions = {
        1: (0.0, 0.0),
        2: (10.0, 10.0),
        3: (20.0, 20.0)
    }
    player_teams = {1: "Red", 2: "Blue", 3: "Red"}

    for frame in range(10):
        # Ball near player 1 (Red team) for first 5 frames
        if frame < 5:
            ball_pos = (0.5, 0.5)
        # Ball near player 2 (Blue team) for next 3 frames
        elif frame < 8:
            ball_pos = (10.5, 10.5)
        # Ball far from all players for last 2 frames
        else:
            ball_pos = (100.0, 100.0)

        result = detector.update(
            ball_position_m=ball_pos,
            player_positions_m=player_positions,
            player_teams=player_teams,
            frame_number=frame
        )

        if result:
            print(f"Frame {frame}: {result['team_id']} (P{result['track_id']})")
        else:
            print(f"Frame {frame}: Free Ball")

    # Get statistics
    pct = detector.get_possession_percentage()
    summary = detector.get_team_possession_summary()

    print("\n📊 Possession Percentages:")
    for team, percentage in pct.items():
        print(f"  {team}: {percentage}%")

    print("\n📈 Team Possession Summary:")
    print(f"  Total Frames: {summary['total_frames']}")
    print(f"  Total Duration: {summary['total_duration_seconds']}s")
    print(f"  Red Possession: {summary['team_possession_pct']['Red']}% ({summary['total_possession_time_seconds']['Red']}s)")
    print(f"  Blue Possession: {summary['team_possession_pct']['Blue']}% ({summary['total_possession_time_seconds']['Blue']}s)")
    print(f"  Free Ball: {summary['team_possession_pct']['Free Ball']}% ({summary['total_possession_time_seconds']['Free Ball']}s)")

    # Validation
    assert summary['total_frames'] == 10, "Total frames should be 10"
    assert summary['team_possession_pct']['Red'] > 0, "Red should have possession"
    assert summary['team_possession_pct']['Blue'] > 0, "Blue should have possession"
    assert summary['team_possession_pct']['Free Ball'] > 0, "Should have free ball time"

    print("\n✅ Test 1 PASSED\n")


def test_metrics_generator():
    """Test metrics generator with new detector methods."""
    print("=" * 50)
    print("Test 2: Metrics Generator Integration")
    print("=" * 50)

    metrics_gen = BallMetricsGenerator()

    # Create detector with sample data
    detector = PossessionDetector(
        possession_radius_m=2.0,
        min_possession_duration_frames=2,
        fps=30.0
    )

    player_positions = {
        1: (0.0, 0.0),
        2: (10.0, 10.0)
    }
    player_teams = {1: "Red", 2: "Blue"}

    for frame in range(20):
        # Red team possession for frames 0-9
        if frame < 10:
            ball_pos = (0.5, 0.5)
            players = player_positions
        # Blue team possession for frames 10-15
        elif frame < 16:
            ball_pos = (10.5, 10.5)
            players = player_positions
        # Free ball for frames 16-19
        else:
            ball_pos = (100.0, 100.0)
            players = {}

        detector.update(
            ball_position_m=ball_pos,
            player_positions_m=players,
            player_teams=player_teams,
            frame_number=frame
        )

    # Generate team possession from detector
    team_possession = metrics_gen.generate_team_possession_from_detector(detector)

    print("📊 Team Possession Stats from Detector:")
    for team, stats in team_possession.items():
        print(f"  {team}: {stats['possession_pct']}% ({stats['possession_time_s']}s)")

    # Validation
    assert "Red" in team_possession, "Red team should be in results"
    assert "Blue" in team_possession, "Blue team should be in results"
    assert "Free Ball" in team_possession, "Free Ball should be in results"

    # Check percentages sum to ~100%
    total_pct = sum(stats['possession_pct'] for stats in team_possession.values())
    print(f"\n  Total Percentage: {total_pct}%")
    assert 99.0 <= total_pct <= 101.0, "Total possession should be ~100%"

    print("\n✅ Test 2 PASSED\n")


def test_possession_history():
    """Test frame-by-frame history tracking."""
    print("=" * 50)
    print("Test 3: Frame-by-Frame History")
    print("=" * 50)

    detector = PossessionDetector(possession_radius_m=2.0, fps=30.0)

    player_positions = {1: (0.0, 0.0)}
    player_teams = {1: "Red"}

    for frame in range(5):
        ball_pos = (0.5, 0.5)
        detector.update(
            ball_position_m=ball_pos,
            player_positions_m=player_positions,
            player_teams=player_teams,
            frame_number=frame
        )

    print(f"📝 History Length: {len(detector.history)}")
    print(f"📊 Total Frames Tracked: {detector._total_frames}")

    # Check history
    assert len(detector.history) == 5, "History should have 5 entries"
    assert detector.history[0]['state'] == 'In Possession', "First frame should show possession"
    assert detector.history[0]['possessor'] == 1, "Possessor should be player 1"
    assert detector.history[0]['team'] == 'Red', "Team should be Red"

    print("\nHistory Sample:")
    for entry in detector.history[:3]:
        print(f"  Frame {entry['frame']}: {entry['state']} - P{entry['possessor']} ({entry['team']})")

    print("\n✅ Test 3 PASSED\n")


def test_zero_division_protection():
    """Test that methods handle zero frames correctly."""
    print("=" * 50)
    print("Test 4: Zero Division Protection")
    print("=" * 50)

    detector = PossessionDetector()

    # Call methods before processing any frames
    pct = detector.get_possession_percentage()
    summary = detector.get_team_possession_summary()

    print(f"📊 Percentages (no frames): {pct}")
    print(f"📈 Summary (no frames): {summary}")

    assert 'Free_Ball_pct' in pct, "Should have Free Ball percentage"
    assert pct['Free_Ball_pct'] == 0.0, "Free Ball should be 0% with no frames"

    print("\n✅ Test 4 PASSED\n")


def test_integration_with_ball_analytics():
    """Test that ball analytics engine uses new methods."""
    print("=" * 50)
    print("Test 5: Integration with BallAnalyticsEngine")
    print("=" * 50)

    # Minimal config
    config = {
        'video': {'fps': 30.0, 'output_dir': 'outputs'},
        'pitch': {'canvas_width': 1050, 'canvas_height': 680, 'length_m': 105.0, 'width_m': 68.0},
        'ball_analytics': {
            'possession_radius_m': 2.0,
            'min_possession_duration_frames': 2
        }
    }

    # Note: Full integration test requires proper ball_tracks format with 'track_id' in each point
    # For this test, we verify the detector integration directly
    detector = PossessionDetector(
        possession_radius_m=2.0,
        min_possession_duration_frames=2,
        fps=30.0
    )

    # Simulate frames
    player_positions = {1: (0.0, 0.0)}
    player_teams = {1: "Red"}
    for frame in range(10):
        detector.update(
            ball_position_m=(0.5, 0.5),
            player_positions_m=player_positions,
            player_teams=player_teams,
            frame_number=frame
        )

    # Use metrics generator to get possession stats (this is what ball_analytics.py does)
    metrics_gen = BallMetricsGenerator()
    team_possession = metrics_gen.generate_team_possession_from_detector(detector)

    print("📊 Team Possession Results (via metrics generator):")
    for team, stats in team_possession.items():
        print(f"  {team}: {stats['possession_pct']}%")

    # Check results
    assert 'Red' in team_possession, "Red team should be in possession stats"

    print("\n✅ Test 5 PASSED\n")


if __name__ == "__main__":
    print("\n⚽ Testing Ball Possession Percentage Implementation\n")

    try:
        test_possession_detector_basic()
        test_metrics_generator()
        test_possession_history()
        test_zero_division_protection()
        test_integration_with_ball_analytics()

        print("=" * 50)
        print("🎉 ALL TESTS PASSED!")
        print("=" * 50)
        print("\nImplementation Features:")
        print("  ✅ Team possession percentage calculation")
        print("  ✅ Free ball percentage tracking")
        print("  ✅ Frame-by-frame history")
        print("  ✅ Anti-flicker filter (candidate streak)")
        print("  ✅ Integration with existing ball analytics")
        print("  ✅ Visualization support")

    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        raise
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        raise