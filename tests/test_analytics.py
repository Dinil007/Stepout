"""
Unit tests for Analytics Modules (Possession, Pass, Shot Detector)
"""

import unittest
from app.analytics.ball_possession import BallPossessionAnalyzer
from app.analytics.pass_detector import PassDetector
from app.analytics.shot_detector import ShotDetector


class TestAnalytics(unittest.TestCase):

    def test_possession_analyzer(self):
        poss = BallPossessionAnalyzer(fps=30.0)
        res = poss.update((50.0, 34.0), {1: (50.0, 34.0)}, team_assignments={1: 0}, frame_number=1)
        self.assertIsNotNone(res)
        self.assertIn("team_name", res)

    def test_pass_detector(self):
        detector = PassDetector(fps=30.0)
        # Player 1 launches pass
        detector.update(frame_number=1, ball_position_m=None, player_positions_m={1: (10.0, 10.0)}, possessor_id=1, team_assignments={1: "Red"})
        detector.update(frame_number=2, ball_position_m=(15.0, 10.0), player_positions_m={1: (10.0, 10.0)}, possessor_id=None, team_assignments={1: "Red"})
        res = detector.update(frame_number=10, ball_position_m=(25.0, 10.0), player_positions_m={2: (25.0, 10.0)}, possessor_id=2, team_assignments={2: "Red"})
        self.assertIsNotNone(res)
        self.assertEqual(res["passer"], 1)
        self.assertEqual(res["receiver"], 2)
        self.assertTrue(res["successful"])

    def test_shot_detector(self):
        detector = ShotDetector(fps=30.0)
        detector.update(frame_number=1, ball_position_m=(80.0, 30.0), player_positions_m={9: (80.0, 30.0)}, possessor_id=9, team_assignments={9: "Red"})
        detector.update(frame_number=2, ball_position_m=(85.0, 31.0), player_positions_m={9: (80.0, 30.0)}, possessor_id=None, team_assignments={9: "Red"})
        res = detector.update(frame_number=15, ball_position_m=(104.0, 33.5), player_positions_m={9: (80.0, 30.0)}, possessor_id=None, team_assignments={9: "Red"})
        self.assertIsNotNone(res)
        self.assertEqual(res["player_id"], 9)
        self.assertIn("Shot", res["shot_type"])


if __name__ == "__main__":
    unittest.main()
