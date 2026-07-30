"""
Unit tests for PassNetworkAnalyzer & PassNetworkVisualizer
"""

import unittest
from app.analytics.pass_network import PassNetworkAnalyzer, PassNetworkVisualizer


class TestPassNetwork(unittest.TestCase):

    def setUp(self):
        self.analyzer = PassNetworkAnalyzer(fps=30.0)
        self.visualizer = PassNetworkVisualizer()

        self.player_histories = {
            1: [(10.0, 20.0), (12.0, 22.0), (11.0, 21.0)],
            2: [(30.0, 25.0), (32.0, 27.0), (31.0, 26.0)],
            3: [(60.0, 30.0), (65.0, 35.0), (62.0, 32.0)]
        }
        self.team_assignments = {1: 0, 2: 0, 3: 1} # 1 & 2: Red, 3: Blue

        self.pass_events = [
            {"event_id": 1, "passer": 1, "receiver": 2, "successful": True, "start_position": [11.0, 21.0], "end_position": [31.0, 26.0]},
            {"event_id": 2, "passer": 1, "receiver": 2, "successful": True, "start_position": [11.0, 21.0], "end_position": [31.0, 26.0]},
            {"event_id": 3, "passer": 2, "receiver": 3, "successful": False, "start_position": [31.0, 26.0], "end_position": [62.0, 32.0]}
        ]

    def test_average_positions(self):
        avg_pos = self.analyzer.compute_average_positions(self.player_histories, self.team_assignments)
        self.assertIn(1, avg_pos)
        self.assertEqual(avg_pos[1]["team"], "Red")
        self.assertEqual(len(avg_pos[1]["average_position"]), 2)

    def test_team_shape(self):
        shapes = self.analyzer.compute_team_shape(self.player_histories, self.team_assignments)
        self.assertIn("Red", shapes)
        self.assertGreater(shapes["Red"]["width_m"], 0.0)
        self.assertGreater(shapes["Red"]["depth_m"], 0.0)

    def test_pass_network_analysis(self):
        res = self.analyzer.analyze_pass_network(self.pass_events, self.player_histories, self.team_assignments)
        self.assertIn("nodes", res)
        self.assertIn("edges", res)
        self.assertGreater(len(res["edges"]), 0)
        # Edge (1, 2) weight should be 2
        edge12 = next((e for e in res["edges"] if e["passer"] == 1 and e["receiver"] == 2), None)
        self.assertIsNotNone(edge12)
        self.assertEqual(edge12["pass_count"], 2)

    def test_visualization_rendering(self):
        res = self.analyzer.analyze_pass_network(self.pass_events, self.player_histories, self.team_assignments)
        canvas = self.visualizer.render_pass_network(res, team_filter="Red")
        self.assertEqual(canvas.shape[2], 3)


if __name__ == "__main__":
    unittest.main()
