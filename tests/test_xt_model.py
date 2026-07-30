import unittest

from app.analytics.xt_grid import XTGrid
from app.analytics.xt_features import XTFeatureExtractor


class TestXTGrid(unittest.TestCase):
    def test_grid_initialization(self):
        grid = XTGrid(grid_key="12x8")
        self.assertEqual(grid.cols, 12)
        self.assertEqual(grid.rows, 8)
        self.assertEqual(len(grid.matrix), 8)
        self.assertEqual(len(grid.matrix[0]), 12)

    def test_cell_from_position(self):
        grid = XTGrid(grid_key="12x8", pitch_length_m=105.0, pitch_width_m=68.0)
        col, row = grid.cell_from_position(0.0, 0.0)
        self.assertEqual(col, 0)
        self.assertEqual(row, 0)
        col, row = grid.cell_from_position(104.0, 67.0)
        self.assertEqual(col, 11)
        self.assertEqual(row, 7)

    def test_centre_increases_xt(self):
        grid = XTGrid(grid_key="12x8")
        centre_xt = grid.get_xt_from_position(52.5, 34.0)
        own_corner_xt = grid.get_xt_from_position(0.0, 0.0)
        self.assertGreater(centre_xt, own_corner_xt)

    def test_near_goal_higher_xt(self):
        grid = XTGrid(grid_key="12x8")
        near_goal = grid.get_xt_from_position(100.0, 34.0)
        midfield = grid.get_xt_from_position(52.5, 34.0)
        self.assertGreater(near_goal, midfield)

    def test_feature_extraction(self):
        extractor = XTFeatureExtractor()
        pass_event = {
            "event_id": 101, "frame_start": 100, "frame_end": 115,
            "passer": 8, "receiver": 11, "team": "Blue",
            "start_position": [50.0, 34.0], "end_position": [75.0, 34.0],
            "distance_m": 25.0, "ball_speed_mps": 18.0,
            "pass_type": "Long Pass (Forward Pass)",
        }
        features = extractor.extract_pass(
            pass_event, xt_start=0.05, xt_end=0.12,
            start_cell=(6, 4), end_cell=(9, 4),
        )
        payload = features.to_dict()
        self.assertEqual(payload["event_id"], 101)
        self.assertEqual(payload["action"], "pass")
        self.assertEqual(payload["player_id"], 8)
        self.assertAlmostEqual(payload["xt_added"], 0.07)

    def test_carry_extraction(self):
        extractor = XTFeatureExtractor()
        carry_event = {
            "event_id": 10001, "player_id": 8, "team": "Blue",
            "start_position": [30.0, 20.0], "end_position": [55.0, 25.0],
            "distance_m": 26.0, "carry_speed_mps": 5.0,
        }
        features = extractor.extract_carry(
            carry_event, xt_start=0.02, xt_end=0.08,
            start_cell=(4, 2), end_cell=(7, 3),
        )
        payload = features.to_dict()
        self.assertEqual(payload["action"], "carry")
        self.assertEqual(payload["player_id"], 8)
        self.assertAlmostEqual(payload["xt_added"], 0.06)


if __name__ == "__main__":
    unittest.main()