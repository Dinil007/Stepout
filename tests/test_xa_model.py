import unittest

from app.analytics.xa_features import XAFeatureExtractor
from app.analytics.xa_model import RuleBasedXAModel


class TestXAModel(unittest.TestCase):
    def test_rule_based_model_scores_high_quality_assist_higher(self):
        extractor = XAFeatureExtractor()
        model = RuleBasedXAModel()

        # High-quality assist: short pass to close-range, high-xG shot
        high_quality_pass = {
            "event_id": 1, "frame_start": 100, "frame_end": 105,
            "passer": 8, "receiver": 11, "team": "Blue",
            "start_position": [90.0, 34.0], "end_position": [96.0, 34.0],
            "distance_m": 6.0, "pass_type": "Short Pass (Forward Pass)",
        }
        high_quality_shot = {
            "event_id": 1, "frame": 115, "player_id": 11, "team": "Blue",
            "launch_position": [96.0, 34.0],
            "distance_m": 8.0, "angle_to_goal_deg": 45.0,
            "xg": 0.45,
            "ball_speed_mps": 22.0, "shot_type": "Shot on Target",
        }

        # Low-quality assist: back pass into own half, low-xG shot
        low_quality_pass = {
            "event_id": 2, "frame_start": 200, "frame_end": 215,
            "passer": 9, "receiver": 10, "team": "Blue",
            "start_position": [70.0, 30.0], "end_position": [50.0, 10.0],
            "distance_m": 28.0, "pass_type": "Long Pass (Back Pass)",
        }
        low_quality_shot = {
            "event_id": 2, "frame": 230, "player_id": 10, "team": "Blue",
            "launch_position": [50.0, 10.0],
            "distance_m": 55.0, "angle_to_goal_deg": 5.0,
            "xg": 0.03,
            "ball_speed_mps": 15.0, "shot_type": "Long-range Shot (Off Target)",
        }

        high = extractor.extract(high_quality_pass, high_quality_shot, fps=30.0)
        low = extractor.extract(low_quality_pass, low_quality_shot, fps=30.0)

        high_xa = model.predict_proba(high, extractor.to_model_vector(high))
        low_xa = model.predict_proba(low, extractor.to_model_vector(low))

        self.assertGreater(high_xa, low_xa)
        self.assertGreaterEqual(high_xa, 0.0)
        self.assertLessEqual(high_xa, 1.0)

    def test_feature_export_contains_required_fields(self):
        extractor = XAFeatureExtractor()
        pass_event = {
            "event_id": 105, "frame_start": 100, "frame_end": 112,
            "passer": 8, "receiver": 11, "team": "Blue",
            "start_position": [85.0, 30.0], "end_position": [94.0, 34.0],
            "distance_m": 10.5, "pass_type": "Medium Pass (Forward Pass)",
        }
        shot_event = {
            "event_id": 4, "frame": 125, "player_id": 11, "team": "Blue",
            "launch_position": [94.0, 34.0],
            "distance_m": 12.0, "angle_to_goal_deg": 35.0,
            "ball_speed_mps": 22.1, "shot_type": "Shot on Target",
        }
        features = extractor.extract(pass_event, shot_event, fps=30.0)
        payload = features.to_dict()

        self.assertEqual(payload["pass_id"], 105)
        self.assertEqual(payload["shot_id"], 4)
        self.assertIn("pass_length_m", payload)
        self.assertIn("shot_xg", payload)
        self.assertIn("pass_type", payload)


if __name__ == "__main__":
    unittest.main()