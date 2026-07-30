import unittest

from app.analytics.xg_features import XGFeatureExtractor
from app.analytics.xg_model import RuleBasedXGModel


class TestXGModel(unittest.TestCase):
    def test_rule_based_model_scores_close_central_chance_higher(self):
        extractor = XGFeatureExtractor()
        model = RuleBasedXGModel()
        # Close-range central shot (8m out, centre of goal) — high xG
        close = extractor.extract({
            "event_id": 1,
            "frame": 10,
            "player_id": 9,
            "team": "Blue",
            "launch_position": [97.0, 34.0],
            "distance_m": 8.0,
            "angle_to_goal_deg": 45.0,
            "ball_speed_mps": 20.0,
        })
        # Long-range wide shot (35m out, far from centre) — low xG
        far = extractor.extract({
            "event_id": 2,
            "frame": 20,
            "player_id": 9,
            "team": "Blue",
            "launch_position": [70.0, 5.0],
            "distance_m": 35.0,
            "angle_to_goal_deg": 8.0,
            "ball_speed_mps": 20.0,
        })

        close_xg = model.predict_proba(close, extractor.to_model_vector(close))
        far_xg = model.predict_proba(far, extractor.to_model_vector(far))

        self.assertGreater(close_xg, far_xg)
        self.assertGreaterEqual(close_xg, 0.0)
        self.assertLessEqual(close_xg, 1.0)

    def test_feature_export_contains_required_fields(self):
        features = XGFeatureExtractor().extract({
            "event_id": 12,
            "frame": 90,
            "team": "Red",
            "player_id": 8,
            "launch_position": [87.0, 31.0],
            "ball_speed_mps": 22.1,
        }, fps=30.0)
        payload = features.to_dict()

        self.assertEqual(payload["shot_id"], 12)
        self.assertIn("distance_m", payload)
        self.assertIn("angle_deg", payload)
        self.assertIn("pressure_score", payload)
        self.assertEqual(payload["ball_speed_mps"], 22.1)


if __name__ == "__main__":
    unittest.main()
