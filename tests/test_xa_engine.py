import json
import tempfile
import unittest
from pathlib import Path

from app.analytics.xa_engine import XAEngine


class TestXAEngine(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.output_dir = Path(self.tmpdir.name)
        self._write("analytics.json", {"match_info": {"fps": 30.0}})
        self._write("pass_events.json", [
            {
                "event_id": 101, "frame_start": 100, "frame_end": 115,
                "passer": 8, "receiver": 11, "team": "Blue",
                "start_position": [85.0, 35.0], "end_position": [92.0, 34.0],
                "distance_m": 7.5, "successful": True, "pass_type": "Short Pass (Forward Pass)",
            },
            {
                "event_id": 102, "frame_start": 120, "frame_end": 135,
                "passer": 11, "receiver": 8, "team": "Blue",
                "start_position": [92.0, 34.0], "end_position": [70.0, 30.0],
                "distance_m": 22.5, "successful": True, "pass_type": "Long Pass (Back Pass)",
            },
        ])
        self._write("shot_events.json", [
            {
                "event_id": 4, "frame": 130, "player_id": 11, "team": "Blue",
                "launch_position": [92.0, 34.0],
                "distance_m": 14.2, "angle_to_goal_deg": 28.4,
                "ball_speed_mps": 22.0, "shot_type": "Shot on Target",
            },
        ])
        self._write("average_positions.json", {
            "8": {"team": "Blue", "average_position": [85.0, 34.0]},
            "11": {"team": "Blue", "average_position": [90.0, 33.0]},
            "4": {"team": "Red", "average_position": [91.0, 35.0]},
        })

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_run_exports_xa_outputs_and_validates(self):
        payload = XAEngine(output_dir=self.output_dir, force_rule_based=True).run()

        self.assertIn("passes", payload)
        self.assertIn("team_summary", payload)
        self.assertIn("player_summary", payload)
        self.assertIn("summary", payload)
        self.assertIn("validation", payload)
        self.assertIn("performance", payload)
        self.assertIn("regression", payload)

        for filename in [
            "xa_passes.json", "team_xa_summary.json", "player_xa_summary.json",
            "xa_summary.json", "xa_validation_report.json", "xa_performance_report.json",
            "xa_regression_report.json",
        ]:
            self.assertTrue((self.output_dir / filename).exists(), filename)

    def test_linking_links_correct_pass_to_shot(self):
        payload = XAEngine(output_dir=self.output_dir, force_rule_based=True).run()
        passes = payload["passes"]
        # pass 101 (frame 115 -> shot frame 130, receiver=11) should be linked; pass 102 (receiver=8) not
        self.assertEqual(len(passes), 1)
        self.assertEqual(passes[0]["pass_id"], 101)
        self.assertEqual(passes[0]["shot_id"], 4)
        self.assertEqual(passes[0]["player"], 8)

    def test_xa_values_between_zero_and_one(self):
        payload = XAEngine(output_dir=self.output_dir, force_rule_based=True).run()
        for pa in payload["passes"]:
            self.assertGreaterEqual(pa["xA"], 0.0)
            self.assertLessEqual(pa["xA"], 1.0)

    def _write(self, filename, payload):
        (self.output_dir / filename).write_text(json.dumps(payload), encoding="utf-8")


if __name__ == "__main__":
    unittest.main()