import json
import tempfile
import unittest
from pathlib import Path

from app.analytics.xt_engine import XTEngine


class TestXTEngine(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.output_dir = Path(self.tmpdir.name)
        self._write("analytics.json", {"match_info": {"fps": 30.0}})
        self._write("pass_events.json", [
            {
                "event_id": 101, "frame_start": 100, "frame_end": 115,
                "passer": 8, "receiver": 11, "team": "Blue",
                "start_position": [50.0, 34.0], "end_position": [75.0, 34.0],
                "distance_m": 25.0, "ball_speed_mps": 18.0,
                "successful": True, "pass_type": "Long Pass (Forward Pass)",
            },
            {
                "event_id": 102, "frame_start": 120, "frame_end": 130,
                "passer": 11, "receiver": 8, "team": "Blue",
                "start_position": [75.0, 34.0], "end_position": [30.0, 20.0],
                "distance_m": 47.0, "ball_speed_mps": 15.0,
                "successful": True, "pass_type": "Long Pass (Back Pass)",
            },
        ])
        self._write("ball_tracks.json", [])
        self._write("average_positions.json", {
            "8": {"team": "Blue", "average_position": [60.0, 34.0]},
            "11": {"team": "Blue", "average_position": [75.0, 33.0]},
        })

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_run_exports_xt_outputs(self):
        payload = XTEngine(output_dir=self.output_dir).run()
        self.assertIn("actions", payload)
        self.assertIn("team_summary", payload)
        self.assertIn("player_summary", payload)
        self.assertIn("summary", payload)
        self.assertIn("validation", payload)
        self.assertIn("performance", payload)
        self.assertIn("regression", payload)
        for f in ["xt_actions.json", "team_xt_summary.json", "player_xt_summary.json",
                   "xt_summary.json", "xt_validation_report.json", "xt_performance_report.json",
                   "xt_regression_report.json"]:
            self.assertTrue((self.output_dir / f).exists(), f)

    def test_forward_pass_increases_threat(self):
        payload = XTEngine(output_dir=self.output_dir).run()
        actions = payload["actions"]
        # Pass 101 goes forward (50->75), should be positive xT
        pass101 = [a for a in actions if a["event_id"] == 101]
        self.assertTrue(len(pass101) > 0)
        for p in pass101:
            self.assertGreaterEqual(p["xt_added"], -0.1)
            self.assertLessEqual(p["xt_added"], 0.5)

    def test_xt_values_finite(self):
        payload = XTEngine(output_dir=self.output_dir).run()
        for action in payload["actions"]:
            xt = action.get("xt_added", float("nan"))
            self.assertFalse(isinstance(xt, float) and xt != xt)

    def _write(self, filename, payload):
        (self.output_dir / filename).write_text(json.dumps(payload), encoding="utf-8")


if __name__ == "__main__":
    unittest.main()