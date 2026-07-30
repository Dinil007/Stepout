import json
import tempfile
import unittest
from pathlib import Path

from app.analytics.xg_engine import XGEngine


class TestXGEngine(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.output_dir = Path(self.tmpdir.name)
        self._write("analytics.json", {"match_info": {"fps": 30.0}})
        self._write("pass_events.json", [])
        self._write("average_positions.json", {
            "8": {"team": "Blue", "average_position": [90.0, 34.0]},
            "4": {"team": "Red", "average_position": [91.0, 35.0]},
        })
        self._write("shot_events.json", [
            {
                "event_id": 4,
                "frame": 120,
                "player_id": 8,
                "team": "Blue",
                "launch_position": [91.0, 34.0],
                "distance_m": 14.2,
                "angle_to_goal_deg": 28.4,
                "ball_speed_mps": 22.0,
                "shot_type": "Shot on Target",
            },
            {
                "event_id": 5,
                "frame": 180,
                "player_id": 9,
                "team": "Red",
                "launch_position": [70.0, 8.0],
                "distance_m": 30.0,
                "angle_to_goal_deg": 8.0,
                "ball_speed_mps": 18.0,
                "shot_type": "Long-range Shot (Off Target)",
            },
        ])

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_run_exports_xg_outputs_and_validates(self):
        payload = XGEngine(output_dir=self.output_dir, force_rule_based=True).run()

        self.assertEqual(len(payload["shots"]), 2)
        self.assertTrue(payload["validation"]["every_detected_shot_has_xg"])
        self.assertTrue(payload["validation"]["xg_values_between_0_and_1"])
        self.assertTrue(payload["validation"]["team_xg_equals_player_xg"])
        for filename in [
            "xg_shots.json",
            "team_xg_summary.json",
            "player_xg_summary.json",
            "xg_summary.json",
            "xg_validation_report.json",
            "xg_performance_report.json",
            "xg_regression_report.json",
            "xg_shot_map.png",
            "team_xg_chart.png",
            "player_xg_chart.png",
            "xg_timeline.png",
        ]:
            self.assertTrue((self.output_dir / filename).exists(), filename)

    def test_team_xg_equals_sum_of_shots(self):
        payload = XGEngine(output_dir=self.output_dir, force_rule_based=True).run()
        team_total = sum(team["total_xg"] for team in payload["team_summary"].values())
        shot_total = sum(shot["xg"] for shot in payload["shots"])

        self.assertAlmostEqual(team_total, shot_total, places=3)

    def _write(self, filename, payload):
        (self.output_dir / filename).write_text(json.dumps(payload), encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
