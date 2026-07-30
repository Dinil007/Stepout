import json
import tempfile
import unittest
from pathlib import Path

from app.ai.match_analyst import MatchAnalyst, OfflineLLMProvider


class TestAIMatchAnalyst(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.output_dir = Path(self.tmpdir.name)
        self._write("analytics.json", {
            "match_info": {"fps": 30.0, "processed_frames": 300},
            "player_count": 2,
        })
        self._write("pass_summary.json", {
            "total_passes": 2,
            "completed_passes": 1,
            "overall_accuracy_pct": 50.0,
            "team_pass_summary": {
                "Blue": {
                    "total_passes": 1,
                    "completed_passes": 1,
                    "accuracy_pct": 100.0,
                    "avg_distance_m": 12.0,
                },
                "Red": {
                    "total_passes": 1,
                    "completed_passes": 0,
                    "accuracy_pct": 0.0,
                    "avg_distance_m": 10.0,
                },
            },
            "player_pass_summary": {
                "8": {"attempted": 1, "completed": 1, "accuracy_pct": 100.0},
                "9": {"attempted": 1, "completed": 0, "accuracy_pct": 0.0},
            },
        })
        self._write("pass_events.json", [
            {
                "event_id": 1,
                "frame_start": 10,
                "team": "Blue",
                "passer": 8,
                "receiver": 9,
                "distance_m": 12.0,
                "successful": True,
            }
        ])
        self._write("shot_summary.json", {"total_shots": 0, "shots_on_target": 0})
        self._write("shot_events.json", [])
        self._write("team_possession_summary.json", {
            "team_possession_pct": {"Blue": 55.0, "Red": 45.0},
            "total_possession_time_seconds": {"Blue": 5.5, "Red": 4.5},
        })
        self._write("team_passing_summary.json", {
            "Blue": {
                "progressive_passes": 1,
                "tactical_shape": {"width_m": 50.0, "depth_m": 40.0, "compactness": 1.0},
            }
        })
        self._write("average_positions.json", {
            "8": {
                "player_id": 8,
                "team": "Blue",
                "average_position": [50.0, 30.0],
                "movement_radius": 20.0,
                "total_samples": 30,
            },
            "9": {
                "player_id": 9,
                "team": "Red",
                "average_position": [40.0, 20.0],
                "movement_radius": 5.0,
                "total_samples": 20,
            },
        })

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_build_context_computes_insights_before_llm(self):
        analyst = MatchAnalyst(
            output_dir=self.output_dir,
            provider=OfflineLLMProvider(),
            match_id="test-match",
        )

        context = analyst.build_context()

        self.assertEqual(context["match_id"], "test-match")
        self.assertEqual(context["insights"]["possession"]["team_dominance"]["id"], "Blue")
        self.assertEqual(context["insights"]["passing"]["best_passing_pair"]["passer"], "8")

    def test_generate_report_exports_required_files_and_all_player_ratings(self):
        analyst = MatchAnalyst(
            output_dir=self.output_dir,
            provider=OfflineLLMProvider(),
            match_id="test-match",
        )

        report = analyst.generate_match_report()

        self.assertIn("8", report["player_reports"])
        self.assertIn("9", report["player_reports"])
        self.assertTrue(report["validation"]["ratings_for_every_detected_player"])
        for filename in [
            "ai_match_summary.md",
            "ai_match_summary.pdf",
            "ai_team_report.json",
            "ai_player_reports.json",
            "coach_report.md",
            "opposition_report.md",
            "recommendations.json",
            "ai_validation_report.json",
            "ai_performance_report.json",
            "ai_regression_report.json",
        ]:
            self.assertTrue((self.output_dir / filename).exists(), filename)

    def _write(self, filename, payload):
        (self.output_dir / filename).write_text(
            json.dumps(payload),
            encoding="utf-8",
        )


if __name__ == "__main__":
    unittest.main()

