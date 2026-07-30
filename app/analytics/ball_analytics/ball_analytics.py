"""
Ball Analytics Engine

Main orchestrator for computing football ball analytics from tracking trajectories.
Combines trajectory cleaning, smoothing, speed, acceleration, possession detection,
touch detection, pass detection, metrics generation, validation, and visualization.
"""

import logging
import os
import time
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from app.analytics.ball_analytics.trajectory import BallTrajectoryCleaner
from app.analytics.ball_analytics.smoothing import BallTrajectorySmoother
from app.analytics.ball_analytics.ball_speed import BallSpeedCalculator
from app.analytics.ball_analytics.acceleration import BallAccelerationCalculator
from app.analytics.ball_analytics.possession import PossessionDetector
from app.analytics.ball_analytics.touch_detection import TouchDetector
from app.analytics.ball_analytics.pass_detection import PassDetector
from app.analytics.ball_analytics.pass_metrics import PassMetricsCalculator
from app.analytics.ball_analytics.metrics import BallMetricsGenerator
from app.analytics.ball_analytics.validation import BallValidator
from app.analytics.ball_analytics.visualization import BallAnalyticsVisualizer

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)


class BallAnalyticsEngine:
    """
    Main ball analytics engine that orchestrates all processing steps.
    """

    def __init__(self, config: Dict):
        self.config = config.get('ball_analytics', {})
        fps = config.get('video', {}).get('fps', 25.0)
        canvas = config.get('pitch', {})
        canvas_size = (canvas.get('canvas_width', 1050), canvas.get('canvas_height', 680))
        pitch_size = (canvas.get('length_m', 105.0), canvas.get('width_m', 68.0))

        self.trajectory_cleaner = BallTrajectoryCleaner(
            max_jump_m=self.config.get('max_jump_m', 3.0),
            min_track_frames=self.config.get('min_track_frames', 3),
            max_gap_frames=self.config.get('max_gap_frames', 2),
            duplicate_threshold_m=self.config.get('duplicate_threshold_m', 0.05)
        )

        self.smoother = BallTrajectorySmoother(
            method=self.config.get('smoothing_method', 'savgol'),
            window_size=self.config.get('smoothing_window', 5),
            polyorder=self.config.get('smoothing_polyorder', 2)
        )

        self.speed_calc = BallSpeedCalculator(
            max_speed_kmh=self.config.get('max_speed_kmh', 120.0),
            rolling_window_frames=self.config.get('rolling_window_frames', 5)
        )

        self.accel_calc = BallAccelerationCalculator(
            rolling_window_frames=self.config.get('rolling_window_frames', 5)
        )

        self.possession_detector = PossessionDetector(
            possession_radius_m=self.config.get('possession_radius_m', 1.5),
            min_possession_duration_frames=self.config.get('min_possession_duration_frames', 3)
        )

        self.touch_detector = TouchDetector(
            min_touch_duration_frames=self.config.get('min_touch_duration_frames', 1)
        )

        self.pass_detector = PassDetector(
            min_pass_distance_m=self.config.get('min_pass_distance_m', 1.0),
            max_pass_duration_s=self.config.get('max_pass_duration_s', 5.0),
            min_pass_speed_kmh=self.config.get('min_pass_speed_kmh', 5.0)
        )

        self.pass_metrics = PassMetricsCalculator()
        self.metrics_gen = BallMetricsGenerator()
        self.validator = BallValidator(
            max_speed_kmh=self.config.get('max_speed_kmh', 120.0),
            max_acceleration_ms2=self.config.get('max_acceleration_ms2', 50.0),
            max_jump_m=self.config.get('max_jump_m', 3.0)
        )

        self.visualizer = BallAnalyticsVisualizer(
            output_path=os.path.join(config.get('video', {}).get('output_dir', 'outputs'),
                                    'ball_analytics_debug.mp4'),
            canvas_size=canvas_size,
            pitch_size_m=pitch_size
        )

        self.fps = fps
        self.results = {}

    def _compute_distance_along_track(self, points: List[Dict]) -> List[Dict]:
        cumulative = 0.0
        for i, p in enumerate(points):
            if i > 0:
                prev = np.array(points[i-1]['smoothed_world_position'])
                curr = np.array(points[i]['smoothed_world_position'])
                cumulative += np.linalg.norm(curr - prev)
            p['distance_m'] = cumulative
        return points

    def process(
        self,
        ball_tracks: Dict[int, List[Dict]],
        player_positions_by_frame: Dict[int, Dict[int, Tuple[float, float]]],
        player_teams: Dict[int, str],
        player_confidences_by_frame: Optional[Dict[int, Dict[int, float]]] = None
    ) -> Dict:
        """
        Runs full ball analytics pipeline.

        Args:
            ball_tracks: Dict mapping track_id to raw ball track data.
            player_positions_by_frame: Dict mapping frame_number to dict of track_id to position.
            player_teams: Dict mapping track_id to team id.
            player_confidences_by_frame: Optional dict of frame -> track_id -> confidence.

        Returns:
            Results dict with all ball analytics.
        """
        start_time = time.time()
        logger.info("Ball Analytics Engine started with %d ball tracks", len(ball_tracks))

        # Step 1: Clean trajectories
        cleaned_tracks, clean_issues = self.trajectory_cleaner.clean_batch(ball_tracks)
        logger.info("Cleaned ball tracks: %d valid", len(cleaned_tracks))

        # Step 2: Smooth trajectories
        smoothed_tracks = self.smoother.smooth_batch(cleaned_tracks)

        # Step 3: Distance along track
        for track_id, points in smoothed_tracks.items():
            smoothed_tracks[track_id] = self._compute_distance_along_track(points)

        # Step 4: Speed
        tracks_speed = self.speed_calc.process_batch(smoothed_tracks)

        # Step 5: Acceleration
        tracks_accel = self.accel_calc.process_batch(tracks_speed)

        # Step 6-8: Possession, touches, passes
        all_possession = {}
        all_touches = []
        all_passes = []

        for track_id, points in tracks_accel.items():
            touches = []
            passes = []
            possession_events = []

            for i, p in enumerate(points):
                frame = p.get('frame_number')
                ts = p.get('timestamp')
                ball_pos = p.get('smoothed_world_position') or p.get('clean_world_position')
                ball_speed_kmh = p.get('ball_speed_kmh', 0)

                player_positions = player_positions_by_frame.get(frame, {})
                player_conf = player_confidences_by_frame.get(frame, {}) if player_confidences_by_frame else {}

                possession = self.possession_detector.update(
                    ball_position_m=ball_pos,
                    player_positions_m=player_positions,
                    player_teams=player_teams,
                    frame_number=frame,
                    ball_confidence=p.get('confidence', 1.0),
                    player_confidences=player_conf
                )
                all_possession[frame] = possession

                touch = self.touch_detector.update(possession, ball_pos, frame, ts)
                if touch:
                    touches.append(touch)

                pass_event = self.pass_detector.update(
                    possession=possession,
                    ball_position=ball_pos,
                    ball_speed_kmh=ball_speed_kmh,
                    frame_number=frame,
                    timestamp=ts,
                    player_positions_m=player_positions
                )
                if pass_event:
                    passes.append(pass_event)

            all_touches.extend(touches)
            all_passes.extend(passes)

        # Step 9: Metrics
        ball_summaries = {}
        for track_id, points in tracks_accel.items():
            touches = [t for t in all_touches if t['track_id'] == track_id]
            passes = [p for p in all_passes if p['passer_id'] == track_id]
            ball_summaries[track_id] = self.metrics_gen.generate_ball_summary(points, touches, passes)

        # Generate team possession using new detector methods
        team_possession = self.metrics_gen.generate_team_possession_from_detector(
            self.possession_detector
        )

        pass_summary = self.pass_metrics.get_summary(all_passes)

        # Step 10: Validation
        validation_reports = self.validator.validate_batch(tracks_accel)
        global_validation = self.validator.get_global_report(validation_reports)

        processing_time_s = time.time() - start_time
        processing_fps = len(ball_tracks) / processing_time_s if processing_time_s > 0 else 0

        logger.info("Ball Analytics Engine completed in %.2fs", processing_time_s)

        return {
            'processed_tracks': tracks_accel,
            'possession_events': self.possession_detector.possession_events,
            'touches': all_touches,
            'passes': all_passes,
            'ball_summaries': ball_summaries,
            'team_possession': team_possession,
            'pass_summary': pass_summary,
            'validation_reports': validation_reports,
            'global_validation': global_validation,
            'clean_issues': clean_issues,
            'processing_time_s': round(processing_time_s, 3),
            'processing_fps': round(processing_fps, 2)
        }

    def export_csvs(self, results: Dict, output_dir: str) -> Dict[str, str]:
        os.makedirs(output_dir, exist_ok=True)
        paths = {}

        # ball_analytics.csv
        rows = []
        for track_id, points in results['processed_tracks'].items():
            for p in points:
                row = {'track_id': track_id}
                row.update(p)
                rows.append(row)
        df = pd.DataFrame(rows)
        ball_path = os.path.join(output_dir, 'ball_analytics.csv')
        df.to_csv(ball_path, index=False)
        paths['ball_analytics'] = ball_path

        # pass_events.csv
        df_passes = pd.DataFrame(results['passes'])
        pass_path = os.path.join(output_dir, 'pass_events.csv')
        df_passes.to_csv(pass_path, index=False)
        paths['pass_events'] = pass_path

        # ball_summary.csv
        summary_rows = []
        for track_id, s in results['ball_summaries'].items():
            row = {'track_id': track_id}
            row.update(s)
            summary_rows.append(row)
        df_ball_summary = pd.DataFrame(summary_rows)
        ball_summary_path = os.path.join(output_dir, 'ball_summary.csv')
        df_ball_summary.to_csv(ball_summary_path, index=False)
        paths['ball_summary'] = ball_summary_path

        # team_possession.csv
        team_rows = []
        for tid, s in results['team_possession'].items():
            row = {'team_id': tid}
            row.update(s)
            team_rows.append(row)
        df_team = pd.DataFrame(team_rows)
        team_path = os.path.join(output_dir, 'team_possession.csv')
        df_team.to_csv(team_path, index=False)
        paths['team_possession'] = team_path

        # ball_validation.csv
        val_rows = []
        for track_id, v in results.get('validation_reports', {}).items():
            row = {'track_id': track_id}
            row.update(v)
            val_rows.append(row)
        df_val = pd.DataFrame(val_rows)
        val_path = os.path.join(output_dir, 'ball_validation.csv')
        df_val.to_csv(val_path, index=False)
        paths['validation'] = val_path

        logger.info("CSV exports saved to %s", output_dir)
        return paths

    def generate_debug_video(self, results: Dict, fps: float) -> Optional[str]:
        all_points_by_frame = {}
        for track_id, points in results['processed_tracks'].items():
            for p in points:
                frame = p.get('frame_number')
                if frame is not None:
                    all_points_by_frame.setdefault(frame, []).append(p)

        # Compute possession percentages from detector
        possession_pct = self.possession_detector.get_possession_percentage()

        success = self.visualizer.generate_debug_video(all_points_by_frame, fps, possession_pct=possession_pct)
        if success:
            return self.visualizer.output_path
        return None

    def generate_validation_report(self, results: Dict, output_path: str, match_name: str = "match30") -> str:
        summaries = results.get('ball_summaries', {})
        validation = results.get('global_validation', {})
        clean_issues = results.get('clean_issues', {})
        pass_summary = results.get('pass_summary', {})
        team_possession = results.get('team_possession', {})

        distances = [s.get('total_distance_m', 0) for s in summaries.values() if s.get('valid')]
        speeds = [s.get('avg_speed_kmh', 0) for s in summaries.values() if s.get('valid')]
        max_speeds = [s.get('max_speed_kmh', 0) for s in summaries.values() if s.get('valid')]

        possession_pcts = [s.get('possession_pct', 0) for s in team_possession.values()]

        report = f"""# Ball Analytics Validation Report - {match_name}

## Summary

| Metric | Value |
|--------|-------|
| Ball Distance | {np.mean(distances):.2f} m |
| Average Speed | {np.mean(speeds):.2f} km/h |
| Maximum Speed | {np.max(max_speeds):.2f} km/h |
| Total Passes | {pass_summary.get('total_passes', 0)} |
| Successful Passes | {pass_summary.get('successful_passes', 0)} |
| Pass Accuracy | {pass_summary.get('accuracy_pct', 0):.2f}% |
| Possession Events | {len(results.get('possession_events', []))} |
| Total Touches | {len(results.get('touches', []))} |
| Rejected Samples | {clean_issues.get('rejected_too_short', 0)} |
| Processing FPS | {results.get('processing_fps', 0):.2f} |
| Processing Time | {results.get('processing_time_s', 0):.3f} s |

## Team Possession

| Team | Possession % | Possession Time (s) | Events |
|-------|-------------|---------------------|--------|
"""

        for tid, s in team_possession.items():
            report += f"| {tid} | {s.get('possession_pct', 0):.2f}% | {s.get('possession_time_s', 0):.2f} | {s.get('possession_events', 0)} |\n"

        report += f"""
## Data Quality

| Issue | Count |
|-------|-------|
| Missing Positions | {validation.get('missing_positions', 0)} |
| Invalid Coordinates | {validation.get('invalid_coordinates', 0)} |
| Negative Distances | {validation.get('negative_distances', 0)} |
| Impossible Speeds | {validation.get('impossible_speeds', 0)} |
| Impossible Accelerations | {validation.get('impossible_accelerations', 0)} |
| Trajectory Jumps | {validation.get('trajectory_jumps', 0)} |

---

*Generated by Ball Analytics Engine*
"""

        with open(output_path, 'w') as f:
            f.write(report)

        logger.info("Ball validation report saved to %s", output_path)
        return output_path