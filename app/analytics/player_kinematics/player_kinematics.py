"""
Player Kinematics Engine

Main orchestrator for computing player movement metrics from tracking trajectories.
Combines trajectory cleaning, smoothing, speed, acceleration, direction,
sprint detection, metrics generation, validation, and visualization.
"""

import logging
import os
import time
from typing import Dict, List, Optional
import numpy as np
import pandas as pd

from app.analytics.player_kinematics.trajectory import TrajectoryCleaner
from app.analytics.player_kinematics.smoothing import TrajectorySmoother
from app.analytics.player_kinematics.speed import SpeedCalculator
from app.analytics.player_kinematics.acceleration import AccelerationCalculator
from app.analytics.player_kinematics.direction import DirectionAnalyzer
from app.analytics.player_kinematics.sprint_detection import SprintDetector
from app.analytics.player_kinematics.metrics import PlayerMetricsGenerator
from app.analytics.player_kinematics.validation import KinematicsValidator
from app.analytics.player_kinematics.visualization import KinematicsVisualizer

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)


class PlayerKinematicsEngine:
    """
    Main kinematics engine that orchestrates all processing steps.
    """

    def __init__(self, config: Dict):
        """
        Initializes PlayerKinematicsEngine with configuration.

        Args:
            config: Configuration dict with kinematics parameters.
        """
        self.config = config.get('player_kinematics', {})
        fps = config.get('video', {}).get('fps', 25.0)
        canvas = config.get('pitch', {})
        canvas_size = (canvas.get('canvas_width', 1050), canvas.get('canvas_height', 680))
        pitch_size = (canvas.get('length_m', 105.0), canvas.get('width_m', 68.0))

        # Modules
        self.trajectory_cleaner = TrajectoryCleaner(
            max_jump_m=self.config.get('max_jump_m', 5.0),
            min_track_frames=self.config.get('min_track_frames', 5),
            max_gap_frames=self.config.get('max_gap_frames', 3),
            duplicate_threshold_m=self.config.get('duplicate_threshold_m', 0.1)
        )

        self.smoother = TrajectorySmoother(
            method=self.config.get('smoothing_method', 'savgol'),
            window_size=self.config.get('smoothing_window', 5),
            polyorder=self.config.get('smoothing_polyorder', 2)
        )

        self.speed_calc = SpeedCalculator(
            max_speed_kmh=self.config.get('max_speed_kmh', 40.0),
            rolling_window_frames=self.config.get('rolling_window_frames', 5)
        )

        self.accel_calc = AccelerationCalculator(
            rolling_window_frames=self.config.get('rolling_window_frames', 5)
        )

        self.direction_analyzer = DirectionAnalyzer(
            heading_smooth_window=self.config.get('heading_smooth_window', 3)
        )

        self.sprint_detector = SprintDetector(
            speed_threshold_kmh=self.config.get('sprint_speed_threshold_kmh', 25.0),
            min_duration_s=self.config.get('sprint_min_duration_s', 2.0),
            min_distance_m=self.config.get('sprint_min_distance_m', 10.0)
        )

        self.metrics_gen = PlayerMetricsGenerator()
        self.validator = KinematicsValidator(
            max_speed_kmh=self.config.get('max_speed_kmh', 40.0),
            max_acceleration_ms2=self.config.get('max_acceleration_ms2', 10.0),
            max_jump_m=self.config.get('max_jump_m', 5.0)
        )

        self.visualizer = KinematicsVisualizer(
            output_path=os.path.join(config.get('video', {}).get('output_dir', 'outputs'),
                                    'player_kinematics_debug.mp4'),
            canvas_size=canvas_size,
            pitch_size_m=pitch_size
        )

        self.fps = fps
        self.results = {}

    def _compute_distance_along_track(self, points: List[Dict]) -> List[Dict]:
        """Adds cumulative distance to trajectory points."""
        cumulative = 0.0
        for i, p in enumerate(points):
            if i > 0:
                prev = np.array(points[i-1]['smoothed_world_position'])
                curr = np.array(points[i]['smoothed_world_position'])
                cumulative += np.linalg.norm(curr - prev)
            p['distance_m'] = cumulative
        return points

    def _compute_high_intensity_runs(
        self,
        points: List[Dict],
        speed_threshold_kmh: Optional[float] = None,
        min_duration_s: Optional[float] = None,
        min_distance_m: Optional[float] = None
    ) -> int:
        """
        Detects high-intensity runs (same criteria as sprints but lower threshold).
        """
        if speed_threshold_kmh is None:
            speed_threshold_kmh = self.config.get('high_intensity_speed_kmh', 20.0)
        if min_duration_s is None:
            min_duration_s = self.config.get('high_intensity_min_duration_s', 3.0)
        if min_distance_m is None:
            min_distance_m = self.config.get('high_intensity_min_distance_m', 15.0)

        run_count = 0
        in_run = False
        run_indices = []

        for i, p in enumerate(points):
            if p.get('speed_kmh', 0) >= speed_threshold_kmh:
                if not in_run:
                    in_run = True
                    run_indices = [i]
                else:
                    run_indices.append(i)
            else:
                if in_run:
                    start = run_indices[0]
                    end = run_indices[-1]
                    duration = points[end]['timestamp'] - points[start]['timestamp']
                    distance = points[end]['distance_m'] - points[start]['distance_m']
                    if duration >= min_duration_s and distance >= min_distance_m:
                        run_count += 1
                    in_run = False
                    run_indices = []

        # Trailing run
        if in_run and run_indices:
            start = run_indices[0]
            end = run_indices[-1]
            duration = points[end]['timestamp'] - points[start]['timestamp']
            distance = points[end]['distance_m'] - points[start]['distance_m']
            if duration >= min_duration_s and distance >= min_distance_m:
                run_count += 1

        return run_count

    def process(
        self,
        all_tracks: Dict[int, List[Dict]]
    ) -> Dict:
        """
        Runs full kinematics pipeline on all player tracks.

        Args:
            all_tracks: Dict mapping track_id to list of raw track dicts.

        Returns:
            Results dict with processed tracks, sprints, summaries, validation, etc.
        """
        start_time = time.time()
        logger.info("Player Kinematics Engine started with %d tracks", len(all_tracks))

        # Step 1: Clean trajectories
        cleaned_tracks, clean_issues = self.trajectory_cleaner.clean_tracks_batch(all_tracks)
        logger.info("Cleaned tracks: %d valid, %d rejected",
                    len(cleaned_tracks), clean_issues['rejected_too_short'])

        # Step 2: Smooth trajectories
        smoothed_tracks = self.smoother.smooth_tracks_batch(cleaned_tracks)

        # Step 3: Distance along track
        for track_id, points in smoothed_tracks.items():
            smoothed_tracks[track_id] = self._compute_distance_along_track(points)

        # Step 4: Speed
        tracks_speed = self.speed_calc.process_batch(smoothed_tracks)

        # Step 5: Acceleration
        tracks_accel = self.accel_calc.process_batch(tracks_speed)

        # Step 6: Direction
        tracks_direction = self.direction_analyzer.process_batch(tracks_accel)

        # Step 7: Sprint detection
        all_sprints = self.sprint_detector.detect_batch(tracks_direction)

        # Step 8: High intensity runs
        high_intensity = {}
        for track_id, points in tracks_direction.items():
            high_intensity[track_id] = self._compute_high_intensity_runs(points)

        # Step 9: Validation
        validation_reports = self.validator.validate_batch(tracks_direction)
        global_validation = self.validator.get_global_report(validation_reports)

        # Step 10: Metrics summary
        summaries = self.metrics_gen.generate_batch_summary(tracks_direction, all_sprints)
        for track_id, s in summaries.items():
            s['rejected_frames'] = validation_reports.get(track_id, {}).get('total_issues', 0)
            s['high_intensity_runs'] = high_intensity.get(track_id, 0)

        # Sprint summary
        sprint_summary = self.sprint_detector.get_summary(all_sprints)

        processing_time_s = time.time() - start_time
        processing_fps = len(all_tracks) / processing_time_s if processing_time_s > 0 else 0

        logger.info("Player Kinematics Engine completed in %.2fs (%.1f tracks/s)",
                    processing_time_s, processing_fps)

        return {
            'processed_tracks': tracks_direction,
            'sprints': all_sprints,
            'sprint_summary': sprint_summary,
            'player_summaries': summaries,
            'validation_reports': validation_reports,
            'global_validation': global_validation,
            'clean_issues': clean_issues,
            'processing_time_s': round(processing_time_s, 3),
            'processing_fps': round(processing_fps, 2)
        }

    def export_csvs(
        self,
        results: Dict,
        output_dir: str
    ) -> Dict[str, str]:
        """
        Exports kinematics data to CSV files.

        Args:
            results: Results dict from process().
            output_dir: Directory to save outputs.

        Returns:
            Dict mapping output type to file path.
        """
        os.makedirs(output_dir, exist_ok=True)
        paths = {}

        # Per-frame kinematics CSV
        rows = []
        for track_id, points in results['processed_tracks'].items():
            for p in points:
                row = {'track_id': track_id}
                row.update(p)
                rows.append(row)
        df = pd.DataFrame(rows)
        kinematics_path = os.path.join(output_dir, 'player_kinematics.csv')
        df.to_csv(kinematics_path, index=False)
        paths['kinematics'] = kinematics_path

        # Player summary CSV
        summary_rows = []
        summaries = results.get('player_summaries', {})
        for track_id, s in summaries.items():
            row = {'track_id': track_id}
            row.update(s)
            summary_rows.append(row)
        df_summary = pd.DataFrame(summary_rows)
        summary_path = os.path.join(output_dir, 'player_summary.csv')
        df_summary.to_csv(summary_path, index=False)
        paths['summary'] = summary_path

        # Validation CSV
        val_rows = []
        for track_id, v in results.get('validation_reports', {}).items():
            row = {'track_id': track_id}
            row.update(v)
            val_rows.append(row)
        df_val = pd.DataFrame(val_rows)
        val_path = os.path.join(output_dir, 'player_validation.csv')
        df_val.to_csv(val_path, index=False)
        paths['validation'] = val_path

        logger.info("CSV exports saved to %s", output_dir)
        return paths

    def generate_debug_video(
        self,
        results: Dict,
        fps: float
    ) -> Optional[str]:
        """
        Generates debug visualization video.

        Args:
            results: Results dict from process().
            fps: Video frame rate.

        Returns:
            Path to debug video or None.
        """
        all_points_by_frame = {}
        for track_id, points in results['processed_tracks'].items():
            for p in points:
                frame = p.get('frame_number')
                if frame is not None:
                    all_points_by_frame.setdefault(frame, []).append(p)

        success = self.visualizer.generate_debug_video(all_points_by_frame, fps)
        if success:
            return self.visualizer.output_path
        return None

    def generate_validation_report(
        self,
        results: Dict,
        output_path: str,
        match_name: str = "match30"
    ) -> str:
        """
        Generates markdown validation report.

        Args:
            results: Results dict from process().
            output_path: Path to save markdown report.
            match_name: Name of the match for report title.

        Returns:
            Path to the generated report.
        """
        summaries = results.get('player_summaries', {})
        validation = results.get('global_validation', {})
        clean_issues = results.get('clean_issues', {})

        distances = [s.get('total_distance_m', 0) for s in summaries.values() if s.get('valid')]
        speeds = [s.get('avg_speed_kmh', 0) for s in summaries.values() if s.get('valid')]
        max_speeds = [s.get('max_speed_kmh', 0) for s in summaries.values() if s.get('valid')]
        accels = [s.get('avg_acceleration_ms2', 0) for s in summaries.values() if s.get('valid')]
        max_accels = [s.get('max_acceleration_ms2', 0) for s in summaries.values() if s.get('valid')]
        sprint_counts = [s.get('sprint_count', 0) for s in summaries.values()]

        total_sprints = sum(sprint_counts)

        report = f"""# Player Kinematics Validation Report - {match_name}

## Summary

| Metric | Value |
|--------|-------|
| Players Processed | {len(summaries)} |
| Average Track Length | {np.mean([s.get('valid_frames', 0) for s in summaries.values()]):.1f} frames |
| Average Distance Covered | {np.mean(distances):.2f} m |
| Maximum Distance | {np.max(distances):.2f} m |
| Average Speed | {np.mean(speeds):.2f} km/h |
| Maximum Speed | {np.max(max_speeds):.2f} km/h |
| Average Acceleration | {np.mean(accels):.3f} m/s² |
| Maximum Acceleration | {np.max(max_accels):.3f} m/s² |
| Sprint Count | {total_sprints} |
| Rejected Samples | {clean_issues.get('rejected_too_short', 0)} |
| Processing FPS | {results.get('processing_fps', 0):.2f} |
| Processing Time | {results.get('processing_time_s', 0):.3f} s |

## Data Quality

| Issue | Count |
|-------|-------|
| Missing Positions | {validation.get('missing_positions', 0)} |
| Invalid Coordinates | {validation.get('invalid_coordinates', 0)} |
| Negative Distances | {validation.get('negative_distances', 0)} |
| Impossible Speeds | {validation.get('impossible_speeds', 0)} |
| Impossible Accelerations | {validation.get('impossible_accelerations', 0)} |
| Trajectory Jumps | {validation.get('trajectory_jumps', 0)} |

## Memory Usage

Processing handled {len(summaries)} players with batch processing.

---

*Generated by Player Kinematics Engine*
"""

        with open(output_path, 'w') as f:
            f.write(report)

        logger.info("Validation report saved to %s", output_path)
        return output_path