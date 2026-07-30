"""
Pipeline Manager

Orchestrates all pipeline stages in order.
Responsible for:
- Loading configuration
- Initializing all stages
- Executing every stage in order
- Validating intermediate outputs
- Handling errors with graceful fallbacks
- Exporting final results
"""

from __future__ import annotations
import time
import json
from pathlib import Path
from typing import Dict, List, Optional, Any
import numpy as np

from app.pipeline.data_models import (
    PipelineInput, PipelineOutput, StageResult,
    VideoMetadata, FrameData
)
from app.pipeline.pipeline_logger import PipelineLogger
from app.pipeline.stages import (
    VideoInputStage, PreprocessingStage, DetectionStage,
    TrackingStage, TeamClassificationStage, PoseEstimationStage,
    CameraMotionStage, HomographyStage, PlayerKinematicsStage,
    BallAnalyticsStage, BiomechanicsStage, VisualizationStage,
    ExportStage
)


class PipelineManager:
    """
    Production pipeline manager that chains all modules.
    Follows a strict stage order with validation at each step.
    """

    STAGE_ORDER = [
        "video_input",
        "preprocessing",
        "detection",
        "tracking",
        "team_classification",
        "pose_estimation",
        "camera_motion",
        "homography",
        "player_kinematics",
        "ball_analytics",
        "biomechanics",
        "visualization",
        "export"
    ]

    def __init__(self, config: Dict):
        self.config = config
        self.output_dir = Path(config.get('video', {}).get('output_dir', 'outputs'))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir = self.output_dir / "logs"
        self.log_dir.mkdir(exist_ok=True)

        self.logger = PipelineLogger(self.log_dir, "stepout_pipeline")
        self.stages: Dict[str, Any] = {}
        self.stage_results: Dict[str, StageResult] = {}

        self._init_stages()

    def _init_stages(self) -> None:
        """Initialises all pipeline stages using DI."""
        self.logger.stage_info("init", "Initialising all pipeline stages")
        self.stages = {
            "video_input": VideoInputStage("video_input", self.config, self.logger),
            "preprocessing": PreprocessingStage("preprocessing", self.config, self.logger),
            "detection": DetectionStage("detection", self.config, self.logger),
            "tracking": TrackingStage("tracking", self.config, self.logger),
            "team_classification": TeamClassificationStage("team_classification", self.config, self.logger),
            "pose_estimation": PoseEstimationStage("pose_estimation", self.config, self.logger),
            "camera_motion": CameraMotionStage("camera_motion", self.config, self.logger),
            "homography": HomographyStage("homography", self.config, self.logger),
            "player_kinematics": PlayerKinematicsStage("player_kinematics", self.config, self.logger),
            "ball_analytics": BallAnalyticsStage("ball_analytics", self.config, self.logger),
            "biomechanics": BiomechanicsStage("biomechanics", self.config, self.logger),
            "visualization": VisualizationStage("visualization", self.config, self.logger),
            "export": ExportStage("export", self.config, self.logger)
        }
        self.logger.stage_info("init", f"{len(self.stages)} stages initialised")

    def _record_result(self, stage_name: str, result: StageResult) -> None:
        self.stage_results[stage_name] = result

    def run(self, pipeline_input: PipelineInput) -> PipelineOutput:
        """
        Executes the full pipeline end-to-end.

        Args:
            pipeline_input: Input containing video path and options.

        Returns:
            PipelineOutput with all results and export paths.
        """
        t_start = time.time()
        output = PipelineOutput()
        self.logger.stage_info("pipeline", f"Pipeline started: {pipeline_input.video_path}")

        try:
            # Stage 1: Video Input
            result = self.stages["video_input"].process(
                pipeline_input.video_path,
                pipeline_input.max_frames
            )
            self._record_result("video_input", result)
            if not result.success or result.data is None:
                raise RuntimeError(f"Video input failed: {result.error}")
            meta: VideoMetadata
            frames: List[FrameData]
            meta, frames = result.data
            output.video_metadata = meta
            output.total_frames_processed = len(frames)
            self.logger.stage_info("pipeline", f"Loaded {len(frames)} frames from {meta.input_path.name}")

            # Stage 2: Preprocessing
            result = self.stages["preprocessing"].process(frames)
            self._record_result("preprocessing", result)
            if result.success and result.data is not None:
                frames = result.data

            # Stage 3: Detection
            result = self.stages["detection"].process(frames)
            self._record_result("detection", result)
            if not result.success or result.data is None:
                raise RuntimeError(f"Detection failed: {result.error}")
            detections = result.data
            total_detections = sum(getattr(d, 'detection_count', 0) for d in detections)
            print(f"[VERIFY] Number of detections: {total_detections}")

            # Stage 4: Tracking
            result = self.stages["tracking"].process(frames, detections)
            self._record_result("tracking", result)
            if not result.success or result.data is None:
                raise RuntimeError(f"Tracking failed: {result.error}")
            tracks = result.data
            total_tracked_players = sum(getattr(t, 'track_count', 0) for t in tracks)
            total_tracked_balls = sum(1 for t in tracks if getattr(t, 'ball_tracked', False))
            print(f"[VERIFY] Number of tracked players: {total_tracked_players}")
            print(f"[VERIFY] Number of tracked balls: {total_tracked_balls}")

            # Stage 5: Team Classification
            result = self.stages["team_classification"].process(tracks)
            self._record_result("team_classification", result)
            if not result.success or result.data is None:
                raise RuntimeError(f"Team classification failed: {result.error}")
            team_data = result.data
            total_classified = sum(len(getattr(t, 'team_assignments', {})) for t in team_data)
            print(f"[VERIFY] Number of classified players: {total_classified}")

            # Stage 6: Pose Estimation
            result = self.stages["pose_estimation"].process(frames, tracks)
            self._record_result("pose_estimation", result)
            pose_data = result.data if result.success else []
            total_pose_skeletons = sum(len(getattr(p, 'player_keypoints', {})) for p in pose_data)
            print(f"[VERIFY] Number of pose skeletons: {total_pose_skeletons}")

            # Stage 7: Camera Motion
            result = self.stages["camera_motion"].process(frames)
            self._record_result("camera_motion", result)
            motions = result.data if result.success else []

            # Stage 8: Homography
            result = self.stages["homography"].process(frames, tracks, motions)
            self._record_result("homography", result)
            if not result.success or result.data is None:
                self.logger.stage_warning("homography", "Homography failed — using pixel coordinates")
                homography_data = []
            else:
                homography_data = result.data
            total_homography_points = sum(len(getattr(h, 'player_world_positions', {})) for h in homography_data)
            print(f"[VERIFY] Number of homography points: {total_homography_points}")

            # Stage 9: Player Kinematics
            result = self.stages["player_kinematics"].process(homography_data, team_data)
            self._record_result("player_kinematics", result)
            kinematics = result.data if result.success else {}
            total_kinematic_records = sum(len(v) for v in kinematics.values()) if isinstance(kinematics, dict) else 0
            print(f"[VERIFY] Number of kinematic records: {total_kinematic_records}")

            # Stage 10: Ball Analytics
            result = self.stages["ball_analytics"].process(homography_data, team_data)
            self._record_result("ball_analytics", result)
            ball_analytics = result.data if result.success else {}

            # Stage 11: Biomechanics
            result = self.stages["biomechanics"].process(kinematics, pose_data)
            self._record_result("biomechanics", result)
            biomechanics = result.data if result.success else {}

            # Stage 12: Visualization
            annotated_path = self.output_dir / "annotated_video.mp4"
            result = self.stages["visualization"].process(frames, tracks, team_data, homography_data, annotated_path)
            self._record_result("visualization", result)
            if result.success:
                output.annotated_video_path = Path(str(result.data)) if result.data else annotated_path

            # Stage 13: Export
            export_stage: ExportStage = self.stages["export"]
            self._export_results(export_stage, output, kinematics, ball_analytics, biomechanics)

            output.success = True
            output.total_execution_time_s = round(time.time() - t_start, 3)
            output.stage_results = self.stage_results

            # Generate summary JSON
            summary = self._build_summary(output)
            summary_path = self.output_dir / "summary.json"
            export_stage.export_summary_json(summary, summary_path)
            output.summary_json_path = summary_path

            self.logger.log_summary(output.total_frames_processed)
            self.logger.save_summary_json(self.log_dir / "pipeline_summary.json")

            self.logger.stage_info("pipeline", f"Pipeline completed in {output.total_execution_time_s:.2f}s")

        except Exception as e:
            output.success = False
            output.error = str(e)
            output.total_execution_time_s = round(time.time() - t_start, 3)
            output.stage_results = self.stage_results
            self.logger.stage_error("pipeline", str(e))

            # Save partial summary even on failure
            try:
                summary = self._build_summary(output)
                summary_path = self.output_dir / "summary.json"
                with open(summary_path, 'w') as f:
                    json.dump(summary, f, indent=2, default=str)
            except Exception:
                pass

        return output

    def _export_results(self, export_stage: ExportStage, output: PipelineOutput,
                        kinematics: Dict, ball_analytics: Dict, biomechanics: Dict) -> None:
        """Exports all analytics to CSV files."""
        # Player metrics
        if kinematics:
            rows = []
            for track_id, points in kinematics.items():
                for p in points:
                    row = {'track_id': track_id}
                    row.update(p)
                    rows.append(row)
            if rows:
                player_csv = self.output_dir / "player_metrics.csv"
                export_stage.export_csv(rows, player_csv)
                output.player_metrics_csv = player_csv
                output.player_kinematics_data = kinematics

        # Ball metrics
        if ball_analytics:
            rows = []
            for track_id, points in ball_analytics.get('processed_tracks', {}).items():
                for p in points:
                    row = {'track_id': track_id}
                    row.update(p)
                    rows.append(row)
            if rows:
                ball_csv = self.output_dir / "ball_metrics.csv"
                export_stage.export_csv(rows, ball_csv)
                output.ball_metrics_csv = ball_csv
                output.ball_analytics_data = ball_analytics

        # Biomechanics metrics
        if biomechanics:
            rows = []
            for track_id, metrics in biomechanics.items():
                if isinstance(metrics, dict):
                    row = {'track_id': track_id}
                    row.update(metrics)
                    rows.append(row)
            if rows:
                biomech_csv = self.output_dir / "biomechanics_metrics.csv"
                export_stage.export_csv(rows, biomech_csv)
                output.biomechanics_csv = biomech_csv

    def _build_summary(self, output: PipelineOutput) -> Dict:
        """Builds a comprehensive summary dict from pipeline outputs."""
        meta = output.video_metadata
        stage_times = {
            name: r.execution_time_s
            for name, r in self.stage_results.items()
        }
        return {
            'pipeline_status': 'SUCCESS' if output.success else 'FAILED',
            'total_execution_time_s': output.total_execution_time_s,
            'video': {
                'path': str(meta.input_path) if meta else '',
                'total_frames': meta.total_frames if meta else 0,
                'fps': meta.fps if meta else 0,
                'resolution': f"{meta.width}x{meta.height}" if meta else "0x0",
                'duration_s': meta.duration_s if meta else 0
            },
            'stage_timings_s': stage_times,
            'outputs': {
                'annotated_video': str(output.annotated_video_path) if output.annotated_video_path else None,
                'player_metrics_csv': str(output.player_metrics_csv) if output.player_metrics_csv else None,
                'ball_metrics_csv': str(output.ball_metrics_csv) if output.ball_metrics_csv else None,
                'biomechanics_csv': str(output.biomechanics_csv) if output.biomechanics_csv else None,
                'summary_json': str(output.summary_json_path) if output.summary_json_path else None
            },
            'error': output.error
        }

    def get_stage_result(self, stage_name: str) -> Optional[StageResult]:
        return self.stage_results.get(stage_name)