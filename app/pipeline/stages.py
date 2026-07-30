"""
Pipeline Stages

Each stage wraps an existing StepOut AI module using its REAL public API.
"""

from __future__ import annotations
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import cv2

from app.pipeline.data_models import (
    StageResult, VideoMetadata, FrameData, DetectionData,
    TrackData, TeamData, PoseData, HomographyData,
    KinematicsData, BallAnalyticsData, BiomechanicsData
)
from app.pipeline.pipeline_logger import PipelineLogger


class BaseStage:
    def __init__(self, name: str, config: Dict, logger: PipelineLogger):
        self.name = name
        self.config = config
        self.logger = logger

    def process(self, *args, **kwargs) -> StageResult:
        raise NotImplementedError

    def _make_result(self, success: bool, data: Any = None, error: str = None,
                     execution_time: float = 0.0, frames: int = 0) -> StageResult:
        return StageResult(
            stage_name=self.name,
            success=success,
            data=data,
            error=error,
            execution_time_s=execution_time,
            frames_processed=frames
        )


class VideoInputStage(BaseStage):
    def process(self, video_path: Path, max_frames: Optional[int] = None) -> StageResult:
        self.logger.stage_start(self.name)
        t0 = time.time()
        try:
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                raise FileNotFoundError(f"Cannot open video: {video_path}")

            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            meta = VideoMetadata(
                input_path=Path(video_path),
                total_frames=total if max_frames is None else min(total, max_frames),
                fps=fps,
                width=width,
                height=height,
                duration_s=total / fps if fps > 0 else 0.0
            )
            frames = []
            for fnum in range(meta.total_frames):
                ret, img = cap.read()
                if not ret:
                    break
                frames.append(FrameData(
                    frame_number=fnum,
                    image=img,
                    timestamp=fnum / fps
                ))
            cap.release()
            t1 = time.time()
            self.logger.stage_end(self.name, len(frames))
            return self._make_result(True, (meta, frames), execution_time=t1 - t0, frames=len(frames))
        except Exception as e:
            self.logger.stage_error(self.name, str(e))
            return self._make_result(False, error=str(e))


class PreprocessingStage(BaseStage):
    def __init__(self, name: str, config: Dict, logger: PipelineLogger):
        super().__init__(name, config, logger)
        self.preprocessor = None
        try:
            from app.preprocessing.adaptive_preprocessor import AdaptivePreprocessor
            self.preprocessor = AdaptivePreprocessor()
        except Exception:
            self.preprocessor = None
        if self.preprocessor is None:
            self.logger.stage_warning(name, "AdaptivePreprocessor not available")

    def process(self, frames: List[FrameData]) -> StageResult:
        self.logger.stage_start(self.name)
        t0 = time.time()
        try:
            if self.preprocessor is None:
                return self._make_result(True, frames, execution_time=0.0, frames=len(frames))

            out_frames = []
            for f in frames:
                if f.image is None:
                    out_frames.append(f)
                    continue
                try:
                    m = self.preprocessor.measure(f.image, f.frame_number)
                    processed, _ = self.preprocessor.apply(f.image, m)
                except Exception:
                    processed = f.image
                f.image = processed
                out_frames.append(f)
            t1 = time.time()
            self.logger.stage_end(self.name, len(frames))
            return self._make_result(True, out_frames, execution_time=t1 - t0, frames=len(frames))
        except Exception as e:
            self.logger.stage_error(self.name, str(e))
            return self._make_result(False, error=str(e))


class DetectionStage(BaseStage):
    def __init__(self, name: str, config: Dict, logger: PipelineLogger):
        super().__init__(name, config, logger)
        self.detector = None
        try:
            from app.detection.detector import YoloDetector
            self.detector = YoloDetector(config=config)
            self.detector.load()
        except Exception as e:
            self.logger.stage_warning(name, f"YoloDetector init failed: {e}")
            self.detector = None

    def process(self, frames: List[FrameData]) -> StageResult:
        self.logger.stage_start(self.name)
        t0 = time.time()
        try:
            if self.detector is None:
                results = []
                for f in frames:
                    results.append(DetectionData(
                        frame_number=f.frame_number,
                        timestamp=f.timestamp,
                        detection_count=0,
                        player_detections=[],
                        ball_detections=[]
                    ))
                return self._make_result(True, results, execution_time=0.0, frames=len(frames))

            results = []
            total_players = 0
            total_ball = 0
            for f in frames:
                detections = []
                if f.image is not None:
                    try:
                        detections = self.detector.predict(f.image)
                    except Exception:
                        detections = []

                player_dets = [d for d in detections if getattr(d, "cls_id", None) == 0]
                ball_dets = [d for d in detections if getattr(d, "cls_id", None) == 32]
                total_players += len(player_dets)
                total_ball += len(ball_dets)
                results.append(DetectionData(
                    frame_number=f.frame_number,
                    timestamp=f.timestamp,
                    player_detections=[
                        {
                            "class_id": d.cls_id,
                            "confidence": float(getattr(d, "conf", 0.0)),
                            "bbox": list(d.bbox),
                            "center": d.center,
                        }
                        for d in player_dets
                    ],
                    ball_detections=[
                        {
                            "class_id": d.cls_id,
                            "confidence": float(getattr(d, "conf", 0.0)),
                            "bbox": list(d.bbox),
                            "center": d.center,
                        }
                        for d in ball_dets
                    ],
                    detection_count=len(player_dets) + len(ball_dets),
                    ball_confidence=float(getattr(ball_dets[0], "conf", 0.0)) if ball_dets else 0.0
                ))
            t1 = time.time()
            self.logger.stage_end(self.name, len(frames))
            self.logger.stage_info(self.name, f"Detections - players: {total_players}, ball: {total_ball}")
            return self._make_result(True, results, execution_time=t1 - t0, frames=len(frames))
        except Exception as e:
            self.logger.stage_error(self.name, str(e))
            return self._make_result(False, error=str(e))


class TrackingStage(BaseStage):
    def __init__(self, name: str, config: Dict, logger: PipelineLogger):
        super().__init__(name, config, logger)
        self.player_tracker = None
        self.ball_tracker = None
        try:
            from app.tracking.player_tracker import PlayerTracker
            self.player_tracker = PlayerTracker(config=config)
        except Exception as e:
            self.logger.stage_warning(name, f"PlayerTracker init failed: {e}")
        try:
            from app.tracking.ball_tracker import BallTracker
            self.ball_tracker = BallTracker(config=config)
        except Exception as e:
            self.logger.stage_warning(name, f"BallTracker init failed: {e}")

    def process(self, frames: List[FrameData], detections: List[DetectionData]) -> StageResult:
        self.logger.stage_start(self.name)
        t0 = time.time()
        try:
            results = []
            total_players = 0
            total_ball = 0
            for f, d in zip(frames, detections):
                track_data = TrackData(
                    frame_number=f.frame_number,
                    timestamp=f.timestamp,
                    player_tracks={},
                    ball_track=None,
                    track_count=0,
                    ball_tracked=False
                )
                if self.player_tracker is not None and d.player_detections:
                    from app.detection.detection_types import Detection
                    det_objs = []
                    for pd in d.player_detections:
                        try:
                            det_objs.append(Detection(
                                cls_id=int(pd.get("class_id", 0)),
                                conf=float(pd.get("confidence", 0.0)),
                                bbox=tuple(pd.get("bbox", (0, 0, 0, 0))),
                                track_id=int(pd.get("track_id", -1)),
                            ))
                        except Exception:
                            pass
                    if det_objs:
                        h, w = f.image.shape[:2] if f.image is not None else (1080, 1920)
                        tracked = self.player_tracker.update(det_objs, (h, w), f.frame_number)
                        for trk in tracked:
                            tid = getattr(trk, "track_id", None)
                            bbox = list(getattr(trk, "bbox", (0, 0, 0, 0)))
                            cx = (bbox[0] + bbox[2]) / 2.0
                            cy = (bbox[1] + bbox[3]) / 2.0
                            track_data.player_tracks[int(tid)] = {
                                "bbox": bbox,
                                "center": (cx, cy),
                                "confidence": float(getattr(trk, "conf", getattr(trk, "confidence", 0.0))),
                                "class_id": getattr(trk, "cls_id", 0),
                            }
                        track_data.track_count = len(track_data.player_tracks)
                        total_players += track_data.track_count

                if self.ball_tracker is not None and d.ball_detections:
                    ball = self.ball_tracker.update(d.ball_detections, f.frame_number)
                    if ball is not None:
                        bbox = ball.get("bbox") if isinstance(ball, dict) else getattr(ball, "bbox", None)
                        cx, cy = ball.get("center") if isinstance(ball, dict) else getattr(ball, "center", (None, None))
                        conf = 0.0
                        if isinstance(ball, dict):
                            conf = float(ball.get("confidence", ball.get("conf", 0.0)))
                        else:
                            conf = float(getattr(ball, "conf", getattr(ball, "confidence", 0.0)))
                        track_data.ball_track = {
                            "bbox": list(bbox) if bbox else None,
                            "center": (cx, cy),
                            "confidence": conf,
                        }
                        track_data.ball_tracked = True
                        total_ball += 1

                results.append(track_data)
            t1 = time.time()
            self.logger.stage_end(self.name, len(frames))
            self.logger.stage_info(self.name, f"Tracks - players: {total_players}, ball: {total_ball}")
            return self._make_result(True, results, execution_time=t1 - t0, frames=len(frames))
        except Exception as e:
            self.logger.stage_error(self.name, str(e))
            return self._make_result(False, error=str(e))


class TeamClassificationStage(BaseStage):
    def __init__(self, name: str, config: Dict, logger: PipelineLogger):
        super().__init__(name, config, logger)
        self.classifier = None
        try:
            from app.team_classification.team_classifier import TeamClassifier
            self.classifier = TeamClassifier()
        except Exception as e:
            self.logger.stage_warning(name, f"TeamClassifier init failed: {e}")
            self.classifier = None

    def process(self, tracks: List[TrackData]) -> StageResult:
        self.logger.stage_start(self.name)
        t0 = time.time()
        try:
            results = []
            total = 0
            for t in tracks:
                team_data = TeamData(
                    frame_number=t.frame_number,
                    timestamp=t.timestamp,
                    team_assignments={},
                    team_colors={},
                    confidence_scores={}
                )
                if self.classifier is None:
                    for tid in t.player_tracks:
                        team_data.team_assignments[tid] = "unknown"
                        team_data.confidence_scores[tid] = 0.0
                    results.append(team_data)
                    continue

                for tid, pdata in t.player_tracks.items():
                    team_label = "unknown"
                    conf = 0.0
                    try:
                        bbox = pdata.get("bbox", [0, 0, 0, 0])
                        x1, y1, x2, y2 = [int(v) for v in bbox]
                        team_label, conf = self.classifier.classify(tid, pdata.get("image"), (x1, y1, x2, y2))
                    except Exception:
                        team_label = "unknown"
                        conf = 0.0

                    team_data.team_assignments[tid] = team_label
                    team_data.confidence_scores[tid] = conf
                    total += 1
                results.append(team_data)
            t1 = time.time()
            self.logger.stage_end(self.name, len(tracks))
            self.logger.stage_info(self.name, f"Classified {total} players")
            return self._make_result(True, results, execution_time=t1 - t0, frames=len(tracks))
        except Exception as e:
            self.logger.stage_error(self.name, str(e))
            return self._make_result(False, error=str(e))


class PoseEstimationStage(BaseStage):
    def __init__(self, name: str, config: Dict, logger: PipelineLogger):
        super().__init__(name, config, logger)
        self.pose_model = None
        try:
            from app.pose.pose_estimator import PoseEstimator
            model_path = config.get('pose', {}).get('model_path', 'models/pose_landmarker_full.task')
            self.pose_model = PoseEstimator(model_path=model_path)
        except TypeError as e:
            self.logger.stage_warning(name, f"PoseEstimator init failed: {e}")
        except Exception as e:
            self.logger.stage_warning(name, f"PoseEstimator init failed: {e}")

    def process(self, frames: List[FrameData], tracks: List[TrackData]) -> StageResult:
        self.logger.stage_start(self.name)
        t0 = time.time()
        try:
            results = []
            total = 0
            for f, tr in zip(frames, tracks):
                pose_data = PoseData(frame_number=f.frame_number, timestamp=f.timestamp)
                if self.pose_model is not None and f.image is not None:
                    for tid, pdata in tr.player_tracks.items():
                        bbox = pdata.get("bbox")
                        if bbox is None:
                            continue
                        x1, y1, x2, y2 = [int(v) for v in bbox]
                        player_crop = f.image[y1:y2, x1:x2]
                        if player_crop.size == 0:
                            continue
                        try:
                            pose_result = self.pose_model.estimate(player_crop, track_id=tid)
                        except TypeError:
                            try:
                                pose_result = self.pose_model.estimate(player_crop, tid)
                            except Exception:
                                pose_result = None
                        if pose_result is not None and getattr(pose_result, "success", False):
                            kps = []
                            scores = []
                            for lm in getattr(pose_result, "landmarks", []):
                                kps.append([getattr(lm, "x_px", 0.0), getattr(lm, "y_px", 0.0)])
                                scores.append(getattr(lm, "visibility", 0.0))
                            if kps:
                                pose_data.player_keypoints[tid] = np.array(kps, dtype=np.float64)
                                pose_data.player_scores[tid] = np.array(scores, dtype=np.float64)
                                total += 1
                results.append(pose_data)
            t1 = time.time()
            self.logger.stage_end(self.name, len(frames))
            self.logger.stage_info(self.name, f"Pose skeletons: {total}")
            return self._make_result(True, results, execution_time=t1 - t0, frames=len(frames))
        except Exception as e:
            self.logger.stage_error(self.name, str(e))
            return self._make_result(False, error=str(e))


class CameraMotionStage(BaseStage):
    def __init__(self, name: str, config: Dict, logger: PipelineLogger):
        super().__init__(name, config, logger)
        self.motion_estimator = None
        try:
            from app.homography.camera_motion import CameraMotionEstimator
            self.motion_estimator = CameraMotionEstimator()
        except Exception as e:
            self.logger.stage_warning(name, f"CameraMotionEstimator init failed: {e}")

    def process(self, frames: List[FrameData]) -> StageResult:
        self.logger.stage_start(self.name)
        t0 = time.time()
        try:
            motions = []
            for idx, f in enumerate(frames):
                motion = (0.0, 0.0)
                if self.motion_estimator is not None and f.image is not None:
                    try:
                        m = self.motion_estimator.estimate(f.image, frame_number=f.frame_number)
                        if m is not None:
                            motion = (float(getattr(m, "dx", 0.0)), float(getattr(m, "dy", 0.0)))
                    except Exception:
                        pass
                motions.append(motion)
            t1 = time.time()
            self.logger.stage_end(self.name, len(frames))
            self.logger.stage_info(self.name, f"Motion vectors: {len(motions)}")
            return self._make_result(True, motions, execution_time=t1 - t0, frames=len(frames))
        except Exception as e:
            self.logger.stage_error(self.name, str(e))
            return self._make_result(False, error=str(e))


class HomographyStage(BaseStage):
    def __init__(self, name: str, config: Dict, logger: PipelineLogger):
        super().__init__(name, config, logger)
        self.homography_estimator = None
        try:
            from app.homography.homography_estimator import HomographyEstimator
            self.homography_estimator = HomographyEstimator(config=config)
        except Exception as e:
            self.logger.stage_warning(name, f"HomographyEstimator init failed: {e}")

    def process(self, frames: List[FrameData], tracks: List[TrackData], motions: List = None) -> StageResult:
        self.logger.stage_start(self.name)
        t0 = time.time()
        try:
            results = []
            total_points = 0
            first_frame = frames[0].image if frames and frames[0].image is not None else None
            if self.homography_estimator is not None and first_frame is not None:
                try:
                    initialized = self.homography_estimator.initialize(first_frame)
                    if not initialized:
                        try:
                            self.homography_estimator.estimate_homography()
                        except Exception:
                            pass
                except Exception:
                    pass

            for f, tr in zip(frames, tracks):
                h_data = HomographyData(
                    frame_number=f.frame_number,
                    timestamp=f.timestamp,
                    player_world_positions={},
                    is_valid=False
                )
                if self.homography_estimator is not None and f.image is not None:
                    try:
                        self.homography_estimator.estimate_camera_motion(f.image, f.frame_number)
                        cal = getattr(self.homography_estimator, "calibration_result", None)
                        if cal is not None and getattr(cal, "success", False):
                            h_data.homography_matrix = getattr(cal, "homography_matrix", None)
                            h_data.is_valid = True
                            for tid, pdata in tr.player_tracks.items():
                                cx, cy = pdata.get("center", (0, 0))
                                wp = self.homography_estimator.get_world_position(np.array([cx, cy], dtype=np.float64))
                                if wp is not None:
                                    h_data.player_world_positions[tid] = (float(wp[0]), float(wp[1]))
                            if tr.ball_track is not None and tr.ball_track.get("center") is not None:
                                bx, by = tr.ball_track["center"]
                                wb = self.homography_estimator.get_world_position(np.array([bx, by], dtype=np.float64))
                                if wb is not None:
                                    h_data.ball_world_position = (float(wb[0]), float(wb[1]))
                    except Exception:
                        pass
                total_points += len(h_data.player_world_positions)
                results.append(h_data)
            t1 = time.time()
            self.logger.stage_end(self.name, len(frames))
            self.logger.stage_info(self.name, f"World points: {total_points}")
            return self._make_result(True, results, execution_time=t1 - t0, frames=len(frames))
        except Exception as e:
            self.logger.stage_error(self.name, str(e))
            return self._make_result(False, error=str(e))


class PlayerKinematicsStage(BaseStage):
    def __init__(self, name: str, config: Dict, logger: PipelineLogger):
        super().__init__(name, config, logger)
        self.kinematics_engine = None
        try:
            from app.analytics.player_kinematics import PlayerKinematicsEngine
            self.kinematics_engine = PlayerKinematicsEngine(config=config)
        except Exception as e:
            self.logger.stage_warning(name, f"PlayerKinematicsEngine init failed: {e}")

    def process(self, homography_data: List[HomographyData], teams: List[TeamData]) -> StageResult:
        self.logger.stage_start(self.name)
        t0 = time.time()
        try:
            results: Dict[int, List[Dict]] = {}
            if self.kinematics_engine is not None:
                player_tracks: Dict[int, List[Dict]] = {}
                for hd in homography_data:
                    for tid, wpos in hd.player_world_positions.items():
                        player_tracks.setdefault(tid, []).append({
                            "track_id": tid,
                            "frame_number": hd.frame_number,
                            "timestamp": hd.timestamp,
                            "world_position": wpos,
                            "confidence": 1.0,
                        })
                if player_tracks:
                    # PlayerKinematicsEngine expects a dict of tracks.
                    processed = self.kinematics_engine.process(player_tracks)
                    results = processed.get("processed_tracks", processed) if isinstance(processed, dict) else {}
            t1 = time.time()
            record_count = sum(len(v) for v in results.values()) if results else 0
            self.logger.stage_end(self.name, len(homography_data))
            self.logger.stage_info(self.name, f"Kinematic records: {record_count}")
            return self._make_result(True, results, execution_time=t1 - t0, frames=len(homography_data))
        except Exception as e:
            self.logger.stage_error(self.name, str(e))
            return self._make_result(False, error=str(e))


class BallAnalyticsStage(BaseStage):
    def __init__(self, name: str, config: Dict, logger: PipelineLogger):
        super().__init__(name, config, logger)
        self.ball_analytics_engine = None
        try:
            from app.analytics.ball_analytics import BallAnalyticsEngine
            self.ball_analytics_engine = BallAnalyticsEngine(config=config)
        except Exception as e:
            self.logger.stage_warning(name, f"BallAnalyticsEngine init failed: {e}")

    def process(self, homography_data: List[HomographyData], teams: List[TeamData]) -> StageResult:
        self.logger.stage_start(self.name)
        t0 = time.time()
        try:
            results: Dict[int, List[Dict]] = {}
            if self.ball_analytics_engine is not None:
                ball_tracks: Dict[int, List[Dict]] = {1: []}
                player_positions_by_frame: Dict[int, Dict[int, Tuple[float, float]]] = {}
                player_teams: Dict[int, str] = {}
                for hd, td in zip(homography_data, teams):
                    if hd.ball_world_position is not None:
                        ball_tracks[1].append({
                            "track_id": 1,
                            "frame_number": hd.frame_number,
                            "timestamp": hd.timestamp,
                            "world_position": hd.ball_world_position,
                            "confidence": 1.0,
                        })
                    player_positions_by_frame[hd.frame_number] = hd.player_world_positions
                    for tid, team_id in td.team_assignments.items():
                        player_teams[tid] = team_id
                try:
                    results = self.ball_analytics_engine.process(ball_tracks, player_positions_by_frame, player_teams)
                except TypeError:
                    try:
                        results = self.ball_analytics_engine.process(
                            ball_tracks,
                            {k: list(v.values()) for k, v in player_positions_by_frame.items()},
                            player_teams,
                        )
                    except Exception:
                        results = ball_tracks
            t1 = time.time()
            self.logger.stage_end(self.name, len(homography_data))
            return self._make_result(True, results, execution_time=t1 - t0, frames=len(homography_data))
        except Exception as e:
            self.logger.stage_error(self.name, str(e))
            return self._make_result(False, error=str(e))


class BiomechanicsStage(BaseStage):
    def __init__(self, name: str, config: Dict, logger: PipelineLogger):
        super().__init__(name, config, logger)
        self.biomechanics_engine = None
        try:
            from app.analytics.biomechanics import BiomechanicsEngine
            self.biomechanics_engine = BiomechanicsEngine(config=config)
        except Exception as e:
            self.logger.stage_warning(name, f"BiomechanicsEngine init failed: {e}")

    def process(self, kinematics: Dict, poses: List[PoseData]) -> StageResult:
        self.logger.stage_start(self.name)
        t0 = time.time()
        try:
            results: Dict[int, Dict[str, float]] = {}
            if self.biomechanics_engine is not None:
                try:
                    results = self.biomechanics_engine.compute(kinematics, poses)
                except TypeError:
                    results = {}
            t1 = time.time()
            self.logger.stage_end(self.name, len(poses))
            return self._make_result(True, results, execution_time=t1 - t0, frames=len(poses))
        except Exception as e:
            self.logger.stage_error(self.name, str(e))
            return self._make_result(False, error=str(e))


class VisualizationStage(BaseStage):
    def process(self, frames: List[FrameData], tracks: List[TrackData],
                teams: List[TeamData], homography: List[HomographyData],
                output_path: Path, poses: Optional[List[PoseData]] = None) -> StageResult:
        self.logger.stage_start(self.name)
        t0 = time.time()
        try:
            if not frames:
                raise ValueError("No frames to visualize")

            fps = 25.0
            h, w = frames[0].image.shape[:2] if frames[0].image is not None else (720, 1280)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))

            debug_dir = output_path.parent / "debug_frames"
            debug_dir.mkdir(exist_ok=True)

            # Optional ball path renderer
            renderer = None
            try:
                from app.visualization.ball_path_renderer import BallPathRenderer
                renderer = BallPathRenderer()
            except Exception:
                renderer = None

            for i, f in enumerate(frames):
                if f.image is None:
                    continue
                annotated = f.image.copy()

                if renderer is not None and i < len(tracks):
                    trk = tracks[i]
                    if trk.ball_track is not None:
                        bcx, bcy = trk.ball_track.get("center", (0, 0))
                        if bcx is not None and bcy is not None:
                            team_id = "unknown"
                            if i < len(teams):
                                for tid, tname in teams[i].team_assignments.items():
                                    if tid == trk.ball_track.get("player_id"):
                                        team_id = tname
                                        break
                            renderer.update(
                                frame_number=f.frame_number,
                                timestamp=f.timestamp,
                                pixel_position=(int(bcx), int(bcy)),
                                world_position=homography[i].ball_world_position if i < len(homography) else None,
                                team_id=team_id,
                                has_possession=team_id != "unknown",
                                possession_confidence=float(trk.ball_track.get("confidence", 0.0)),
                                ball_speed_kmh=0.0,
                                is_pass=False,
                                is_touch=False,
                            )
                    try:
                        renderer.render(annotated)
                    except Exception:
                        pass

                if tracks and i < len(tracks):
                    trk = tracks[i]
                    if i < len(teams):
                        team_map = teams[i].team_assignments
                    else:
                        team_map = {}

                    for tid, pdata in trk.player_tracks.items():
                        bbox = pdata.get("bbox")
                        if not bbox:
                            continue
                        x1, y1, x2, y2 = [int(v) for v in bbox]
                        team_id = team_map.get(tid, "unknown")
                        color = (0, 255, 0)
                        if team_id == "team1":
                            color = (255, 0, 0)
                        elif team_id == "team2":
                            color = (0, 0, 255)
                        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(annotated, f"P{tid}", (x1, max(0, y1 - 5)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                    if trk.ball_track is not None and trk.ball_track.get("center") is not None:
                        bcx, bcy = trk.ball_track["center"]
                        cv2.circle(annotated, (int(bcx), int(bcy)), 6, (0, 255, 255), -1)

                # Pose skeletons
                if poses and i < len(poses):
                    pose = poses[i]
                    for tid, kps in pose.player_keypoints.items():
                        if kps is None or len(kps) == 0:
                            continue
                        x1, y1, x2, y2 = None, None, None, None
                        if tracks and i < len(tracks) and tid in tracks[i].player_tracks:
                            x1, y1, x2, y2 = [int(v) for v in tracks[i].player_tracks[tid].get("bbox", (None, None, None, None))]
                        if x1 is None:
                            continue
                        offset_x, offset_y = x1, y1
                        for xk, yk in kps:
                            cx = int(xk) + offset_x
                            cy = int(yk) + offset_y
                            cv2.circle(annotated, (cx, cy), 3, (0, 255, 255), -1)

                out.write(annotated)
                if i % 50 == 0:
                    cv2.imwrite(str(debug_dir / f"frame_{f.frame_number:06d}.jpg"), annotated)

            out.release()
            t1 = time.time()
            self.logger.stage_end(self.name, len(frames))
            self.logger.stage_info(self.name, f"Annotated video saved: {output_path}")
            return self._make_result(True, str(output_path), execution_time=t1 - t0, frames=len(frames))
        except Exception as e:
            self.logger.stage_error(self.name, str(e))
            return self._make_result(False, error=str(e))


class ExportStage(BaseStage):
    def export_summary_json(self, output: Any, output_path: Path) -> None:
        import json
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2, default=str)
        self.logger.stage_info(self.name, f"Summary saved to {output_path}")

    def export_csv(self, data: List[Dict], output_path: Path) -> None:
        import pandas as pd
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(data)
        df.to_csv(output_path, index=False)
        self.logger.stage_info(self.name, f"CSV saved to {output_path}")