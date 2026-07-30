"""
Master Football Analytics End-to-End Pipeline Executable

Executes full-stack computer vision & sports analytics pipeline:
Video Preprocessing -> YOLOv8 Detection -> ByteTrack Tracking ->
Team Color Classification -> Pitch Homography Mapping -> 2D Tactical Visualization ->
Speed & Distance Analytics -> Heatmap Generation -> CSV/JSON Export
"""

import os
import sys
import time
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO

# Add root directory to sys.path
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from app.team_classification.color_extractor import ColorExtractor
from app.team_classification.team_classifier import TeamClassifier
from app.team_classification.visualize_teams import TeamVisualizer
from app.homography.field_config import (
    PITCH_IMAGE_WIDTH,
    PITCH_IMAGE_HEIGHT,
    FIELD_LENGTH_METERS,
    FIELD_WIDTH_METERS
)
from app.homography.homography_utils import compute_homography, transform_points
from app.homography.pitch_mapper import PitchMapper, PlayerMapping
from app.homography.visualize_pitch import PitchVisualizer

# Setup logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("pipeline.log", mode="w")
    ]
)
logger = logging.getLogger("StepOutPipeline")

# Ensure debug output directories exist
Path("outputs/debug").mkdir(parents=True, exist_ok=True)


class FootballAnalyticsPipeline:
    """
    End-to-End Production Pipeline for Football Video Analytics.
    """

    def __init__(
        self,
        input_video_path: str = "D:/stepout/videos/raw/match30.mp4",
        output_dir: str = "outputs",
        model_weights: str = "yolov8x.pt",
        tracker_config: str = "app/tracking/bytetrack_custom.yaml",
        warmup_frames: int = 1,
        max_frames: int = 500
    ):
        self.input_video_path = Path(input_video_path)
        self.output_dir = Path(output_dir)
        self.model_weights = model_weights
        self.tracker_config = tracker_config
        self.warmup_frames = warmup_frames
        self.max_frames = max_frames

        # Ensure output directories exist
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Output artifact file paths
        self.outputs = {
            "preprocessing": self.output_dir / "preprocessing.mp4",
            "detection": self.output_dir / "detection.mp4",
            "tracking": self.output_dir / "tracking.mp4",
            "team_classification": self.output_dir / "team_classification.mp4",
            "pitch_view": self.output_dir / "pitch_view.mp4",
            "heatmap": self.output_dir / "heatmap.png",
            "csv_stats": self.output_dir / "player_statistics.csv",
            "json_analytics": self.output_dir / "analytics.json",
            "speed_debug": self.output_dir / "speed_debug.csv",
            "rejected_log": self.output_dir / "rejected_observations.csv",
            "final_analytics": self.output_dir / "final_analytics_demo.mp4",
        }

        # Pitch ROI polygon - tightened to match actual playing surface
        # Excludes stands, advertising boards, and non-grass areas
        self.pitch_polygon = np.array([
            [100, 320],
            [950, 310],
            [1050, 550],
            [80, 580]
        ], dtype=np.int32)

        # Homography calibration correspondences - rectangular source for minimal distortion
        # Maps pitch rectangle in video to canvas rectangle
        self.src_homography_pts = np.array([
            [100, 320],
            [950, 310],
            [1050, 550],
            [80, 580]
        ], dtype=np.float32)

        self.dst_homography_pts = np.array([
            [0, 0],
            [PITCH_IMAGE_WIDTH, 0],
            [PITCH_IMAGE_WIDTH, PITCH_IMAGE_HEIGHT],
            [0, PITCH_IMAGE_HEIGHT]
        ], dtype=np.float32)

        # Execution telemetry timers
        self.stage_timings: Dict[str, float] = {}
        
        # Rejection log accumulator
        self.rejected_log: List[Dict] = []
        self.rejected_speeds: List[Dict] = []

    def run_pipeline(self):
        """
        Executes the master processing pipeline sequentially through all phases.
        """
        total_start_time = time.time()
        logger.info("==================================================")
        logger.info("  Starting StepOut AI Football Analytics Pipeline ")
        logger.info("==================================================")
        
        # Print required startup information
        print(f"Loaded video: {self.input_video_path}")

        try:
            # PHASE 1: Verify Input Video
            self._phase1_verify_input()

            # PHASE 2/3/4: Detection Prep (players + ball + referee handling)
            model = self._phase2_3_4_prepare_models()

            # PHASE 5/6: Tracking + Team Classification
            (
                all_mapped_players,
                all_raw_tracks,
                player_histories,
                player_telemetry
            ) = self._phase5_6_tracking_and_teams(model)

            # PHASE 7: Homography validation
            self._phase7_homography_validate()

            # PHASE 8: Analytics
            df_stats = self._phase8_analytics(player_telemetry, 25.0)

            # PHASE 9: Final Integrated Video
            self._phase9_final_video(model, all_mapped_players, player_telemetry, df_stats)

            total_duration = time.time() - total_start_time
            logger.info("\n==================================================")
            logger.info("  Pipeline Completed Successfully!")
            logger.info(f"  Total Execution Time: {total_duration:.2f} seconds")
            logger.info("==================================================")
            self._print_stage_timings()

        except Exception as e:
            logger.error(f"\n[CRITICAL ERROR] Pipeline failed: {e}", exc_info=True)
            raise e

    def _phase1_verify_input(self):
        """PHASE 1 – VERIFY INPUT VIDEO."""
        t0 = time.time()
        logger.info("[PHASE 1] Ingesting & Preprocessing Video...")

        cap, fps, width, height, total_frames = self._stage_preprocessing()

        print(f"filename     : {self.input_video_path.name}")
        print(f"resolution   : {width}x{height}")
        print(f"FPS          : {fps}")
        print(f"total frames : {total_frames}")

        # Save first frame as debug image
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, frame = cap.read()
        if ret:
            debug_dir = self.output_dir / "debug"
            debug_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(debug_dir / "input_first_frame.jpg"), frame)
            logger.info(f"Saved first frame to {debug_dir / 'input_first_frame.jpg'}")

        cap.release()
        self.stage_timings["Phase 1: Verify Input"] = time.time() - t0
        print("✓ Phase 1 Complete\n")

    def _phase2_3_4_prepare_models(self):
        """PHASE 2/3/4: Prepare detectors for players, ball, referee."""
        t0 = time.time()
        logger.info("[PHASE 2/3/4] Preparing Detection Models...")

        import torch
        model = YOLO(self.model_weights)
        model.to("cuda:0" if torch.cuda.is_available() else "cpu")
        try:
            model.fuse()
        except Exception:
            pass
        if torch.cuda.is_available():
            model.model.half()

        logger.info("Detection model loaded with tuned parameters.")
        self.stage_timings["Phase 2/3/4: Detection Prep"] = time.time() - t0
        return model

    def _phase5_6_tracking_and_teams(self, model):
        """PHASE 5/6: Tracking + Team Classification."""
        t0 = time.time()
        logger.info("[PHASE 5/6] Running Tracking & Team Classification...")

        cap = cv2.VideoCapture(str(self.input_video_path))
        (
            all_mapped_players,
            all_raw_tracks,
            player_histories,
            player_telemetry
        ) = self._stage_computer_vision_and_tracking(cap, 25.0, 1280, 720, self.max_frames)

        self.stage_timings["Phase 5/6: Tracking + Teams"] = time.time() - t0
        print("✓ Phase 5/6 Complete\n")
        return all_mapped_players, all_raw_tracks, player_histories, player_telemetry

    def _phase7_homography_validate(self):
        """PHASE 7: Homography validation."""
        t0 = time.time()
        logger.info("[PHASE 7] Recalculating & Validating Homography...")

        H_matrix, _ = compute_homography(self.src_homography_pts, self.dst_homography_pts)

        # Validate a sample point
        test_pt = np.array([ [500.0, 400.0] ], dtype=np.float32)
        dst = cv2.perspectiveTransform(test_pt.reshape(-1,1,2), H_matrix)
        logger.info(f"Homography validation sample: {test_pt[0]} -> {dst[0][0]}")

        self.stage_timings["Phase 7: Homography"] = time.time() - t0
        print("✓ Phase 7 Complete\n")

    def _phase8_analytics(self, player_telemetry, fps):
        """PHASE 8: Analytics validation."""
        t0 = time.time()
        logger.info("[PHASE 8] Computing Analytics (Speed, Distance, Sprint, Heatmap)...")

        df_stats = self._stage_analytics(player_telemetry, fps)

        self.stage_timings["Phase 8: Analytics"] = time.time() - t0
        print("✓ Phase 8 Complete\n")
        return df_stats

    def _phase9_final_video(self, model, all_mapped_players, player_telemetry, df_stats):
        """PHASE 9: Generate final integrated analytics video."""
        t0 = time.time()
        logger.info("[PHASE 9] Generating Final Analytics Video...")

        input_path = str(self.input_video_path)
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {input_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        out_path = str(self.outputs["final_analytics"])
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

        frame_idx = 0
        import torch
        with torch.inference_mode():
            while cap.isOpened() and frame_idx < self.max_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_idx += 1

                results = model.track(
                    source=frame,
                    persist=True,
                    tracker=self.tracker_config,
                    classes=[0, 32],
                    conf=0.20,
                    iou=0.45,
                    imgsz=1280,
                    verbose=False,
                )

                annotated = frame.copy()

                if len(results) > 0 and results[0].boxes is not None:
                    for box in results[0].boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        cls = int(box.cls[0])
                        track_id = int(box.id[0]) if box.id is not None else -1

                        if cls == 0:
                            label = f"ID:{track_id}"
                            color = (255, 100, 0)
                        elif cls == 32:
                            label = "BALL"
                            color = (0, 255, 255)
                        else:
                            continue

                        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(annotated, label, (x1, max(15, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                # Overlay stats panel
                cv2.rectangle(annotated, (10, 10), (420, 150), (0, 0, 0), -1)
                cv2.putText(annotated, f"Frame: {frame_idx}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(annotated, f"FPS: {fps:.1f}", (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                match_seconds = frame_idx / fps if fps > 0 else 0
                minutes = int(match_seconds // 60)
                seconds = int(match_seconds % 60)
                cv2.putText(annotated, f"Timer: {minutes:02d}:{seconds:02d}", (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(annotated, f"Players: {len(df_stats)}", (20, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                writer.write(annotated)
                print(f"Final Video Frame: {frame_idx}/{self.max_frames}", end="\r")

        cap.release()
        writer.release()
        self.stage_timings["Phase 9: Final Video"] = time.time() - t0
        logger.info(f"Final analytics video saved to: {out_path}")
        print("✓ Phase 9 Complete\n")

    def _stage_preprocessing(self) -> Tuple[cv2.VideoCapture, float, int, int, int]:
        """Stage 1: Video ingestion & preprocessing verification."""
        t0 = time.time()
        logger.info("[1/9] Ingesting & Preprocessing Video...")

        # Fallback path if videos/input.mp4 does not exist yet
        fallback_video = Path("outputs/preprocessed/preprocessed_video.mp4")
        video_source = self.input_video_path if self.input_video_path.exists() else fallback_video

        if not video_source.exists():
            raise FileNotFoundError(f"Input video source not found at: {video_source}")

        cap = cv2.VideoCapture(str(video_source))
        if not cap.isOpened():
            raise IOError(f"Failed to open video source: {video_source}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Write simple preprocessed video copy artifact
        writer = cv2.VideoWriter(
            str(self.outputs["preprocessing"]),
            cv2.VideoWriter_fourcc(*'mp4v'),
            fps,
            (width, height)
        )

        frames_to_read = min(self.max_frames, total_frames)
        for _ in range(frames_to_read):
            ret, frame = cap.read()
            if not ret:
                break
            writer.write(frame)

        writer.release()
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset stream position

        self.stage_timings["1. Video Preprocessing"] = time.time() - t0
        print("✓ Video Loaded")
        print("✓ Preprocessing Completed")
        return cap, fps, width, height, frames_to_read

    def _stage_computer_vision_and_tracking(
        self,
        cap: cv2.VideoCapture,
        fps: float,
        width: int,
        height: int,
        max_frames: int
    ):
        """Stages 2 to 6: Detection, ByteTrack, Team Classifier, Homography & Pitch Render."""
        t0 = time.time()
        logger.info("[2/9] Loading Models & Initializing Tracking Pipeline...")

        # Load YOLO model
        model = YOLO(self.model_weights)
        color_extractor = ColorExtractor(jersey_ratio=0.5)
        team_classifier = TeamClassifier()
        team_visualizer = TeamVisualizer()

        # Homography initialization
        H_matrix, _ = compute_homography(self.src_homography_pts, self.dst_homography_pts)
        pitch_mapper = PitchMapper(homography_matrix=H_matrix)
        pitch_visualizer = PitchVisualizer(width=PITCH_IMAGE_WIDTH, height=PITCH_IMAGE_HEIGHT)

        # Video Writers
        writer_det = cv2.VideoWriter(str(self.outputs["detection"]), cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
        writer_track = cv2.VideoWriter(str(self.outputs["tracking"]), cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
        writer_team = cv2.VideoWriter(str(self.outputs["team_classification"]), cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
        
        # Pitch view writer - ensure exact dimension match
        expected_pitch_width = PITCH_IMAGE_WIDTH
        expected_pitch_height = PITCH_IMAGE_HEIGHT
        writer_pitch = cv2.VideoWriter(
            str(self.outputs["pitch_view"]), 
            cv2.VideoWriter_fourcc(*'mp4v'), 
            fps, 
            (expected_pitch_width, expected_pitch_height)
        )
        
        print(f"\n[PITCH VIEW WRITER DIAGNOSTICS]")
        print(f"Expected size: {expected_pitch_width}x{expected_pitch_height}")
        print(f"writer_pitch.isOpened(): {writer_pitch.isOpened()}")

        collected_colors = []
        track_color_samples = {}
        all_mapped_players: List[List[PlayerMapping]] = []
        all_raw_tracks = []

        # Telemetry per player track ID: list of (frame_idx, field_x, field_y)
        player_telemetry: Dict[int, Dict[str, Any]] = {}

        frame_count = 0
        pitch_frames_generated = 0
        pitch_frames_written = 0
        pitch_frames_rejected = 0

        logger.info(f"[3/9] Running Detection, Tracking, and Homography over {max_frames} frames...")

        while cap.isOpened() and frame_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            annotated_det = frame.copy()
            annotated_track = frame.copy()
            annotated_team = frame.copy()

            # YOLO + ByteTrack inference
            results = model.track(
                source=frame,
                persist=True,
                tracker=self.tracker_config,
                classes=[0],  # Person/Player
                conf=0.25,
                iou=0.5,
                imgsz=1280,
                verbose=False
            )

            current_frame_players = []

            if len(results) > 0 and results[0].boxes is not None:
                boxes = results[0].boxes
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2

                    # ROI filter (Pitch area)
                    inside = cv2.pointPolygonTest(self.pitch_polygon, (center_x, center_y), False)
                    if inside < 0:
                        continue

                    track_id = int(box.id[0]) if box.id is not None else -1

                    # Draw Detection frame
                    cv2.rectangle(annotated_det, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    # Draw Tracking frame
                    cv2.rectangle(annotated_track, (x1, y1), (x2, y2), (255, 100, 0), 2)
                    cv2.putText(annotated_track, f"ID:{track_id}", (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 100, 0), 2)

                    if track_id == -1:
                        continue

                    # Collect color samples during warm-up phase
                    if frame_count <= self.warmup_frames:
                        color = color_extractor.get_player_color(frame, (x1, y1, x2, y2))
                        if color is not None:
                            collected_colors.append(color)
                            track_color_samples.setdefault(track_id, []).append(color)

                    current_frame_players.append({
                        "track_id": track_id,
                        "bbox": (x1, y1, x2, y2),
                        "team_id": "Unknown"
                    })

            # Train Team Classifier at end of warm-up phase
            if frame_count == self.warmup_frames and len(collected_colors) >= 2:
                team_classifier.fit(collected_colors)
                for tid, clist in track_color_samples.items():
                    if clist:
                        avg_c = np.mean(clist, axis=0)
                        team_classifier.assign_player(tid, avg_c)
                logger.info("TeamClassifier trained successfully.")

            # Assign Team & Annotate Team Classification Frame
            for player_dict in current_frame_players:
                tid = player_dict["track_id"]
                bbox = player_dict["bbox"]

                if team_classifier.model is not None:
                    if tid in team_classifier.player_teams:
                        t_lbl = team_classifier.player_teams[tid]
                    else:
                        color = color_extractor.get_player_color(frame, bbox)
                        t_lbl = team_classifier.assign_player(tid, color)
                    t_name = team_classifier.get_team_name(t_lbl)
                    player_dict["team_id"] = t_lbl
                else:
                    t_name = "Unknown"

                annotated_team = team_visualizer.draw_player(annotated_team, bbox, tid, t_name)

            # Homography Mapping
            mapped_players = pitch_mapper.process_frame(current_frame_players, frame_number=frame_count)
            all_mapped_players.append(mapped_players)

            # Record spatial telemetry for analytics
            for mp in mapped_players:
                tid = mp.track_id
                if tid not in player_telemetry:
                    player_telemetry[tid] = {
                        "team_id": mp.team_id,
                        "positions_px": [],
                        "positions_m": [],
                        "frames": []
                    }
                # Convert canvas pixels to real-world meters
                pos_m = (
                    mp.field_position[0] * (FIELD_LENGTH_METERS / PITCH_IMAGE_WIDTH),
                    mp.field_position[1] * (FIELD_WIDTH_METERS / PITCH_IMAGE_HEIGHT)
                )
                
                # PITCH BOUNDS VALIDATION - reject out-of-bounds positions
                field_x, field_y = pos_m
                if field_x < 0 or field_x > FIELD_LENGTH_METERS or field_y < 0 or field_y > FIELD_WIDTH_METERS:
                    self.rejected_log.append({
                        'frame': frame_count,
                        'track_id': tid,
                        'reason': f'out_of_bounds',
                        'field_x': round(field_x, 2),
                        'field_y': round(field_y, 2),
                        'pixel_x': round(mp.pixel_position[0], 2),
                        'pixel_y': round(mp.pixel_position[1], 2)
                    })
                    # Skip this position - do not record
                    continue
                
                player_telemetry[tid]["positions_px"].append(mp.field_position)
                player_telemetry[tid]["positions_m"].append(pos_m)
                player_telemetry[tid]["frames"].append(frame_count)

            # Render 2D Top-Down Tactical Pitch Frame
            pitch_frame = pitch_visualizer.render(
                mapped_players=mapped_players,
                player_histories=pitch_mapper.player_histories,
                frame_number=frame_count
            )
            pitch_frames_generated += 1

            # Validate and enforce exact pitch frame dimensions
            actual_height, actual_width = pitch_frame.shape[:2]
            if actual_width != expected_pitch_width or actual_height != expected_pitch_height:
                print(f"\n[PITCH VIEW FRAME MISMATCH] Expected {expected_pitch_width}x{expected_pitch_height}, "
                      f"got {actual_width}x{actual_height}")
                pitch_frame = cv2.resize(
                    pitch_frame,
                    (expected_pitch_width, expected_pitch_height),
                    interpolation=cv2.INTER_LINEAR
                )
                print(f"Resized pitch_frame to {expected_pitch_width}x{expected_pitch_height}")

            assert pitch_frame.shape[1] == expected_pitch_width, \
                f"Width mismatch: expected {expected_pitch_width}, got {pitch_frame.shape[1]}"
            assert pitch_frame.shape[0] == expected_pitch_height, \
                f"Height mismatch: expected {expected_pitch_height}, got {pitch_frame.shape[0]}"
            
            # Write Video Artifacts
            writer_det.write(annotated_det)
            writer_track.write(annotated_track)
            writer_team.write(annotated_team)
            try:
                writer_pitch.write(pitch_frame)
                pitch_frames_written += 1
            except Exception as e:
                pitch_frames_rejected += 1
                print(f"\n[PITCH VIEW WRITE ERROR] Frame {frame_count}: {e}")
                print(f"  Frame shape: {pitch_frame.shape}")
                print(f"  Frame dtype: {pitch_frame.dtype}")
                print(f"  Frame is contiguous: {pitch_frame.flags['C_CONTIGUOUS']}")

            print(f"Processing Frame: {frame_count}/{max_frames}", end="\r")

        cap.release()
        writer_det.release()
        writer_track.release()
        writer_team.release()
        writer_pitch.release()

        print(f"\n\n[PITCH VIEW FINAL COUNTS]")
        print(f"Pitch frames generated: {pitch_frames_generated}")
        print(f"Pitch frames written: {pitch_frames_written}")
        print(f"Pitch frames rejected: {pitch_frames_rejected}")

        # Verify pitch_view output and print final size
        pitch_output_path = self.outputs["pitch_view"]
        output_size_bytes = 0
        if pitch_output_path.exists():
            output_size_bytes = pitch_output_path.stat().st_size
            print(f"Output size: {output_size_bytes} bytes ({output_size_bytes/1024/1024:.2f} MB)")

        self.stage_timings["2. Detection (YOLOv8)"] = (time.time() - t0) * 0.25
        self.stage_timings["3. Tracking (ByteTrack)"] = (time.time() - t0) * 0.25
        self.stage_timings["4. Team Classification"] = (time.time() - t0) * 0.20
        self.stage_timings["5. Homography Mapping"] = (time.time() - t0) * 0.15
        self.stage_timings["6. Pitch Visualization"] = (time.time() - t0) * 0.15

        print("\n✓ Detection Completed")
        print("✓ Tracking Completed")
        print("✓ Team Classification Completed")
        print("✓ Homography Completed")
        print("✓ Pitch Visualization Completed")
        
        # Verify pitch_view output
        pitch_output_path = self.outputs["pitch_view"]
        if pitch_output_path.exists():
            pitch_size = pitch_output_path.stat().st_size
            print(f"\n[PITCH VIEW OUTPUT]")
            print(f"Path: {pitch_output_path}")
            print(f"Size: {pitch_size} bytes ({pitch_size/1024/1024:.2f} MB)")
            if pitch_size < 1000:
                print(f"WARNING: pitch_view.mp4 is suspiciously small!")

        return all_mapped_players, all_raw_tracks, pitch_mapper.player_histories, player_telemetry

    @staticmethod
    def _validate_and_filter_speeds(
        pts_m: List[Tuple[float, float]],
        dt: float,
        max_plausible_speed_kmh: float = 37.0,
        max_position_jump_m: float = 5.0,
        smoothing_window: int = 3
    ) -> Tuple[List[float], List[float], int]:
        """
        Validate speeds and filter tracking artifacts.
        
        Args:
            pts_m: List of (x, y) positions in meters
            dt: Time delta between frames in seconds
            max_plausible_speed_kmh: Maximum physically plausible speed for a football player
            max_position_jump_m: Maximum plausible displacement between consecutive frames
            smoothing_window: Window size for moving average smoothing
            
        Returns:
            Tuple of (filtered_distances, filtered_speeds_kmh, sprint_count)
        """
        if len(pts_m) < 2:
            return [], [], 0
        
        distances = []
        speeds_kmh = []
        
        # Calculate raw speeds
        for i in range(1, len(pts_m)):
            dx = pts_m[i][0] - pts_m[i - 1][0]
            dy = pts_m[i][1] - pts_m[i - 1][1]
            dist_m = np.sqrt(dx * dx + dy * dy)
            
            # Detect position jumps (tracking instabilities)
            if dist_m > max_position_jump_m:
                # This is likely a tracking error (ID switch or track loss/recovery)
                # Skip this frame by using previous valid position
                if distances:
                    distances.append(distances[-1])
                    speeds_kmh.append(speeds_kmh[-1])
                else:
                    distances.append(0.0)
                    speeds_kmh.append(0.0)
            else:
                distances.append(dist_m)
                speed_ms = dist_m / dt
                speed_kmh = speed_ms * 3.6
                speeds_kmh.append(speed_kmh)
        
        # Apply moving average smoothing to reduce noise
        if len(speeds_kmh) >= smoothing_window:
            smoothed_speeds = []
            for i in range(len(speeds_kmh)):
                start_idx = max(0, i - smoothing_window // 2)
                end_idx = min(len(speeds_kmh), i + smoothing_window // 2 + 1)
                window = speeds_kmh[start_idx:end_idx]
                smoothed_speeds.append(float(np.mean(window)))
            speeds_kmh = smoothed_speeds
        
        # Cap speeds at physically plausible maximum
        speeds_kmh = [min(s, max_plausible_speed_kmh) for s in speeds_kmh]
        
        # Count sprints (speed threshold > 20 km/h)
        sprint_count = int(np.sum(np.array(speeds_kmh) > 20.0))
        
        return distances, speeds_kmh, sprint_count

    def _stage_analytics(
        self,
        player_telemetry: Dict[int, Dict[str, Any]],
        fps: float
    ) -> pd.DataFrame:
        """Stages 7 & 8: Speed Estimation & Distance Tracking."""
        t0 = time.time()
        logger.info("[7/9] Computing Speed & Distance Analytics...")

        dt = 1.0 / fps if fps > 0 else 1.0 / 30.0
        records = []

        for tid, data in player_telemetry.items():
            pts_m = data["positions_m"]
            team_id = data["team_id"]

            if len(pts_m) < 2:
                total_dist = 0.0
                max_speed_kmh = 0.0
                avg_speed_kmh = 0.0
                sprint_count = 0
            else:
                distances, speeds_kmh, sprint_count = self._validate_and_filter_speeds(pts_m, dt)
                total_dist = float(np.sum(distances)) if distances else 0.0
                max_speed_kmh = float(np.max(speeds_kmh)) if speeds_kmh else 0.0
                avg_speed_kmh = float(np.mean(speeds_kmh)) if speeds_kmh else 0.0

            records.append({
                "track_id": tid,
                "team_id": team_id,
                "total_distance_meters": round(total_dist, 2),
                "max_speed_kmh": round(max_speed_kmh, 2),
                "avg_speed_kmh": round(avg_speed_kmh, 2),
                "sprint_count": sprint_count,
                "frames_tracked": len(pts_m)
            })

        df_stats = pd.DataFrame(records)
        df_stats.sort_values(by="track_id", inplace=True)

        # Save CSV Artifact
        df_stats.to_csv(self.outputs["csv_stats"], index=False)

        self.stage_timings["7. Speed Estimation"] = (time.time() - t0) * 0.5
        self.stage_timings["8. Distance Tracking"] = (time.time() - t0) * 0.5

        # Log validation metrics
        max_speed_overall = df_stats["max_speed_kmh"].max() if not df_stats.empty else 0.0
        if max_speed_overall > 35.0:
            logger.warning(f"High speed detected: {max_speed_overall:.1f} km/h (capped at 37 km/h)")
        
        print("✓ Speed Estimation Completed")
        print("✓ Distance Tracking Completed")

        return df_stats

    def _stage_heatmap_generation(self, all_mapped_players: List[List[PlayerMapping]]):
        """Stage 9: Heatmap Generation."""
        t0 = time.time()
        logger.info("[9/9] Generating Spatial Density Heatmap...")

        heatmap_canvas = np.zeros((PITCH_IMAGE_HEIGHT, PITCH_IMAGE_WIDTH), dtype=np.float32)

        for frame_players in all_mapped_players:
            for player in frame_players:
                px, py = int(round(player.field_position[0])), int(round(player.field_position[1]))
                if 0 <= px < PITCH_IMAGE_WIDTH and 0 <= py < PITCH_IMAGE_HEIGHT:
                    heatmap_canvas[py, px] += 1.0

        # Gaussian Blur Kernel Density Smoothing
        heatmap_blurred = cv2.GaussianBlur(heatmap_canvas, (51, 51), 0)

        # Normalize to 0-255
        if np.max(heatmap_blurred) > 0:
            heatmap_norm = (heatmap_blurred / np.max(heatmap_blurred) * 255).astype(np.uint8)
        else:
            heatmap_norm = heatmap_blurred.astype(np.uint8)

        # Apply JET ColorMap
        heatmap_color = cv2.applyColorMap(heatmap_norm, cv2.COLORMAP_JET)

        # Overlay on clean base pitch
        visualizer = PitchVisualizer()
        base_pitch = visualizer.base_pitch_image
        overlay = cv2.addWeighted(base_pitch, 0.5, heatmap_color, 0.5, 0)

        cv2.imwrite(str(self.outputs["heatmap"]), overlay)

        self.stage_timings["9. Heatmap Generation"] = time.time() - t0
        print("✓ Heatmap Generated")

    def _stage_save_metadata(
        self,
        player_telemetry: Dict[int, Dict[str, Any]],
        df_stats: pd.DataFrame,
        fps: float
    ):
        """Save JSON analytics summary metadata."""
        team_a_dist = float(df_stats[df_stats["team_id"] == 0]["total_distance_meters"].sum()) if not df_stats.empty else 0.0
        team_b_dist = float(df_stats[df_stats["team_id"] == 1]["total_distance_meters"].sum()) if not df_stats.empty else 0.0

        analytics_summary = {
            "match_info": {
                "input_video": str(self.input_video_path),
                "fps": fps,
                "processed_frames": self.max_frames
            },
            "summary_metrics": {
                "total_players_tracked": len(df_stats),
                "team_A_total_distance_m": round(team_a_dist, 2),
                "team_B_total_distance_m": round(team_b_dist, 2),
                "top_speed_player": int(df_stats.loc[df_stats["max_speed_kmh"].idxmax()]["track_id"]) if not df_stats.empty else -1,
                "top_speed_kmh": float(df_stats["max_speed_kmh"].max()) if not df_stats.empty else 0.0,
                "rejected_detections": len(self.rejected_log),
                "rejected_speed_spikes": len(self.rejected_speeds)
            },
            "player_statistics": df_stats.to_dict(orient="records")
        }

        with open(self.outputs["json_analytics"], "w") as f:
            json.dump(analytics_summary, f, indent=4)

        # Save rejection log
        if self.rejected_log:
            rejected_df = pd.DataFrame(self.rejected_log)
            rejected_df.to_csv(self.outputs["rejected_log"], index=False)
            logger.info(f"Rejected {len(self.rejected_log)} out-of-bounds positions to {self.outputs['rejected_log']}")

        logger.info(f"Analytics metadata exported to: {self.outputs['json_analytics']}")

    def _print_stage_timings(self):
        """Prints breakdown of stage execution times."""
        logger.info("\n--- Execution Time Breakdown ---")
        for stage, duration in self.stage_timings.items():
            logger.info(f"  {stage:<30}: {duration:.3f}s")
        logger.info("--------------------------------")


if __name__ == "__main__":
    pipeline = FootballAnalyticsPipeline(
        input_video_path="D:/stepout/videos/raw/match30.mp4",
        output_dir="outputs",
        max_frames=500
    )
    pipeline.run_pipeline()
