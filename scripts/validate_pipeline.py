"""
Complete Production Validation of SPORTA VISTA PRO CV Pipeline

Validates every stage independently and together.
Generates comprehensive reports and debug outputs.
"""

import csv
import json
import logging
import math
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.core.config import get_config
from app.detection.detection_filter import parse_yolo_results, DetectionFilter
from app.preprocessing.adaptive_preprocessor import AdaptivePreprocessor
from app.tracking.ball_tracker import BallTracker
from app.tracking.ball_tracking_metrics import BallTrackingMetricsWriter
from app.team_classification.jersey_classifier import JerseyClassifier
from app.team_classification.team_metrics import TeamMetricsCollector
from app.utils.roi_loader import load_pitch_roi_as_numpy

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

INPUT_VIDEO = ROOT / "videos" / "raw" / "match30.mp4"
OUTPUT_DIR = ROOT / "outputs" / "validation"
FAILURES_DIR = OUTPUT_DIR / "failures"
MODEL_WEIGHTS = ROOT / "yolov8x.pt"
TRACKER_CONFIG = ROOT / "app" / "tracking" / "bytetrack_custom.yaml"
PITCH_ROI, _ = load_pitch_roi_as_numpy(ROOT, verbose=True)


class PipelineValidator:
    """Complete CV pipeline validator."""

    def __init__(self) -> None:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        FAILURES_DIR.mkdir(parents=True, exist_ok=True)

        self.preprocessor = AdaptivePreprocessor()
        self.classifier = JerseyClassifier()
        self.ball_tracker = BallTracker()
        self.ball_metrics_writer = BallTrackingMetricsWriter(OUTPUT_DIR / "ball_tracking_report.csv")
        self.team_metrics = TeamMetricsCollector(OUTPUT_DIR / "classification_report.csv")
        self.detection_filter = DetectionFilter(pitch_roi=PITCH_ROI)

        self.preprocessing_rows = []
        self.detection_rows = []
        self.ball_rows = []
        self.classification_rows = []
        self.temporal_rows = []
        self.team_switches = []
        self.prev_teams = {}
        self.track_presence = defaultdict(list)
        self.possible_switches = []
        self.prev_lost = {}

        self.perf = {
            "preprocessing": [],
            "detection": [],
            "tracking": [],
            "ball_tracking": [],
            "classification": [],
        }

    def validate(self, max_frames: int = 0) -> Dict:
        logger.info("Starting complete pipeline validation...")

        cap = cv2.VideoCapture(str(INPUT_VIDEO))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open {INPUT_VIDEO}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if max_frames > 0:
            total = min(total, max_frames)

        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        model = YOLO(str(MODEL_WEIGHTS))
        model.to(device)
        try:
            model.fuse()
        except Exception:
            pass
        model.model.half()

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(OUTPUT_DIR / "validation_debug.mp4"), fourcc, fps, (width, height))

        frame_no = 0
        with torch.inference_mode():
            while frame_no < total:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_no += 1

                t0 = time.perf_counter()
                m = self.preprocessor.measure(frame, frame_no)
                pre, steps = self.preprocessor.apply(frame, m)
                t1 = time.perf_counter()
                self.perf["preprocessing"].append(t1 - t0)

                results = model.track(
                    source=pre,
                    persist=True,
                    tracker=str(TRACKER_CONFIG),
                    classes=[0, 32],
                    conf=0.25,
                    iou=0.55,
                    verbose=False,
                    device=device,
                )
                t2 = time.perf_counter()
                self.perf["detection"].append(t2 - t1)

                dets = parse_yolo_results(results)
                players, rejected, balls = self.detection_filter.split(dets)
                tracked_players = [d for d in players if d.track_id >= 0]
                t3 = time.perf_counter()
                self.perf["tracking"].append(t3 - t2)

                # Use all ball detections (class 32) regardless of filter
                all_balls = [d for d in dets if d.cls_id == 32]
                ball_result = self.ball_tracker.update([{
                    "center": d.center,
                    "bbox": d.bbox,
                    "confidence": d.conf,
                } for d in all_balls], frame_no)
                t4 = time.perf_counter()
                self.perf["ball_tracking"].append(t4 - t3)

                # Call fit_if_ready periodically (matches cv_quality_pipeline behavior)
                if frame_no in (45, 90, 150) or (frame_no > 150 and frame_no % 75 == 0):
                    self.classifier.fit_if_ready()

                debug_frame = frame.copy()
                for d in tracked_players:
                    label, tconf = self.classifier.classify(d.track_id, frame, d.bbox)
                    t5 = time.perf_counter()
                    self.perf["classification"].append(t5 - t4)

                    self.team_metrics.record(frame_no, d.track_id, label, tconf, d.bbox[3] - d.bbox[1], tconf)
                    self._draw_player(debug_frame, d, label, tconf)

                    if d.track_id in self.prev_teams and self.prev_teams[d.track_id] != label and label != "Unknown":
                        self.team_switches.append({
                            "track_id": d.track_id,
                            "frame": frame_no,
                            "previous": self.prev_teams[d.track_id],
                            "new": label,
                        })
                    self.prev_teams[d.track_id] = label

                # Track presence and ID switching
                active_ids = set()
                for d in tracked_players:
                    active_ids.add(d.track_id)
                    self.track_presence[d.track_id].append(frame_no)

                for d in tracked_players:
                    if d.track_id not in self.prev_lost:
                        for old_tid, (lost_frame, old_center) in list(self.prev_lost.items()):
                            if frame_no - lost_frame <= 20 and math.hypot(
                                d.center[0] - old_center[0],
                                d.center[1] - old_center[1]
                            ) < 45:
                                self.possible_switches.append({
                                    "frame": frame_no, "old_track": old_tid, "new_track": d.track_id
                                })
                                break

                for tid in list(self.prev_lost.keys()):
                    if tid not in active_ids:
                        del self.prev_lost[tid]
                for tid, frames in list(self.track_presence.items()):
                    if frames and frames[-1] < frame_no and frame_no - frames[-1] == 1:
                        self.prev_lost[tid] = (frame_no, (0, 0))

                if ball_result:
                    bc = (int(ball_result["center"][0]), int(ball_result["center"][1]))
                    cv2.circle(debug_frame, bc, 8, (0, 255, 255), 2)
                    cv2.putText(debug_frame, f"BALL {ball_result.get('confidence', 0):.2f}", (bc[0] + 10, bc[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 2)

                cv2.putText(debug_frame, f"Frame {frame_no}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                writer.write(debug_frame)

                self._record_metrics(frame_no, m, players, rejected, balls, tracked_players, ball_result)

                if frame_no % 50 == 0:
                    logger.info(f"Processed {frame_no}/{total}")

        cap.release()
        writer.release()

        self.ball_metrics_writer.flush()
        self.team_metrics.flush()
        self._write_csvs(total)

        report = self._generate_report(total, fps)
        self._write_report(report)

        logger.info("Pipeline validation complete.")
        return report

    def _draw_player(self, frame, det, label, conf):
        x1, y1, x2, y2 = det.bbox
        color = (40, 60, 255) if label == "Team A" else (255, 70, 30) if label == "Team B" else (180, 180, 180)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"{label} {conf:.2f} ID {det.track_id}", (x1, max(15, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)

    def _record_metrics(self, frame, m, players, rejected, balls, tracked, ball_result):
        self.preprocessing_rows.append({"frame": frame, **m})
        self.detection_rows.append({
            "frame": frame,
            "players_detected": len(players),
            "rejected": len(rejected),
            "balls_detected": len(balls),
        })

        if ball_result:
            self.ball_rows.append({
                "frame": frame,
                "detected": 1,
                "predicted": int(ball_result.get("is_predicted", False)),
                "confidence": ball_result.get("confidence", 0),
            })
        else:
            self.ball_rows.append({"frame": frame, "detected": 0, "predicted": 0, "confidence": 0})

        self.classification_rows.append({
            "frame": frame,
            "classified": len([d for d in tracked if self.classifier.track_team.get(d.track_id) is not None]),
            "unknown": len([d for d in tracked if self.classifier.track_team.get(d.track_id) is None]),
        })

        self.temporal_rows.append({
            "frame": frame,
            "team_switches": len(self.team_switches),
        })

    def _write_csvs(self, total_frames):
        for rows, name in [
            (self.preprocessing_rows, "preprocessing.csv"),
            (self.detection_rows, "player_detection.csv"),
            (self.ball_rows, "ball_tracking_report.csv"),
            (self.classification_rows, "classification_report.csv"),
            (self.temporal_rows, "temporal_consistency.csv"),
        ]:
            if rows:
                with (OUTPUT_DIR / name).open("w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                    writer.writeheader()
                    writer.writerows(rows)

        with (FAILURES_DIR / "team_switches.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["track_id", "frame", "previous", "new"])
            writer.writeheader()
            writer.writerows(self.team_switches)

    def _get_tracking_metrics(self, total_frames: int) -> Dict:
        lifetimes = []
        lost = 0
        recovered = 0
        for tid, frames in self.track_presence.items():
            if frames:
                lifetimes.append(len(frames))
                if frames[-1] < total_frames:
                    lost += 1
                    gaps = [b - a for a, b in zip(frames, frames[1:])]
                    recovered += sum(1 for g in gaps if g > 1)
        avg_life = sum(lifetimes) / len(lifetimes) if lifetimes else 0
        max_life = max(lifetimes) if lifetimes else 0
        return {
            "total_unique_tracks": len(self.track_presence),
            "lost_tracks": lost,
            "recovered_tracks": recovered,
            "average_track_lifetime": round(avg_life, 2),
            "longest_track_lifetime": max_life,
            "estimated_id_switches": len(self.possible_switches),
        }

    def _compute_scores(self, total_frames, fps) -> Dict:
        avg_det = np.mean([r["players_detected"] for r in self.detection_rows]) if self.detection_rows else 0
        max_det = max([r["players_detected"] for r in self.detection_rows]) if self.detection_rows else 0
        ball_cov = sum(1 for r in self.ball_rows if r["detected"] > 0) / max(total_frames, 1)

        track_metrics = self._get_tracking_metrics(total_frames)
        avg_track_life = track_metrics.get("average_track_lifetime", 0)

        class_metrics = self.team_metrics.summary()
        unknown_pct = class_metrics.get("unknown", 0) / max(class_metrics.get("total_records", 1), 1) * 100

        detection_score = min(100, avg_det / max(22, 1) * 100)
        tracking_score = max(0, 100 - track_metrics.get("estimated_id_switches", 0) * 5)
        ball_score = ball_cov * 100
        team_score = max(0, 100 - unknown_pct)

        overall = np.mean([detection_score, tracking_score, ball_score, team_score])

        return {
            "overall_score": round(overall, 1),
            "detection_score": round(detection_score, 1),
            "tracking_score": round(tracking_score, 1),
            "ball_score": round(ball_score, 1),
            "team_score": round(team_score, 1),
            "avg_players_per_frame": round(float(avg_det), 1),
            "max_players_per_frame": int(max_det),
            "ball_coverage_pct": round(ball_cov * 100, 1),
            "unknown_classification_pct": round(unknown_pct, 1),
            "avg_track_lifetime": round(avg_track_life, 1),
            "track_metrics": track_metrics,
            "fps": round(fps, 1),
        }

    def _generate_report(self, total_frames, fps) -> Dict:
        scores = self._compute_scores(total_frames, fps)

        preprocessing_time = np.mean(self.perf["preprocessing"]) * 1000 if self.perf["preprocessing"] else 0
        detection_time = np.mean(self.perf["detection"]) * 1000 if self.perf["detection"] else 0
        tracking_time = np.mean(self.perf["tracking"]) * 1000 if self.perf["tracking"] else 0
        ball_time = np.mean(self.perf["ball_tracking"]) * 1000 if self.perf["ball_tracking"] else 0
        classification_time = np.mean(self.perf["classification"]) * 1000 if self.perf["classification"] else 0
        total_latency = preprocessing_time + detection_time + tracking_time + ball_time + classification_time

        grade = "NOT READY"
        if scores["overall_score"] >= 85:
            grade = "A+"
        elif scores["overall_score"] >= 75:
            grade = "A"
        elif scores["overall_score"] >= 65:
            grade = "B"
        elif scores["overall_score"] >= 50:
            grade = "C"
        elif scores["overall_score"] >= 35:
            grade = "D"

        issues = []
        if scores["ball_coverage_pct"] < 70:
            issues.append("Ball tracking coverage below 70%")
        if scores["unknown_classification_pct"] > 30:
            issues.append("High unknown classification rate")
        if scores["tracking_score"] < 80:
            issues.append("Frequent ID switches detected")
        if total_latency > 200:
            issues.append("High pipeline latency")
        if scores["detection_score"] < 70:
            issues.append("Low player detection rate")

        return {
            **scores,
            "total_frames": total_frames,
            "preprocessing_ms": round(preprocessing_time, 2),
            "detection_ms": round(detection_time, 2),
            "tracking_ms": round(tracking_time, 2),
            "ball_tracking_ms": round(ball_time, 2),
            "classification_ms": round(classification_time, 2),
            "total_latency_ms": round(total_latency, 2),
            "grade": grade,
            "issues": issues,
            "team_switches": len(self.team_switches),
        }

    def _write_report(self, report: Dict) -> None:
        tm = report.get("track_metrics", {})
        lines = []
        lines.append("# CV Pipeline Validation Report\n")
        lines.append(f"Overall Score: {report['overall_score']}/100")
        lines.append(f"Grade: {report['grade']}")
        lines.append(f"Total Frames: {report['total_frames']}")
        lines.append(f"FPS: {report['fps']}")

        lines.append("\n---\n")
        lines.append("## Detection Quality")
        lines.append(f"- Avg players/frame: {report['avg_players_per_frame']}")
        lines.append(f"- Max players/frame: {report['max_players_per_frame']}")
        lines.append(f"- Score: {report['detection_score']}/100")

        lines.append("\n---\n")
        lines.append("## Tracking Quality")
        lines.append(f"- Unique tracks: {tm.get('total_unique_tracks', 0)}")
        lines.append(f"- Avg track lifetime: {report['avg_track_lifetime']}")
        lines.append(f"- Longest track: {tm.get('longest_track_lifetime', 0)}")
        lines.append(f"- ID switches: {tm.get('estimated_id_switches', 0)}")
        lines.append(f"- Recovered tracks: {tm.get('recovered_tracks', 0)}")
        lines.append(f"- Score: {report['tracking_score']}/100")

        lines.append("\n---\n")
        lines.append("## Ball Tracking")
        lines.append(f"- Coverage: {report['ball_coverage_pct']}%")
        lines.append(f"- Score: {report['ball_score']}/100")

        lines.append("\n---\n")
        lines.append("## Team Classification")
        lines.append(f"- Unknown %: {report['unknown_classification_pct']}%")
        lines.append(f"- Score: {report['team_score']}/100")
        lines.append(f"- Team switches: {report['team_switches']}")

        lines.append("\n---\n")
        lines.append("## Performance")
        lines.append(f"- Preprocessing: {report['preprocessing_ms']:.2f} ms")
        lines.append(f"- Detection: {report['detection_ms']:.2f} ms")
        lines.append(f"- Tracking: {report['tracking_ms']:.2f} ms")
        lines.append(f"- Ball tracking: {report['ball_tracking_ms']:.2f} ms")
        lines.append(f"- Classification: {report['classification_ms']:.2f} ms")
        lines.append(f"- Total: {report['total_latency_ms']:.2f} ms")

        lines.append("\n---\n")
        lines.append("## Issues")
        for issue in report.get("issues", []):
            lines.append(f"- {issue}")

        lines.append("\n---\n")
        lines.append("## Production Readiness")
        if report["grade"] in ["A+", "A"]:
            lines.append("READY for analytics development.")
        else:
            lines.append("NOT READY for production. Address issues above.")

        (OUTPUT_DIR / "validation_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    validator = PipelineValidator()
    report = validator.validate()
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())