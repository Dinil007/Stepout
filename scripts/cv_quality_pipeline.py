import csv
import json
import math
import shutil
import subprocess
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from sklearn.cluster import KMeans
from ultralytics import YOLO

from app.core.config import get_config
from app.detection.detection_types import Detection
from app.detection.detection_filter import parse_yolo_results, inside_pitch, split_dets
from app.tracking.player_tracker import PlayerTracker
from app.tracking.tracking_metrics import TrackingMetricsCollector
from app.tracking.ball_tracker import BallTracker
from app.tracking.ball_tracking_metrics import BallTrackingMetricsWriter
from app.team_classification.jersey_classifier import JerseyClassifier
from app.team_classification.team_metrics import TeamMetricsCollector


ROOT = Path(__file__).resolve().parents[1]
INPUT_VIDEO = ROOT / "videos" / "raw" / "match30.mp4"
OUTPUT_DIR = ROOT / "outputs" / "cv_quality_match30"
DEBUG_DIR = ROOT / "outputs" / "debug"
MODEL_WEIGHTS = ROOT / "yolov8x.pt"
TRACKER_CONFIG = OUTPUT_DIR / "bytetrack_cv_quality.yaml"

PITCH_ROI = np.array(
    [[12, 520], [1827, 492], [1875, 793], [81, 915]],
    dtype=np.int32,
)

CONF_VALUES = [0.15, 0.20, 0.25, 0.30]
IMGSZ_VALUES = [640, 960, 1280]


class AdaptivePreprocessor:
    def __init__(self) -> None:
        self.metrics: List[Dict] = []

    @staticmethod
    def measure(frame: np.ndarray, frame_no: int) -> Dict:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mean = float(gray.mean())
        std = float(gray.std())
        lap = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        noise = float(np.std(gray.astype(np.float32) - cv2.GaussianBlur(gray, (5, 5), 0).astype(np.float32)))
        low = float((gray < 35).mean())
        high = float((gray > 220).mean())
        shadow = float((gray < 55).mean())
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        mag = cv2.magnitude(gx, gy)
        angle = cv2.phase(gx, gy, angleInDegrees=True)
        hist, _ = np.histogram(angle[mag > np.percentile(mag, 80)], bins=18, range=(0, 180))
        motion_blur = float(hist.max() / max(hist.sum(), 1))
        return {
            "frame": frame_no,
            "brightness": mean,
            "contrast": std,
            "blur_laplacian_var": lap,
            "noise_level": noise,
            "underexposed_ratio": low,
            "overexposed_ratio": high,
            "shadow_ratio": shadow,
            "motion_blur_score": motion_blur,
        }

    @staticmethod
    def white_balance(frame: np.ndarray) -> np.ndarray:
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB).astype(np.float32)
        avg_a = lab[:, :, 1].mean()
        avg_b = lab[:, :, 2].mean()
        lab[:, :, 1] -= (avg_a - 128.0) * (lab[:, :, 0] / 255.0) * 0.6
        lab[:, :, 2] -= (avg_b - 128.0) * (lab[:, :, 0] / 255.0) * 0.6
        return cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)

    @staticmethod
    def gamma(frame: np.ndarray, gamma: float) -> np.ndarray:
        inv = 1.0 / max(gamma, 1e-6)
        table = np.array([(i / 255.0) ** inv * 255 for i in range(256)], dtype=np.uint8)
        return cv2.LUT(frame, table)

    @staticmethod
    def clahe(frame: np.ndarray, clip: float = 1.8, grid: Tuple[int, int] = (8, 8)) -> np.ndarray:
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        cl = cv2.createCLAHE(clipLimit=clip, tileGridSize=grid).apply(l)
        return cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2BGR)

    @staticmethod
    def sharpen(frame: np.ndarray, strength: float = 0.35) -> np.ndarray:
        blur = cv2.GaussianBlur(frame, (0, 0), 1.1)
        return cv2.addWeighted(frame, 1.0 + strength, blur, -strength, 0)

    def apply(self, frame: np.ndarray, m: Dict) -> Tuple[np.ndarray, List[str]]:
        out = frame
        steps = []
        if abs(cv2.cvtColor(out, cv2.COLOR_BGR2LAB)[:, :, 1].mean() - 128) > 4:
            out = self.white_balance(out)
            steps.append("white_balance")
        if m["brightness"] < 85:
            out = self.gamma(out, 1.18)
            steps.append("gamma_brighten")
        elif m["brightness"] > 185 or m["overexposed_ratio"] > 0.08:
            out = self.gamma(out, 0.92)
            steps.append("gamma_darken")
        if m["contrast"] < 42:
            out = self.clahe(out, clip=1.8, grid=(8, 8))
            steps.append("clahe")
        if m["noise_level"] > 9.0:
            out = cv2.fastNlMeansDenoisingColored(out, None, 3, 3, 7, 21)
            steps.append("denoise_light")
        if m["blur_laplacian_var"] < 95 or m["motion_blur_score"] > 0.28:
            out = self.sharpen(out, 0.30)
            steps.append("adaptive_sharpen")
        return out, steps


def write_tracker_config() -> None:
    TRACKER_CONFIG.write_text(
        "\n".join(
            [
                "tracker_type: bytetrack",
                "track_high_thresh: 0.35",
                "track_low_thresh: 0.08",
                "new_track_thresh: 0.28",
                "track_buffer: 150",
                "match_thresh: 0.72",
                "fuse_score: True",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def setup() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    DEBUG_DIR.mkdir(parents=True, exist_ok=True)
    write_tracker_config()


def device_report() -> Dict:
    info = {
        "pytorch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count(),
        "current_device": torch.cuda.current_device() if torch.cuda.is_available() else None,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "nvidia_smi": shutil.which("nvidia-smi"),
    }
    if info["nvidia_smi"]:
        try:
            info["nvidia_driver"] = subprocess.check_output(
                [info["nvidia_smi"], "--query-gpu=driver_version", "--format=csv,noheader"],
                text=True,
                timeout=10,
            ).strip()
        except Exception as exc:
            info["nvidia_driver"] = str(exc)
    return info


def evaluate_frames(preprocessor: AdaptivePreprocessor) -> Tuple[float, int, int, int]:
    cap = cv2.VideoCapture(str(INPUT_VIDEO))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {INPUT_VIDEO}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    csv_path = OUTPUT_DIR / "frame_quality_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = None
        frame_no = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_no += 1
            m = preprocessor.measure(frame, frame_no)
            preprocessor.metrics.append(m)
            if writer is None:
                writer = csv.DictWriter(f, fieldnames=list(m.keys()))
                writer.writeheader()
            writer.writerow(m)
            print(f"Quality scan: {frame_no}/{total}", end="\r")
    cap.release()
    print()
    return fps, total, width, height


def load_sample_frames(stride: int = 25, limit: int = 30) -> List[Tuple[int, np.ndarray]]:
    cap = cv2.VideoCapture(str(INPUT_VIDEO))
    frames = []
    idx = 0
    while len(frames) < limit:
        ret, frame = cap.read()
        if not ret:
            break
        idx += 1
        if idx == 1 or idx % stride == 0:
            frames.append((idx, frame))
    cap.release()
    return frames


def choose_config(model: YOLO, device: str, preprocessor: AdaptivePreprocessor, sample_frames) -> Dict:
    metric_by_frame = {m["frame"]: m for m in preprocessor.metrics}
    candidates = []
    for use_pre in [False, True]:
        for conf in CONF_VALUES:
            for imgsz in IMGSZ_VALUES:
                player_counts = []
                ball_frames = 0
                rejected = 0
                confs = []
                for frame_no, frame in sample_frames:
                    infer = frame
                    if use_pre:
                        infer, _ = preprocessor.apply(frame, metric_by_frame.get(frame_no) or preprocessor.measure(frame, frame_no))
                    with torch.inference_mode():
                        results = model.predict(
                            source=infer,
                            classes=[0, 32],
                            conf=conf,
                            iou=0.55,
                            imgsz=imgsz,
                            verbose=False,
                            device=device,
                        )
                    players = 0
                    has_ball = False
                    for d in parse_results(results):
                        if d.cls_id == 0:
                            if inside_pitch(d.foot):
                                players += 1
                                confs.append(d.conf)
                            else:
                                rejected += 1
                        elif d.cls_id == 32 and inside_pitch(d.center, margin=65):
                            has_ball = True
                    if has_ball:
                        ball_frames += 1
                    player_counts.append(players)
                avg_players = float(np.mean(player_counts)) if player_counts else 0.0
                stability = float(np.std(player_counts)) if player_counts else 99.0
                avg_conf = float(np.mean(confs)) if confs else 0.0
                score = avg_players + 0.45 * ball_frames - 0.05 * stability - 0.03 * rejected + 0.5 * avg_conf
                candidates.append(
                    {
                        "preprocess": use_pre,
                        "conf": conf,
                        "imgsz": imgsz,
                        "avg_players": avg_players,
                        "ball_frames": ball_frames,
                        "rejected": rejected,
                        "avg_conf": avg_conf,
                        "score": score,
                    }
                )
                print(f"Config eval pre={use_pre} conf={conf} imgsz={imgsz} score={score:.2f}")
    candidates.sort(key=lambda c: c["score"], reverse=True)
    (OUTPUT_DIR / "config_search_results.json").write_text(json.dumps(candidates, indent=2), encoding="utf-8")
    return candidates[0]


def draw_detections(frame: np.ndarray, accepted: List[Det], rejected: List[Det], balls: List[Det], title: str) -> np.ndarray:
    out = frame.copy()
    cv2.polylines(out, [PITCH_ROI], True, (255, 255, 0), 2)
    for d in rejected:
        x1, y1, x2, y2 = d.bbox
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(out, "REJECT", (x1, max(15, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
    for d in accepted:
        x1, y1, x2, y2 = d.bbox
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 220, 0), 2)
        cv2.circle(out, d.foot, 3, (0, 255, 255), -1)
        cv2.putText(out, f"P {d.conf:.2f}", (x1, max(15, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 0), 1)
    for d in balls:
        cv2.circle(out, d.center, max(5, int(math.sqrt(max(d.area, 1)) / 2) + 4), (0, 255, 255), 2)
        cv2.putText(out, f"BALL {d.conf:.2f}", (d.center[0] + 8, d.center[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
    cv2.putText(out, title, (18, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    return out


def make_comparison(model: YOLO, device: str, preprocessor: AdaptivePreprocessor, selected: Dict) -> None:
    frame_no, frame = load_sample_frames(stride=125, limit=4)[-1]
    m = preprocessor.metrics[frame_no - 1]
    pre, steps = preprocessor.apply(frame, m)
    with torch.inference_mode():
        results = model.predict(source=pre if selected["preprocess"] else frame, classes=[0, 32], conf=selected["conf"], iou=0.55, imgsz=selected["imgsz"], verbose=False, device=device)
    accepted, rejected, balls = split_dets(parse_results(results))
    h, w = frame.shape[:2]
    panels = [
        cv2.resize(frame, (w // 2, h // 2)),
        cv2.resize(pre, (w // 2, h // 2)),
        cv2.resize(draw_detections(pre if selected["preprocess"] else frame, accepted, rejected, balls, "Detection Result"), (w // 2, h // 2)),
    ]
    label_names = ["Original", "Preprocessed: " + ",".join(steps or ["none"]), "Detection Result"]
    for panel, label in zip(panels, label_names):
        cv2.rectangle(panel, (0, 0), (panel.shape[1], 46), (0, 0, 0), -1)
        cv2.putText(panel, label, (12, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
    canvas = np.vstack(panels)
    cv2.imwrite(str(DEBUG_DIR / "preprocessing_comparison.jpg"), canvas)


def run_full(model: YOLO, device: str, preprocessor: AdaptivePreprocessor, selected: Dict, fps: float, total: int, width: int, height: int) -> Dict:
    cap = cv2.VideoCapture(str(INPUT_VIDEO))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writers = {
        "detection": cv2.VideoWriter(str(OUTPUT_DIR / "detection.mp4"), fourcc, fps, (width, height)),
        "tracking": cv2.VideoWriter(str(OUTPUT_DIR / "tracking.mp4"), fourcc, fps, (width, height)),
        "team": cv2.VideoWriter(str(OUTPUT_DIR / "team_classification.mp4"), fourcc, fps, (width, height)),
    }
    jersey = JerseyClassifier()
    team_metrics = TeamMetricsCollector(OUTPUT_DIR / "team_classification_metrics.csv")
    ball_tracker = BallTracker()
    ball_metrics_writer = BallTrackingMetricsWriter(OUTPUT_DIR / "ball_tracking_metrics.csv")
    player_tracker = PlayerTracker()
    metrics_collector = TrackingMetricsCollector(OUTPUT_DIR)
    frame_rows = []
    all_player_conf = []
    all_ball_conf = []
    false_positive_removed = 0
    ball_detect_frames = 0
    ball_track_frames = 0
    max_players = 0

    frame_no = 0
    with torch.inference_mode():
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_no += 1
            m = preprocessor.metrics[frame_no - 1]
            infer, steps = preprocessor.apply(frame, m) if selected["preprocess"] else (frame, [])
            results = model.track(
                source=infer,
                persist=True,
                tracker=str(TRACKER_CONFIG),
                classes=[0, 32],
                conf=float(selected["conf"]),
                iou=0.55,
                imgsz=int(selected["imgsz"]),
                verbose=False,
                device=device,
            )
            players, rejected, balls = split_dets(parse_results(results))
            false_positive_removed += len(rejected)
            max_players = max(max_players, len(players))
            all_player_conf.extend([d.conf for d in players])
            all_ball_conf.extend([d.conf for d in balls])
            ball_result = ball_tracker.update([{
                "center": b.center,
                "bbox": b.bbox,
                "confidence": b.confidence,
            } for b in balls], frame_no)
            detected_ball = ball_result is not None
            active_ball = ball_result is not None and not ball_result.get("is_predicted", False)
            if detected_ball:
                ball_detect_frames += 1
            if active_ball:
                ball_track_frames += 1

            ball_metrics_writer.record(
                frame_index=frame_no,
                raw_detection=len(balls),
                accepted_detection=1 if detected_ball else 0,
                predicted=ball_result.get("is_predicted", False) if ball_result else False,
                missing=ball_result is None,
                confidence=ball_result.get("confidence", 0.0) if ball_result else 0.0,
                track_length=ball_tracker.longest_streak(),
                coverage_ratio=ball_tracker.coverage_frames / max(ball_tracker.total_frames, 1),
            )

            # Update player tracker with config-driven tracking
            tracked_players = player_tracker.update(players, frame.shape[:2], frame_no)
            metrics_collector.record_frame(frame_no, tracked_players)

            for switch in player_tracker.possible_switches:
                metrics_collector.record_id_switch(switch)

            if frame_no in (45, 90, 150) or (frame_no > 150 and frame_no % 75 == 0):
                jersey.fit_if_ready()
            team_counts = Counter()
            unknown = 0
            team_lines = []
            for d in tracked_players:
                if d.track_id >= 0:
                    jersey.update_sample(d.track_id, frame, d.bbox)
                label, tconf = jersey.classify(d.track_id, frame, d.bbox)
                team_metrics.record(frame_no, d.track_id, label, tconf, d.bbox[3] - d.bbox[1], tconf)
                if label == "Unknown":
                    unknown += 1
                    color = (180, 180, 180)
                elif label.startswith("Team A"):
                    team_counts["A"] += 1
                    color = (40, 60, 255)
                else:
                    team_counts["B"] += 1
                    color = (255, 70, 30)
                x1, y1, x2, y2 = d.bbox
                cv2.rectangle(team_frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(team_frame, f"{label} {tconf:.2f} ID {d.track_id}", (x1, max(15, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)

            det_frame = draw_detections(frame, players, rejected, balls, f"Detections F{frame_no}")
            trk_frame = frame.copy()
            cv2.polylines(trk_frame, [PITCH_ROI], True, (255, 255, 0), 2)
            for d in tracked_players:
                x1, y1, x2, y2 = d.bbox
                cv2.rectangle(trk_frame, (x1, y1), (x2, y2), (255, 140, 0), 2)
                cv2.putText(trk_frame, f"ID {d.track_id}", (x1, max(15, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 140, 0), 2)
            if ball_track.center is not None and active_ball:
                bc = (int(ball_track.center[0]), int(ball_track.center[1]))
                cv2.circle(trk_frame, bc, 9, (0, 255, 255) if detected_ball else (0, 165, 255), 2)
                for i in range(1, len(ball_track.history)):
                    p1 = tuple(map(int, ball_track.history[i - 1]))
                    p2 = tuple(map(int, ball_track.history[i]))
                    cv2.line(trk_frame, p1, p2, (0, 220, 255), 1)

            team_frame = frame.copy()
            team_counts = Counter()
            unknown = 0
            for d in tracked_players:
                label, tconf = jersey.assign(d.track_id)
                if label == "Unknown":
                    unknown += 1
                    color = (180, 180, 180)
                elif label.endswith("A"):
                    team_counts["A"] += 1
                    color = (40, 60, 255)
                else:
                    team_counts["B"] += 1
                    color = (255, 70, 30)
                x1, y1, x2, y2 = d.bbox
                cv2.rectangle(team_frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(team_frame, f"{label} {tconf:.2f} ID {d.track_id}", (x1, max(15, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)

            for name, img in [("detection", det_frame), ("tracking", trk_frame), ("team", team_frame)]:
                writers[name].write(img)

            frame_rows.append(
                {
                    "frame": frame_no,
                    "players_detected": len(players),
                    "players_tracked": len(tracked_players),
                    "ball_detected": int(detected_ball),
                    "ball_tracked": int(active_ball),
                    "rejected_non_players": len(rejected),
                    "rejected_coaches": len(rejected),
                    "rejected_spectators": 0,
                    "team_a": team_counts["A"],
                    "team_b": team_counts["B"],
                    "unknown": unknown,
                    "preprocessing_steps": ",".join(steps),
                }
            )
            print(f"CV full pass: {frame_no}/{total}", end="\r")
    print()
    cap.release()
    for w in writers.values():
        w.release()

    frame_csv = OUTPUT_DIR / "frame_cv_counts.csv"
    with frame_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(frame_rows[0].keys()))
        writer.writeheader()
        writer.writerows(frame_rows)

    tracking_metrics = metrics_collector.get_metrics(frame_no)
    avg_players = float(np.mean([r["players_detected"] for r in frame_rows])) if frame_rows else 0.0
    avg_tracked = float(np.mean([r["players_tracked"] for r in frame_rows])) if frame_rows else 0.0
    conf_percentiles = np.percentile(all_player_conf, [5, 25, 50, 75, 95]).tolist() if all_player_conf else []
    ball_metrics = ball_tracker.get_metrics()
    selected_report = {
        "frames_processed": frame_no,
        "average_players_detected": avg_players,
        "maximum_players_detected": max_players,
        "missed_players_proxy": int(sum(max(0, max_players - r["players_detected"]) for r in frame_rows)),
        "false_positives_removed": false_positive_removed,
        "coach_filtering_performance_proxy": {
            "rejected_by_pitch_foot_filter": false_positive_removed,
            "method": "bottom-center foot point outside calibrated playable-pitch polygon",
        },
        "player_confidence_distribution": {
            "count": len(all_player_conf),
            "mean": float(np.mean(all_player_conf)) if all_player_conf else 0.0,
            "percentiles_5_25_50_75_95": conf_percentiles,
        },
        "ball_tracking_statistics": {
            "frames_with_ball_detection": ball_detect_frames,
            "frames_without_ball_detection": frame_no - ball_detect_frames,
            "frames_with_ball_track_or_prediction": ball_track_frames,
            "detection_confidence_mean": float(np.mean(all_ball_conf)) if all_ball_conf else 0.0,
            "longest_continuous_ball_track": ball_tracker.longest_streak(),
        },
        "tracking_statistics": tracking_metrics,
        "team_classification": {
            "team_a_tracks": sum(1 for v in jersey.track_team.values() if v == 0),
            "team_b_tracks": sum(1 for v in jersey.track_team.values() if v == 1),
            "unknown_tracks": sum(1 for v in jersey.track_team.values() if v is None),
            "mean_confidence": float(np.mean(list(jersey.track_conf.values()))) if jersey.track_conf else 0.0,
            "accuracy": "not measured: no ground-truth team labels supplied",
        },
        "possible_id_switch_events": tracking_metrics.get("possible_id_switch_events", []),
    }
    metrics_collector.flush()
    ball_metrics_writer.flush()
    (OUTPUT_DIR / "cv_quality_report.json").write_text(json.dumps(selected_report, indent=2), encoding="utf-8")
    return selected_report


def video_validation(width: int, height: int) -> Dict:
    result = {}
    for name in ["detection.mp4", "tracking.mp4", "team_classification.mp4"]:
        path = OUTPUT_DIR / name
        cap = cv2.VideoCapture(str(path))
        result[name] = {
            "exists": path.exists(),
            "size_bytes": path.stat().st_size if path.exists() else 0,
            "opens": bool(cap.isOpened()),
            "frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.isOpened() else 0,
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) if cap.isOpened() else 0,
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) if cap.isOpened() else 0,
            "expected_width": width,
            "expected_height": height,
        }
        cap.release()
    return result


def main() -> int:
    setup()
    gpu = device_report()
    if not gpu["cuda_available"]:
        raise RuntimeError(f"CUDA unavailable for CV quality run: {gpu}")
    device = "cuda:0"
    print(json.dumps(gpu, indent=2))
    print("Running on GPU")

    preprocessor = AdaptivePreprocessor()
    fps, total, width, height = evaluate_frames(preprocessor)
    model = YOLO(str(MODEL_WEIGHTS))
    model.to(device)
    try:
        model.fuse()
    except Exception:
        pass
    model.model.half()

    sample_frames = load_sample_frames()
    selected = choose_config(model, device, preprocessor, sample_frames)
    make_comparison(model, device, preprocessor, selected)
    report = run_full(model, device, preprocessor, selected, fps, total, width, height)
    final = {
        "input_video": str(INPUT_VIDEO),
        "output_dir": str(OUTPUT_DIR),
        "gpu": gpu,
        "selected_config": selected,
        "preprocessing_summary": {
            "mean_brightness": float(np.mean([m["brightness"] for m in preprocessor.metrics])),
            "mean_contrast": float(np.mean([m["contrast"] for m in preprocessor.metrics])),
            "mean_blur_laplacian_var": float(np.mean([m["blur_laplacian_var"] for m in preprocessor.metrics])),
            "mean_noise_level": float(np.mean([m["noise_level"] for m in preprocessor.metrics])),
            "mean_shadow_ratio": float(np.mean([m["shadow_ratio"] for m in preprocessor.metrics])),
            "mean_motion_blur_score": float(np.mean([m["motion_blur_score"] for m in preprocessor.metrics])),
        },
        "cv_report": report,
        "video_validation": video_validation(width, height),
        "debug_preprocessing_comparison": str(DEBUG_DIR / "preprocessing_comparison.jpg"),
    }
    (OUTPUT_DIR / "final_cv_quality_report.json").write_text(json.dumps(final, indent=2), encoding="utf-8")
    print(json.dumps(final, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
