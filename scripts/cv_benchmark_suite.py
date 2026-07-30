import csv
import json
import math
import time
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from sklearn.cluster import KMeans
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]
VIDEO = ROOT / "videos" / "raw" / "match30.mp4"
OUT = ROOT / "outputs" / "cv_benchmark_match30"
PRED_DIR = OUT / "per_frame_predictions"
BALL_VIZ = OUT / "ball_track_visualizations"

MODELS = {
    "YOLOv8x": ROOT / "yolov8x.pt",
    "YOLO11x": ROOT / "yolo11x.pt",
    "YOLO11l": ROOT / "yolo11l.pt",
    "YOLO11m": ROOT / "yolo11m.pt",
}

PITCH_ROI = np.array([[12, 520], [1827, 492], [1875, 793], [81, 915]], dtype=np.int32)
CONF = 0.30
IOU = 0.55
IMGSZ = 960


def ensure_dirs() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    BALL_VIZ.mkdir(parents=True, exist_ok=True)


def sample_frame_numbers(total: int, n: int) -> List[int]:
    return sorted(set(int(round(x)) for x in np.linspace(1, total, n)))


def read_frames(frame_numbers: List[int]) -> Dict[int, np.ndarray]:
    cap = cv2.VideoCapture(str(VIDEO))
    frames = {}
    wanted = set(frame_numbers)
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        idx += 1
        if idx in wanted:
            frames[idx] = frame
    cap.release()
    return frames


def preprocess(frame: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    if float(l.std()) < 45:
        l = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(8, 8)).apply(l)
    balanced = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)
    gray = cv2.cvtColor(balanced, cv2.COLOR_BGR2GRAY)
    if float(cv2.Laplacian(gray, cv2.CV_64F).var()) < 100:
        blur = cv2.GaussianBlur(balanced, (0, 0), 1.1)
        balanced = cv2.addWeighted(balanced, 1.3, blur, -0.3, 0)
    return balanced


def foot(b: Tuple[int, int, int, int]) -> Tuple[int, int]:
    x1, _, x2, y2 = b
    return int((x1 + x2) / 2), int(y2)


def center(b: Tuple[int, int, int, int]) -> Tuple[int, int]:
    x1, y1, x2, y2 = b
    return int((x1 + x2) / 2), int((y1 + y2) / 2)


def inside_pitch(pt: Tuple[int, int], margin: float = 28.0) -> bool:
    return cv2.pointPolygonTest(PITCH_ROI, (float(pt[0]), float(pt[1])), True) >= -margin


def parse_boxes(result) -> List[Dict]:
    rows = []
    if not result or result[0].boxes is None:
        return rows
    for box in result[0].boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        rows.append({"cls": cls_id, "conf": conf, "bbox": [x1, y1, x2, y2]})
    return rows


def split_predictions(preds: List[Dict]) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    players, rejected, balls = [], [], []
    for p in preds:
        b = tuple(p["bbox"])
        if p["cls"] == 0:
            x1, y1, x2, y2 = b
            h = y2 - y1
            w = x2 - x1
            if inside_pitch(foot(b)) and h >= 16 and 0.12 <= w / max(h, 1) <= 0.95:
                players.append(p)
            else:
                rejected.append(p)
        elif p["cls"] == 32:
            if inside_pitch(center(b), 65) and (b[2] - b[0]) * (b[3] - b[1]) <= 2600:
                balls.append(p)
    return players, rejected, balls


def load_model(path: Path, device: str) -> Optional[YOLO]:
    if not path.exists() or path.stat().st_size < 1_000_000:
        return None
    try:
        model = YOLO(str(path))
        model.to(device)
        try:
            model.fuse()
        except Exception:
            pass
        if device.startswith("cuda"):
            model.model.half()
        return model
    except Exception as exc:
        print(f"Model unavailable or invalid: {path.name}: {exc}")
        return None


def benchmark_detectors(device: str, frames: Dict[int, np.ndarray], manual_labels: Dict[int, Dict]) -> Tuple[Optional[str], List[Dict]]:
    rows = []
    summary = []
    best_model = None
    best_score = -1.0
    for model_name, path in MODELS.items():
        model = load_model(path, device)
        if model is None:
            summary.append({
                "model": model_name,
                "status": "unavailable_weight",
                "player_recall": "",
                "ball_recall": "",
                "fps": "",
                "small_player_recall": "",
                "false_positives_per_frame": "",
                "rank_score": "",
            })
            continue
        pred_rows = []
        t0 = time.perf_counter()
        for frame_no, frame in frames.items():
            img = preprocess(frame)
            with torch.inference_mode():
                result = model.predict(source=img, classes=[0, 32], conf=CONF, iou=IOU, imgsz=IMGSZ, verbose=False, device=device)
            preds = parse_boxes(result)
            players, rejected, balls = split_predictions(preds)
            gt = manual_labels[str(frame_no)]
            player_recall = min(len(players) / max(gt["visible_players"], 1), 1.0)
            ball_recall = 1.0 if gt["ball_visible"] and balls else (0.0 if gt["ball_visible"] else 1.0)
            small_recall = min(len([p for p in players if (p["bbox"][3] - p["bbox"][1]) < 60]) / max(gt["visible_small_players"], 1), 1.0) if gt["small_player_frame"] else ""
            fp = max(0, len(players) - gt["visible_players"]) + len(rejected)
            row = {
                "model": model_name,
                "frame": frame_no,
                "visible_players_proxy": gt["visible_players"],
                "detected_players": len(players),
                "player_recall_proxy": player_recall,
                "false_positives": fp,
                "ball_visible_proxy": gt["ball_visible"],
                "ball_detected": int(bool(balls)),
                "ball_recall_proxy": ball_recall,
                "small_player_frame": gt["small_player_frame"],
                "visible_small_players_proxy": gt["visible_small_players"],
                "small_players_detected": len([p for p in players if (p["bbox"][3] - p["bbox"][1]) < 60]),
                "small_player_recall_proxy": small_recall,
                "mean_confidence": float(np.mean([p["conf"] for p in players])) if players else 0.0,
                "rejected_non_players": len(rejected),
            }
            rows.append(row)
            pred_rows.append({"frame": frame_no, "players": players, "rejected": rejected, "balls": balls})
        elapsed = time.perf_counter() - t0
        (PRED_DIR / f"{model_name}_predictions.json").write_text(json.dumps(pred_rows, indent=2), encoding="utf-8")
        model_rows = [r for r in rows if r["model"] == model_name]
        fps = len(frames) / max(elapsed, 1e-6)
        player_recall = float(np.mean([r["player_recall_proxy"] for r in model_rows]))
        ball_recall = float(np.mean([r["ball_recall_proxy"] for r in model_rows if r["ball_visible_proxy"]]))
        small_vals = [r["small_player_recall_proxy"] for r in model_rows if r["small_player_recall_proxy"] != ""]
        small_recall = float(np.mean(small_vals)) if small_vals else 0.0
        fppf = float(np.mean([r["false_positives"] for r in model_rows]))
        score = player_recall * 0.45 + ball_recall * 0.35 + min(fps / 20, 1) * 0.15 - min(fppf / 10, 1) * 0.05
        summary.append({
            "model": model_name,
            "status": "ok",
            "player_recall": player_recall,
            "ball_recall": ball_recall,
            "fps": fps,
            "small_player_recall": small_recall,
            "false_positives_per_frame": fppf,
            "rank_score": score,
        })
        if score > best_score:
            best_score = score
            best_model = model_name
    write_csv(OUT / "benchmark_results.csv", rows)
    write_csv(OUT / "benchmark_summary.csv", sorted(summary, key=lambda x: str(x.get("rank_score", "")), reverse=True))
    draw_bar_plot(OUT / "comparison_plot.png", [s for s in summary if s["status"] == "ok"], ["player_recall", "ball_recall", "fps"])
    return best_model, summary


def write_csv(path: Path, rows: List[Dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    for r in rows:
        for k in r.keys():
            if k not in fields:
                fields.append(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def draw_bar_plot(path: Path, rows: List[Dict], keys: List[str]) -> None:
    img = np.full((720, 1100, 3), 255, dtype=np.uint8)
    if not rows:
        cv2.putText(img, "No available models to plot", (60, 350), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2)
        cv2.imwrite(str(path), img)
        return
    colors = [(40, 90, 230), (30, 170, 80), (230, 130, 20)]
    cv2.putText(img, "Detector Benchmark Comparison", (40, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2)
    x0 = 120
    for i, r in enumerate(rows):
        y = 120 + i * 130
        cv2.putText(img, r["model"], (30, y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2)
        for j, k in enumerate(keys):
            val = float(r[k])
            scale = 20.0 if k == "fps" else 1.0
            bw = int(min(val / scale, 1.0) * 760)
            cv2.rectangle(img, (x0, y + j * 30), (x0 + bw, y + 22 + j * 30), colors[j], -1)
            cv2.putText(img, f"{k}: {val:.3f}", (x0 + 780, y + 18 + j * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)
    cv2.imwrite(str(path), img)


def build_proxy_manual_labels(frames: Dict[int, np.ndarray], model_name: str, device: str) -> Dict[str, Dict]:
    model = load_model(MODELS[model_name], device)
    labels = {}
    manual_frames = sorted(list(frames.keys()))[:50]
    for frame_no in manual_frames:
        frame = preprocess(frames[frame_no])
        with torch.inference_mode():
            result = model.predict(source=frame, classes=[0, 32], conf=0.20, iou=0.55, imgsz=1280, verbose=False, device=device)
        players, _, balls = split_predictions(parse_boxes(result))
        small = [p for p in players if (p["bbox"][3] - p["bbox"][1]) < 60]
        labels[str(frame_no)] = {
            "frame": frame_no,
            "visible_players": max(len(players), 1),
            "detected_players": len(players),
            "ball_visible": bool(balls),
            "ball_detected": bool(balls),
            "small_player_frame": bool(small),
            "visible_small_players": len(small),
            "label_source": "proxy_from_yolov8x_highres_not_human",
        }
    all_labels = {}
    nearest = manual_frames
    for frame_no in frames:
        closest = min(nearest, key=lambda x: abs(x - frame_no))
        all_labels[str(frame_no)] = dict(labels[str(closest)])
        all_labels[str(frame_no)]["frame"] = frame_no
    write_csv(OUT / "manual_validation.csv", list(labels.values()))
    player_recall = float(np.mean([r["detected_players"] / max(r["visible_players"], 1) for r in labels.values()]))
    ball_visible = [r for r in labels.values() if r["ball_visible"]]
    ball_recall = float(np.mean([1.0 if r["ball_detected"] else 0.0 for r in ball_visible])) if ball_visible else 0.0
    small_rows = [r for r in labels.values() if r["small_player_frame"]]
    small_recall = float(np.mean([r["detected_players"] / max(r["visible_players"], 1) for r in small_rows])) if small_rows else 0.0
    write_csv(OUT / "validation_summary.csv", [{
        "label_source": "proxy_from_yolov8x_highres_not_human",
        "sampled_frames": len(labels),
        "player_recall_proxy": player_recall,
        "ball_recall_proxy": ball_recall,
        "small_player_recall_proxy": small_recall,
        "note": "Replace manual_validation.csv counts with human labels for true recall.",
    }])
    return all_labels


def write_tracker_yaml(path: Path, params: Dict) -> None:
    path.write_text(
        "\n".join([
            "tracker_type: bytetrack",
            f"track_high_thresh: {params['track_high_thresh']}",
            f"track_low_thresh: {params['track_low_thresh']}",
            f"new_track_thresh: {params['new_track_thresh']}",
            f"track_buffer: {params['track_buffer']}",
            f"match_thresh: {params['match_thresh']}",
            "fuse_score: True",
        ]) + "\n",
        encoding="utf-8",
    )


def tracking_benchmark(model_name: str, device: str, total_frames: int) -> Dict:
    model = load_model(MODELS[model_name], device)
    baseline = {"track_high_thresh": 0.5, "track_low_thresh": 0.1, "new_track_thresh": 0.5, "track_buffer": 120, "match_thresh": 0.6}
    variants = [
        ("baseline", baseline),
        ("lower_new_track", {**baseline, "new_track_thresh": 0.35}),
        ("longer_buffer", {**baseline, "track_buffer": 180}),
        ("higher_match", {**baseline, "match_thresh": 0.72}),
        ("cv_quality_combo", {"track_high_thresh": 0.35, "track_low_thresh": 0.08, "new_track_thresh": 0.28, "track_buffer": 150, "match_thresh": 0.72}),
    ]
    rows = []
    for name, params in variants:
        yaml_path = OUT / f"tracker_{name}.yaml"
        write_tracker_yaml(yaml_path, params)
        cap = cv2.VideoCapture(str(VIDEO))
        track_frames = defaultdict(list)
        lost = 0
        recovered = 0
        active = set()
        seen = set()
        frame_no = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_no += 1
            with torch.inference_mode():
                result = model.track(source=preprocess(frame), persist=True, tracker=str(yaml_path), classes=[0], conf=CONF, iou=IOU, imgsz=IMGSZ, verbose=False, device=device)
            players, _, _ = split_predictions(parse_boxes(result))
            current = set()
            for p in players:
                # Ultralytics ID is absent in parse_boxes for detect fallback; tracking uses result boxes.
                pass
            if result and result[0].boxes is not None:
                for box in result[0].boxes:
                    if box.id is None:
                        continue
                    b = tuple(map(int, box.xyxy[0]))
                    if not inside_pitch(foot(b)):
                        continue
                    tid = int(box.id[0])
                    current.add(tid)
                    track_frames[tid].append(frame_no)
                    if tid in seen and tid not in active:
                        recovered += 1
                    seen.add(tid)
            for tid in list(active):
                if tid not in current:
                    lost += 1
            active = current
            print(f"Tracking {name}: {frame_no}/{total_frames}", end="\r")
        print()
        cap.release()
        lifetimes = [len(v) for v in track_frames.values()]
        rows.append({
            "variant": name,
            **params,
            "total_unique_tracks": len(track_frames),
            "average_track_length": float(np.mean(lifetimes)) if lifetimes else 0.0,
            "median_track_length": float(np.median(lifetimes)) if lifetimes else 0.0,
            "lost_tracks": lost,
            "recovered_tracks": recovered,
            "id_switches": "not_measurable_without_ground_truth",
        })
    write_csv(OUT / "tracking_metrics.csv", rows)
    best = sorted(rows, key=lambda r: (r["total_unique_tracks"], -r["average_track_length"]))[0]
    write_csv(OUT / "tracking_before_after_summary.csv", [rows[0], best])
    return best


class SimpleBallTracker:
    def __init__(self) -> None:
        self.kf = None
        self.center = None
        self.missing = 999
        self.longest = 0
        self.current = 0
        self.frames_tracked = 0
        self.history = deque(maxlen=60)

    def _init_kf(self, c):
        kf = cv2.KalmanFilter(4, 2)
        kf.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
        kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
        kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 2.0
        kf.statePre = np.array([[c[0]], [c[1]], [0], [0]], np.float32)
        kf.statePost = kf.statePre.copy()
        return kf

    def update(self, balls: List[Dict]) -> Tuple[Optional[Tuple[int, int]], bool]:
        if self.kf is not None:
            pred = self.kf.predict()
            pred_center = (float(pred[0]), float(pred[1]))
        else:
            pred_center = None
        if balls:
            if pred_center is None:
                best = max(balls, key=lambda b: b["conf"])
            else:
                best = min(balls, key=lambda b: math.hypot(center(tuple(b["bbox"]))[0] - pred_center[0], center(tuple(b["bbox"]))[1] - pred_center[1]) - b["conf"] * 30)
            c = center(tuple(best["bbox"]))
            if self.kf is None:
                self.kf = self._init_kf(c)
            corrected = self.kf.correct(np.array([[np.float32(c[0])], [np.float32(c[1])]]))
            self.center = (int(corrected[0]), int(corrected[1]))
            self.missing = 0
            detected = True
        elif pred_center is not None and self.missing < 45:
            self.center = (int(pred_center[0]), int(pred_center[1]))
            self.missing += 1
            detected = False
        else:
            self.center = None
            self.kf = None
            self.missing = 999
            self.current = 0
            return None, False
        self.frames_tracked += 1
        self.current += 1
        self.longest = max(self.longest, self.current)
        self.history.append(self.center)
        return self.center, detected


def ball_tracking_benchmark(model_name: str, device: str, total_frames: int) -> Dict:
    model = load_model(MODELS[model_name], device)
    cap = cv2.VideoCapture(str(VIDEO))
    rows = []
    tracker = SimpleBallTracker()
    frame_no = 0
    detections = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_no += 1
        with torch.inference_mode():
            result = model.predict(source=preprocess(frame), classes=[32], conf=0.12, iou=IOU, imgsz=1280, verbose=False, device=device)
        _, _, balls = split_predictions(parse_boxes(result))
        if balls:
            detections += 1
        pos, detected = tracker.update(balls)
        rows.append({"frame": frame_no, "raw_ball_detected": int(bool(balls)), "ball_tracked": int(pos is not None), "predicted": int(pos is not None and not detected), "x": pos[0] if pos else "", "y": pos[1] if pos else ""})
        if frame_no % 75 == 0 and pos is not None:
            vis = frame.copy()
            for i in range(1, len(tracker.history)):
                cv2.line(vis, tracker.history[i - 1], tracker.history[i], (0, 255, 255), 2)
            cv2.circle(vis, pos, 10, (0, 255, 255), 2)
            cv2.imwrite(str(BALL_VIZ / f"ball_track_frame_{frame_no:04d}.jpg"), vis)
        print(f"Ball tracking: {frame_no}/{total_frames}", end="\r")
    print()
    cap.release()
    write_csv(OUT / "ball_tracking_metrics.csv", rows)
    return {
        "frames": frame_no,
        "raw_detection_frames": detections,
        "tracked_or_predicted_frames": tracker.frames_tracked,
        "longest_track": tracker.longest,
    }


def jersey_feature(frame: np.ndarray, bbox: Tuple[int, int, int, int], space: str) -> Optional[np.ndarray]:
    x1, y1, x2, y2 = bbox
    w, h = x2 - x1, y2 - y1
    if w < 8 or h < 18:
        return None
    crop = frame[max(0, y1 + int(.18 * h)):min(frame.shape[0], y1 + int(.55 * h)), max(0, x1 + int(.22 * w)):min(frame.shape[1], x2 - int(.22 * w))]
    if crop.size == 0:
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    sat, val, hue = hsv[:, :, 1], hsv[:, :, 2], hsv[:, :, 0]
    green = (hue > 32) & (hue < 88) & (sat > 45)
    mask = (sat > 25) & (val > 35) & (val < 235) & (~green)
    if mask.mean() < 0.08:
        return None
    pix = (hsv if space == "HSV" else cv2.cvtColor(crop, cv2.COLOR_BGR2LAB))[mask]
    if len(pix) < 8:
        return None
    return np.median(pix[:, :3 if space == "HSV" else 1:3], axis=0).astype(np.float32)


def jersey_benchmark(model_name: str, device: str, frames: Dict[int, np.ndarray]) -> Dict:
    model = load_model(MODELS[model_name], device)
    samples = {"HSV": [], "LAB": []}
    for frame_no, frame in frames.items():
        with torch.inference_mode():
            result = model.predict(source=preprocess(frame), classes=[0], conf=CONF, iou=IOU, imgsz=IMGSZ, verbose=False, device=device)
        players, _, _ = split_predictions(parse_boxes(result))
        for p in players:
            b = tuple(p["bbox"])
            for space in samples:
                feat = jersey_feature(frame, b, space)
                if feat is not None:
                    samples[space].append(feat)
    rows = []
    matrices = {}
    best = None
    best_unknown = 1e9
    for space, feats in samples.items():
        if len(feats) < 2:
            rows.append({"color_space": space, "status": "insufficient_samples", "team_a_accuracy": "", "team_b_accuracy": "", "unknown_rate": 1.0, "mean_confidence": 0.0})
            continue
        X = np.array(feats)
        km = KMeans(n_clusters=2, random_state=7, n_init=10).fit(X)
        d = km.transform(X)
        conf = (d.max(axis=1) - d.min(axis=1)) / np.maximum(d.max(axis=1), 1e-6)
        unknown = float((conf < 0.22).mean())
        rows.append({"color_space": space, "status": "ok_proxy_no_gt", "team_a_accuracy": "not_measured", "team_b_accuracy": "not_measured", "unknown_rate": unknown, "mean_confidence": float(conf.mean())})
        matrices[space] = Counter(km.labels_)
        if unknown < best_unknown:
            best_unknown = unknown
            best = space
    write_csv(OUT / "jersey_classification_report.csv", rows)
    draw_confusion_proxy(OUT / "confusion_matrix.png", matrices)
    return {"best_color_space": best, "rows": rows}


def draw_confusion_proxy(path: Path, matrices: Dict[str, Counter]) -> None:
    img = np.full((520, 820, 3), 255, dtype=np.uint8)
    cv2.putText(img, "Jersey Cluster Counts (Proxy, No GT)", (40, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
    y = 130
    for space, counts in matrices.items():
        cv2.putText(img, space, (50, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        total = max(sum(counts.values()), 1)
        for i in [0, 1]:
            w = int(500 * counts[i] / total)
            cv2.rectangle(img, (180, y - 25 + i * 45), (180 + w, y + i * 45), (60 + i * 130, 90, 220 - i * 120), -1)
            cv2.putText(img, f"cluster {i}: {counts[i]}", (700, y - 7 + i * 45), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)
        y += 130
    cv2.imwrite(str(path), img)


def write_report(best_detector, best_tracker, ball_stats, jersey_stats, summary) -> None:
    lines = [
        "# CV Benchmark Report - match30.mp4",
        "",
        "Ground-truth note: no human annotations were present in the repository. `manual_validation.csv` was generated as a proxy seed from high-resolution YOLOv8x detections and must be manually corrected before treating recall/accuracy as true labels.",
        "",
        f"Best available detector: {best_detector or 'none'}",
        f"Best tracker settings: {best_tracker}",
        f"Ball tracking: {ball_stats}",
        f"Best jersey color space: {jersey_stats.get('best_color_space')}",
        "",
        "Generated artifacts:",
        "- benchmark_results.csv",
        "- benchmark_summary.csv",
        "- comparison_plot.png",
        "- tracking_metrics.csv",
        "- tracking_before_after_summary.csv",
        "- ball_tracking_metrics.csv",
        "- ball_track_visualizations/",
        "- jersey_classification_report.csv",
        "- confusion_matrix.png",
        "- manual_validation.csv",
        "- validation_summary.csv",
        "",
        "Next improvement: replace proxy manual labels with real human labels for the 50 sampled frames, then rerun this same benchmark so threshold/model changes are judged against true recall and false positives.",
    ]
    (OUT / "cv_benchmark_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    ensure_dirs()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark run.")
    device = "cuda:0"
    cap = cv2.VideoCapture(str(VIDEO))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    bench_nums = sample_frame_numbers(total, 100)
    manual_nums = sample_frame_numbers(total, 50)
    frames = read_frames(sorted(set(bench_nums + manual_nums)))
    manual_labels = build_proxy_manual_labels({k: frames[k] for k in manual_nums}, "YOLOv8x", device)
    for n in bench_nums:
        if str(n) not in manual_labels:
            closest = min(manual_nums, key=lambda x: abs(x - n))
            manual_labels[str(n)] = dict(manual_labels[str(closest)])
            manual_labels[str(n)]["frame"] = n
    best_detector, summary = benchmark_detectors(device, {k: frames[k] for k in bench_nums}, manual_labels)
    if best_detector is None:
        best_detector = "YOLOv8x"
    best_tracker = tracking_benchmark(best_detector, device, total)
    ball_stats = ball_tracking_benchmark(best_detector, device, total)
    jersey_stats = jersey_benchmark(best_detector, device, {k: frames[k] for k in bench_nums})
    write_report(best_detector, best_tracker, ball_stats, jersey_stats, summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

