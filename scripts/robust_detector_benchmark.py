import csv
import gc
import json
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import cv2
import numpy as np
import torch

try:
    import psutil
except Exception:
    psutil = None

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None


ROOT = Path(__file__).resolve().parents[1]
VIDEO = ROOT / "videos" / "raw" / "match30.mp4"
OUT = ROOT / "outputs" / "cv_benchmark_match30"
BATCH_DIR = OUT / "batch_results"
PRED_DIR = OUT / "per_frame_predictions"
DEBUG_LOG = OUT / "benchmark_debug.log"

MODEL_NAMES = ["yolov8x.pt", "yolo11x.pt", "yolo11l.pt", "yolo11m.pt"]
MIN_MODEL_BYTES = {
    "yolov8x.pt": 100_000_000,
    "yolo11x.pt": 100_000_000,
    "yolo11l.pt": 45_000_000,
    "yolo11m.pt": 35_000_000,
}

CONF = 0.30
IOU = 0.55
IMGSZ = 960
BATCH_SIZE = 100
DOWNLOAD_TIMEOUT_SEC = 180

PITCH_ROI = np.array([[12, 520], [1827, 492], [1875, 793], [81, 915]], dtype=np.int32)


def log_debug(message: str) -> None:
    DEBUG_LOG.parent.mkdir(parents=True, exist_ok=True)
    with DEBUG_LOG.open("a", encoding="utf-8") as f:
        f.write(message.rstrip() + "\n")


def write_csv(path: Path, rows: List[Dict], append: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        if not path.exists():
            path.write_text("", encoding="utf-8")
        return
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    exists = path.exists() and path.stat().st_size > 0
    with path.open("a" if append else "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if not append or not exists:
            writer.writeheader()
        writer.writerows(rows)


def video_frame_count() -> int:
    cap = cv2.VideoCapture(str(VIDEO))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {VIDEO}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return total


def representative_frames(total: int, count: int = 100) -> List[int]:
    if total <= count:
        return list(range(total))
    return sorted({int(round(x)) for x in np.linspace(0, total - 1, count)})


def manual_frames(total: int, count: int = 50) -> List[int]:
    if total <= count:
        return list(range(total))
    return sorted({int(round(x)) for x in np.linspace(0, total - 1, count)})


def preprocess(frame: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    if float(l.std()) < 45:
        l = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(8, 8)).apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)


def foot(bbox: Tuple[int, int, int, int]) -> Tuple[int, int]:
    x1, _, x2, y2 = bbox
    return int((x1 + x2) / 2), int(y2)


def center(bbox: Tuple[int, int, int, int]) -> Tuple[int, int]:
    x1, y1, x2, y2 = bbox
    return int((x1 + x2) / 2), int((y1 + y2) / 2)


def inside_pitch(point: Tuple[int, int], margin: float = 28.0) -> bool:
    return cv2.pointPolygonTest(PITCH_ROI, (float(point[0]), float(point[1])), True) >= -margin


def parse_predictions(result) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    players, rejected, balls = [], [], []
    if not result or result[0].boxes is None:
        return players, rejected, balls
    for box in result[0].boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        bbox = (x1, y1, x2, y2)
        if cls_id == 0:
            h = y2 - y1
            w = x2 - x1
            row = {"class": cls_id, "conf": conf, "x1": x1, "y1": y1, "x2": x2, "y2": y2}
            if inside_pitch(foot(bbox)) and h >= 16 and 0.12 <= w / max(h, 1) <= 0.95:
                players.append(row)
            else:
                rejected.append(row)
        elif cls_id == 32:
            area = max(0, x2 - x1) * max(0, y2 - y1)
            if area <= 2600 and inside_pitch(center(bbox), 65):
                balls.append({"class": cls_id, "conf": conf, "x1": x1, "y1": y1, "x2": x2, "y2": y2})
    return players, rejected, balls


def try_download_model(weight_name: str) -> Tuple[bool, str]:
    code = (
        "from ultralytics import YOLO\n"
        f"YOLO({weight_name!r})\n"
        "print('download_or_load_ok')\n"
    )
    try:
        proc = subprocess.run(
            [sys.executable, "-c", code],
            cwd=str(ROOT),
            capture_output=True,
            timeout=DOWNLOAD_TIMEOUT_SEC,
        )
        stdout = (proc.stdout or b"").decode("utf-8", errors="ignore")
        stderr = (proc.stderr or b"").decode("utf-8", errors="ignore")
        if proc.returncode == 0:
            return True, stdout.strip()
        return False, (stderr or stdout).strip()
    except subprocess.TimeoutExpired:
        return False, f"download timeout after {DOWNLOAD_TIMEOUT_SEC}s"
    except Exception:
        return False, traceback.format_exc()

def validate_model(weight_name: str) -> Dict:
    row = {
        "model_name": weight_name,
        "requested": True,
        "exists": False,
        "loaded": False,
        "status": "NOT_AVAILABLE",
        "error_message": "",
    }
    if YOLO is None:
        row["status"] = "NOT_AVAILABLE"
        row["error_message"] = "ultralytics import failed"
        return row

    path = ROOT / weight_name
    if not path.exists():
        ok, msg = try_download_model(weight_name)
        log_debug(f"download attempt {weight_name}: ok={ok}; {msg}")
    row["exists"] = path.exists()
    if not path.exists():
        row["error_message"] = "missing; download failed or timed out"
        return row
    min_bytes = MIN_MODEL_BYTES.get(weight_name, 1_000_000)
    if path.stat().st_size < min_bytes:
        row["error_message"] = f"corrupt or incomplete file: {path.stat().st_size} bytes < {min_bytes}"
        row["status"] = "NOT_AVAILABLE"
        return row

    try:
        model = YOLO(str(path))
        model.to("cuda:0" if torch.cuda.is_available() else "cpu")
        try:
            model.fuse()
        except Exception:
            pass
        if torch.cuda.is_available():
            model.model.half()
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        row["loaded"] = True
        row["status"] = "AVAILABLE"
        return row
    except Exception:
        row["error_message"] = traceback.format_exc().replace("\n", " | ")
        log_debug(f"load failure {weight_name}\n{traceback.format_exc()}")
        return row


def resource_row(batch_id: int, model_name: str, phase: str) -> Dict:
    ram_mb = ""
    if psutil is not None:
        ram_mb = round(psutil.Process().memory_info().rss / (1024 * 1024), 2)
    return {
        "batch_id": batch_id,
        "model_name": model_name,
        "phase": phase,
        "ram_mb": ram_mb,
        "gpu_allocated_mb": round(torch.cuda.memory_allocated() / (1024 * 1024), 2) if torch.cuda.is_available() else 0,
        "gpu_reserved_mb": round(torch.cuda.memory_reserved() / (1024 * 1024), 2) if torch.cuda.is_available() else 0,
    }


def load_manual_labels(frame_ids: List[int], reference_model) -> Dict[int, Dict]:
    labels_path = OUT / "manual_validation.csv"
    labels = {}
    if labels_path.exists() and labels_path.stat().st_size > 0:
        with labels_path.open("r", newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                try:
                    frame = int(row["frame"])
                    labels[frame] = {
                        "visible_players": int(float(row.get("visible_players", row.get("visible_players_proxy", 0)))),
                        "ball_visible": str(row.get("ball_visible", row.get("ball_visible_proxy", "False"))).lower() in {"true", "1", "yes"},
                        "visible_small_players": int(float(row.get("visible_small_players", row.get("visible_small_players_proxy", 0)))),
                        "small_player_frame": str(row.get("small_player_frame", "False")).lower() in {"true", "1", "yes"},
                    }
                except Exception:
                    continue
    missing = [f for f in frame_ids if f not in labels]
    if missing and reference_model is not None:
        generated = []
        for frame_id, frame in stream_selected_frames(missing):
            result = reference_model.predict(source=preprocess(frame), classes=[0, 32], conf=0.20, iou=IOU, imgsz=1280, verbose=False, device="cuda:0", stream=False)
            players, _, balls = parse_predictions(result)
            small = [p for p in players if (p["y2"] - p["y1"]) < 60]
            labels[frame_id] = {
                "visible_players": max(len(players), 1),
                "ball_visible": bool(balls),
                "visible_small_players": len(small),
                "small_player_frame": bool(small),
            }
            generated.append({
                "frame": frame_id,
                "visible_players": labels[frame_id]["visible_players"],
                "detected_players": len(players),
                "ball_visible": labels[frame_id]["ball_visible"],
                "ball_detected": bool(balls),
                "visible_small_players": labels[frame_id]["visible_small_players"],
                "small_player_frame": labels[frame_id]["small_player_frame"],
                "label_source": "proxy_from_yolov8x_highres_not_human",
            })
            del frame, result
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        write_csv(labels_path, generated, append=labels_path.exists() and labels_path.stat().st_size > 0)
    return labels


def stream_selected_frames(frame_ids: Iterable[int]):
    wanted = set(frame_ids)
    cap = cv2.VideoCapture(str(VIDEO))
    idx = -1
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            idx += 1
            if idx in wanted:
                yield idx, frame
    finally:
        cap.release()


def batch_id_for_index(i: int) -> int:
    return i // BATCH_SIZE


def load_runtime_model(weight_name: str):
    model = YOLO(str(ROOT / weight_name))
    model.to("cuda:0" if torch.cuda.is_available() else "cpu")
    try:
        model.fuse()
    except Exception:
        pass
    if torch.cuda.is_available():
        model.model.half()
    return model


def write_progress(model_name: str, last_completed_batch: int, processed_selected_frames: int, total_selected_frames: int) -> None:
    payload = {
        "current_model": model_name,
        "last_completed_batch": last_completed_batch,
        "processed_selected_frames": processed_selected_frames,
        "total_selected_frames": total_selected_frames,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (OUT / "benchmark_progress.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def completed_frame_ids(weight_name: str) -> set:
    done = set()
    stem = Path(weight_name).stem
    for path in sorted(BATCH_DIR.glob(f"{stem}_benchmark_results_batch_*.csv")):
        with path.open("r", newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                try:
                    if row.get("status") == "ok":
                        done.add(int(float(row["frame"])))
                except Exception:
                    continue
    return done


def flush_batch(weight_name: str, batch_id: int, batch_rows: List[Dict], batch_started: float, resource_rows: List[Dict], processed_count: int, total_selected: int) -> None:
    if not batch_rows:
        return
    elapsed = time.perf_counter() - batch_started
    for row in batch_rows:
        row["batch_elapsed_sec"] = elapsed
    out_path = BATCH_DIR / f"{Path(weight_name).stem}_benchmark_results_batch_{batch_id:03d}.csv"
    write_csv(out_path, batch_rows)
    resource_rows.append(resource_row(batch_id, weight_name, "batch_end"))
    write_csv(OUT / "resource_usage.csv", resource_rows, append=(OUT / "resource_usage.csv").exists())
    write_progress(weight_name, batch_id, processed_count, total_selected)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def benchmark_model(weight_name: str, frame_ids: List[int], labels: Dict[int, Dict]) -> List[Dict]:
    selected = set(frame_ids)
    done = completed_frame_ids(weight_name)
    remaining = selected - done
    if not remaining:
        return []

    model = load_runtime_model(weight_name)
    rows = []
    batch_rows = []
    resource_rows = []
    selected_seen = len(done)
    batch_id = selected_seen // BATCH_SIZE
    batch_started = time.perf_counter()
    last_resource_time = None
    resource_rows.append(resource_row(batch_id, weight_name, "batch_start"))

    cap = cv2.VideoCapture(str(VIDEO))
    if not cap.isOpened():
        error = f"VideoCapture failed to open {VIDEO}"
        log_debug(error)
        raise RuntimeError(error)

    frame_idx = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                if frame_idx == 0:
                    log_debug(f"Video stream unusable for {VIDEO}; decoder returned no frames")
                break

            if frame_idx not in remaining:
                del frame
                frame_idx += 1
                continue

            t0 = time.perf_counter()
            try:
                processed = preprocess(frame)
                result = model.predict(source=processed, classes=[0, 32], conf=CONF, iou=IOU, imgsz=IMGSZ, verbose=False, device="cuda:0", stream=False)
                infer_sec = time.perf_counter() - t0
                players, rejected, balls = parse_predictions(result)
                label = labels.get(frame_idx, {"visible_players": max(len(players), 1), "ball_visible": bool(balls), "small_player_frame": False, "visible_small_players": 0})
                small_detected = len([p for p in players if (p["y2"] - p["y1"]) < 60])
                row = {
                    "model_name": weight_name,
                    "frame": frame_idx,
                    "detected_players": len(players),
                    "visible_players": label["visible_players"],
                    "player_recall": min(len(players) / max(label["visible_players"], 1), 1.0),
                    "false_positives": max(0, len(players) - label["visible_players"]) + len(rejected),
                    "ball_visible": label["ball_visible"],
                    "ball_detected": bool(balls),
                    "ball_recall": 1.0 if label["ball_visible"] and balls else (0.0 if label["ball_visible"] else 1.0),
                    "small_player_frame": label["small_player_frame"],
                    "visible_small_players": label["visible_small_players"],
                    "small_players_detected": small_detected,
                    "small_player_recall": min(small_detected / max(label["visible_small_players"], 1), 1.0) if label["small_player_frame"] else "",
                    "fps": 1.0 / max(infer_sec, 1e-6),
                    "inference_time_sec": infer_sec,
                    "rejected_non_players": len(rejected),
                    "mean_player_confidence": float(np.mean([p["conf"] for p in players])) if players else 0.0,
                    "status": "ok",
                }
                pred_path = PRED_DIR / f"{Path(weight_name).stem}_frame_{frame_idx:04d}.json"
                pred_path.write_text(json.dumps({"players": players, "rejected": rejected, "balls": balls}, indent=2), encoding="utf-8")
                del processed, result, players, rejected, balls
            except Exception:
                row = {
                    "model_name": weight_name,
                    "frame": frame_idx,
                    "status": "frame_failed",
                    "error_message": traceback.format_exc().replace("\n", " | "),
                }
                log_debug(f"frame failure {weight_name} frame {frame_idx}\n{traceback.format_exc()}")

            batch_rows.append(row)
            rows.append(row)
            selected_seen += 1
            if selected_seen % 50 == 0:
                resource_rows.append(resource_row(batch_id, weight_name, "selected_frame_50_interval"))
            if len(batch_rows) >= BATCH_SIZE:
                flush_batch(weight_name, batch_id, batch_rows, batch_started, resource_rows, selected_seen, len(frame_ids))
                batch_rows = []
                resource_rows = []
                batch_id += 1
                batch_started = time.perf_counter()
                resource_rows.append(resource_row(batch_id, weight_name, "batch_start"))
            del frame
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            frame_idx += 1
    finally:
        cap.release()
        flush_batch(weight_name, batch_id, batch_rows, batch_started, resource_rows, selected_seen, len(frame_ids))
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return rows

def collect_batch_rows() -> List[Dict]:
    rows = []
    for path in sorted(BATCH_DIR.glob("*_batch_*.csv")):
        with path.open("r", newline="", encoding="utf-8") as f:
            rows.extend(list(csv.DictReader(f)))
    return rows


def summarize(results: List[Dict], availability: List[Dict]) -> List[Dict]:
    rows = []
    for avail in availability:
        model = avail["model_name"]
        if avail["status"] != "AVAILABLE":
            rows.append({
                "model_name": model,
                "status": avail["status"],
                "player_recall": "",
                "false_positives_per_frame": "",
                "ball_recall": "",
                "fps": "",
                "small_player_recall": "",
                "rank_score": "",
                "error_message": avail["error_message"],
            })
            continue
        model_rows = [r for r in results if r.get("model_name") == model and r.get("status") == "ok"]
        if not model_rows:
            rows.append({"model_name": model, "status": "No Completed Rows"})
            continue
        pr = np.mean([float(r["player_recall"]) for r in model_rows])
        fp = np.mean([float(r["false_positives"]) for r in model_rows])
        br_vals = [float(r["ball_recall"]) for r in model_rows if str(r["ball_visible"]).lower() in {"true", "1"}]
        br = np.mean(br_vals) if br_vals else 0.0
        fps = np.mean([float(r["fps"]) for r in model_rows])
        spr_vals = [float(r["small_player_recall"]) for r in model_rows if str(r.get("small_player_recall", "")) not in {"", "nan"}]
        spr = np.mean(spr_vals) if spr_vals else 0.0
        score = pr * 0.45 + br * 0.35 + min(fps / 20, 1) * 0.15 - min(fp / 10, 1) * 0.05
        rows.append({
            "model_name": model,
            "status": "Completed",
            "player_recall": pr,
            "false_positives_per_frame": fp,
            "ball_recall": br,
            "fps": fps,
            "small_player_recall": spr,
            "rank_score": score,
            "error_message": "",
        })
    return sorted(rows, key=lambda r: float(r["rank_score"]) if str(r.get("rank_score", "")) else -1, reverse=True)


def validation_summary(results: List[Dict], best_model: str) -> List[Dict]:
    rows = [r for r in results if r.get("model_name") == best_model and r.get("status") == "ok"]
    manual_ids = set(manual_frames(video_frame_count(), 50))
    rows = [r for r in rows if int(float(r["frame"])) in manual_ids]
    if not rows:
        return [{"model_name": best_model, "status": "no_validation_rows"}]
    small = [r for r in rows if str(r.get("small_player_recall", "")) not in {"", "nan"}]
    ball = [r for r in rows if str(r.get("ball_visible")).lower() in {"true", "1"}]
    return [{
        "model_name": best_model,
        "sampled_frames": len(rows),
        "visible_players": sum(int(float(r["visible_players"])) for r in rows),
        "detected_players": sum(int(float(r["detected_players"])) for r in rows),
        "player_recall": np.mean([float(r["player_recall"]) for r in rows]),
        "ball_visible_frames": len(ball),
        "ball_detected_frames": sum(1 for r in ball if str(r["ball_detected"]).lower() in {"true", "1"}),
        "ball_recall": np.mean([float(r["ball_recall"]) for r in ball]) if ball else 0.0,
        "small_player_recall": np.mean([float(r["small_player_recall"]) for r in small]) if small else 0.0,
        "label_source": "manual_validation.csv if present; otherwise proxy_from_yolov8x_highres_not_human",
    }]


def draw_plot(summary: List[Dict]) -> None:
    img = np.full((720, 1100, 3), 255, dtype=np.uint8)
    cv2.putText(img, "Robust Detector Benchmark", (40, 55), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2)
    y = 120
    for row in summary:
        name = row["model_name"]
        cv2.putText(img, name, (35, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        if row.get("status") != "Completed":
            cv2.putText(img, row.get("status", ""), (220, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 180), 2)
            y += 95
            continue
        metrics = [("player", float(row["player_recall"]), (40, 90, 230)), ("ball", float(row["ball_recall"]), (30, 170, 80)), ("fps/20", min(float(row["fps"]) / 20, 1), (220, 130, 30))]
        for i, (label, val, color) in enumerate(metrics):
            x0 = 220
            yy = y + i * 25
            cv2.rectangle(img, (x0, yy), (x0 + int(700 * val), yy + 18), color, -1)
            cv2.putText(img, f"{label}: {val:.3f}", (940, yy + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 0), 1)
        y += 115
    cv2.imwrite(str(OUT / "comparison_plot.png"), img)


def report(summary: List[Dict], availability: List[Dict]) -> None:
    available = [r["model_name"] for r in availability if r["status"] == "AVAILABLE"]
    failed = [f"{r['model_name']}: {r['error_message']}" for r in availability if r["status"] != "AVAILABLE"]
    best = next((r["model_name"] for r in summary if r.get("status") == "Completed"), "None")
    lines = [
        "# Robust Detector Benchmark Report",
        "",
        f"Input: `{VIDEO}`",
        f"Available models: {', '.join(available) if available else 'None'}",
        "Failed/unavailable models:",
        *(f"- {x}" for x in failed),
        "",
        f"Benchmark status: {'completed' if any(r.get('status') == 'Completed' for r in summary) else 'not completed'}",
        "Memory safety: streaming frames, batch CSV writes, CUDA cache clearing, and resource logging are enabled.",
        f"Best currently usable detector: {best}",
        f"YOLO11 comparison possible: {'yes' if any(m.startswith('yolo11') for m in available) else 'no'}",
    ]
    (OUT / "robust_benchmark_report.md").write_text("\n".join(lines), encoding="utf-8")


def video_diagnostics() -> Dict:
    cap = cv2.VideoCapture(str(VIDEO))
    row = {
        "video_path": str(VIDEO),
        "opened": bool(cap.isOpened()),
        "frame_count": 0,
        "fps": 0.0,
        "width": 0,
        "height": 0,
        "sample_frames_read": 0,
        "decoder_stable": False,
        "diagnostic_message": "",
    }
    if not cap.isOpened():
        row["diagnostic_message"] = "VideoCapture could not open source. If this persists, remux/transcode with ffmpeg: ffmpeg -i input.mp4 -c:v libx264 -c:a aac cleaned_input.mp4"
        write_csv(OUT / "video_diagnostics.csv", [row])
        cap.release()
        return row
    row["frame_count"] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    row["fps"] = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    row["width"] = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    row["height"] = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    read_ok = 0
    for _ in range(30):
        ret, frame = cap.read()
        if not ret:
            break
        read_ok += 1
        del frame
    row["sample_frames_read"] = read_ok
    row["decoder_stable"] = read_ok > 0
    if not row["decoder_stable"]:
        row["diagnostic_message"] = "Decoder returned no sample frames. Consider creating a cleaned benchmark copy with ffmpeg."
    cap.release()
    write_csv(OUT / "video_diagnostics.csv", [row])
    return row

def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    BATCH_DIR.mkdir(parents=True, exist_ok=True)
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    if DEBUG_LOG.exists():
        DEBUG_LOG.unlink()
    if not torch.cuda.is_available():
        log_debug("CUDA unavailable; running on CPU would be slower.")

    total = video_frame_count()
    frame_ids = representative_frames(total, 100)
    validation_ids = manual_frames(total, 50)

    availability = []
    for name in MODEL_NAMES:
        availability.append(validate_model(name))
    write_csv(OUT / "model_availability.csv", availability)

    valid_models = [r["model_name"] for r in availability if r.get("status") == "AVAILABLE"]
    if not valid_models:
        log_debug("No valid detector weights available; benchmark cannot run.")
        summary = summarize([], availability)
        write_csv(OUT / "benchmark_results.csv", [])
        write_csv(OUT / "benchmark_summary.csv", summary)
        write_csv(OUT / "validation_summary.csv", [])
        draw_plot(summary)
        report(summary, availability)
        return 1

    reference_name = "yolov8x.pt" if "yolov8x.pt" in valid_models else valid_models[0]
    reference_model = load_runtime_model(reference_name)
    labels = load_manual_labels(sorted(set(frame_ids + validation_ids)), reference_model)
    del reference_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    for name in valid_models:
        benchmark_model(name, frame_ids, labels)

    merged = collect_batch_rows()
    write_csv(OUT / "benchmark_results.csv", merged)
    summary = summarize(merged, availability)
    write_csv(OUT / "benchmark_summary.csv", summary)
    best = next((r["model_name"] for r in summary if r.get("status") == "Completed"), "")
    write_csv(OUT / "validation_summary.csv", validation_summary(merged, best) if best else [])
    draw_plot(summary)
    report(summary, availability)
    print(json.dumps({"available": valid_models, "best": best, "output_dir": str(OUT)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())






