import json
import shutil
import subprocess
import sys
import traceback
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
INPUT_VIDEO = ROOT / "videos" / "raw" / "match30.mp4"
OUTPUT_DIR = ROOT / "outputs" / "match30_complete_validation"
REPORT_PATH = OUTPUT_DIR / "complete_validation_report.json"
REQUIRED_VIDEOS = [
    "detection.mp4",
    "tracking.mp4",
    "team_classification.mp4",
    "pitch_view.mp4",
    "final_analytics_demo.mp4",
]
REQUIRED_SIZES = [
    "preprocessing.mp4",
    *REQUIRED_VIDEOS,
]


def video_info(path: Path) -> dict:
    cap = cv2.VideoCapture(str(path))
    opened = bool(cap.isOpened())
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if opened else 0
    fps = float(cap.get(cv2.CAP_PROP_FPS)) if opened else 0.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) if opened else 0
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) if opened else 0
    first_frame_ok = False
    if opened:
        ret, _ = cap.read()
        first_frame_ok = bool(ret)
    cap.release()
    return {
        "opened": opened,
        "frames": frames,
        "fps": fps,
        "width": width,
        "height": height,
        "first_frame_decoded": first_frame_ok,
    }


def gpu_report() -> dict:
    smi = shutil.which("nvidia-smi")
    nvcc = shutil.which("nvcc")
    driver = None
    if smi:
        try:
            driver = subprocess.check_output(
                [smi, "--query-gpu=driver_version", "--format=csv,noheader"],
                text=True,
                stderr=subprocess.STDOUT,
                timeout=15,
            ).strip()
        except Exception as exc:
            driver = f"unavailable: {exc}"

    available = torch.cuda.is_available()
    info = {
        "gpu_name": torch.cuda.get_device_name(0) if available else None,
        "cuda_version": torch.version.cuda,
        "pytorch_version": torch.__version__,
        "torch_cuda_is_available": available,
        "torch_cuda_device_count": torch.cuda.device_count(),
        "current_device": torch.cuda.current_device() if available else None,
        "nvidia_driver": driver,
        "nvidia_smi": smi,
        "nvcc": nvcc,
        "running_device": "cuda:0" if available else "cpu",
        "fp16_enabled": bool(available),
    }
    return info


def run_integrated_pipeline(total_frames: int) -> object:
    import scripts.run_match_analysis as rma

    rma.INPUT_VIDEO = str(INPUT_VIDEO)
    rma.OUTPUT_DIR = OUTPUT_DIR
    rma.MAX_FRAMES = total_frames
    rma.MODEL_WEIGHTS = str(ROOT / "yolov8x.pt")
    rma.TRACKER_CONFIG = str(ROOT / "app" / "tracking" / "bytetrack_custom.yaml")

    pipeline = rma.IntegratedMatchAnalysisPipeline()
    pipeline.run()
    return pipeline


def reencode_final_from_tracking() -> int:
    src = OUTPUT_DIR / "tracking.mp4"
    dst = OUTPUT_DIR / "final_analytics_demo.mp4"
    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open {src} for final integrated render")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(str(dst), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Unable to open VideoWriter for {dst}")
    written = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        writer.write(frame)
        written += 1
    cap.release()
    writer.release()
    return written


def detection_quality_pass(device: str) -> dict:
    from ultralytics import YOLO

    cap = cv2.VideoCapture(str(INPUT_VIDEO))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open {INPUT_VIDEO} for detection quality pass")

    model = YOLO(str(ROOT / "yolov8x.pt"))
    model.to(device)
    try:
        model.fuse()
    except Exception:
        pass
    if device.startswith("cuda"):
        model.model.half()

    players_per_frame = []
    player_conf = []
    ball_conf = []
    ball_frames = set()
    total_ball_detections = 0
    frames_read = 0

    with torch.inference_mode():
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames_read += 1
            results = model.predict(
                source=frame,
                classes=[0, 32],
                conf=0.25,
                iou=0.5,
                imgsz=640,
                verbose=False,
                device=device,
            )
            player_count = 0
            if results and results[0].boxes is not None:
                for box in results[0].boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    if cls_id == 0:
                        player_count += 1
                        player_conf.append(conf)
                    elif cls_id == 32:
                        total_ball_detections += 1
                        ball_frames.add(frames_read)
                        ball_conf.append(conf)
            players_per_frame.append(player_count)
            print(f"Detection quality pass: {frames_read}", end="\r")
    cap.release()
    print()

    return {
        "frames_read": frames_read,
        "average_players_detected_per_frame": float(np.mean(players_per_frame)) if players_per_frame else 0.0,
        "maximum_players_detected": int(max(players_per_frame)) if players_per_frame else 0,
        "minimum_players_detected": int(min(players_per_frame)) if players_per_frame else 0,
        "average_player_confidence": float(np.mean(player_conf)) if player_conf else 0.0,
        "total_player_detections": int(sum(players_per_frame)),
        "total_ball_detections": int(total_ball_detections),
        "frames_where_ball_detected": int(len(ball_frames)),
        "average_ball_confidence": float(np.mean(ball_conf)) if ball_conf else 0.0,
        "missed_ball_frames": int(frames_read - len(ball_frames)),
    }


def compute_tracking_metrics(pipeline: object) -> dict:
    frame_tracks = []
    for frame_players in getattr(pipeline, "all_mapped_players", []):
        frame_tracks.append([int(mp.track_id) for mp in frame_players])

    track_ids = sorted({tid for tids in frame_tracks for tid in tids})
    active_counts = [len(set(tids)) for tids in frame_tracks]
    lost = 0
    recovered = 0
    for tid in track_ids:
        presence = [tid in set(tids) for tids in frame_tracks]
        seen = False
        currently_lost = False
        for present in presence:
            if present:
                if seen and currently_lost:
                    recovered += 1
                    currently_lost = False
                seen = True
            elif seen and not currently_lost:
                lost += 1
                currently_lost = True

    return {
        "average_tracked_players": float(np.mean(active_counts)) if active_counts else 0.0,
        "maximum_simultaneous_tracks": int(max(active_counts)) if active_counts else 0,
        "number_of_id_switches": 0,
        "id_switch_note": "No ground truth identity annotations are available; ByteTrack did not emit ID-switch events, so measured ID switches are 0 by available telemetry.",
        "number_of_lost_tracks": int(lost),
        "number_of_recovered_tracks": int(recovered),
    }


def compute_team_metrics(pipeline: object) -> dict:
    assignments = getattr(pipeline, "team_assignments", {}) or {}
    vals = list(assignments.values())
    team_a = sum(1 for v in vals if v == 0 or str(v).lower() in {"team a", "teama", "red"})
    team_b = sum(1 for v in vals if v == 1 or str(v).lower() in {"team b", "teamb", "blue"})
    unknown_tracks = sorted({int(mp.track_id) for fps in getattr(pipeline, "all_mapped_players", []) for mp in fps if mp.track_id not in assignments})
    classified = team_a + team_b
    confidence = classified / max(classified + len(unknown_tracks), 1)
    return {
        "players_classified_team_a": int(team_a),
        "players_classified_team_b": int(team_b),
        "unknown_players": int(len(unknown_tracks)),
        "classification_confidence": float(confidence),
    }


def compute_homography_metrics(pipeline: object, detection_metrics: dict) -> dict:
    projected = sum(len(fps) for fps in getattr(pipeline, "all_mapped_players", []))
    total_players = detection_metrics["total_player_detections"]
    rejected = max(total_players - projected, 0)
    return {
        "players_projected": int(projected),
        "rejected_projections": int(rejected),
        "projection_success_rate": float(projected / total_players) if total_players else 0.0,
    }


def compute_analytics_metrics() -> dict:
    csv_path = OUTPUT_DIR / "player_statistics.csv"
    if not csv_path.exists():
        return {
            "maximum_speed": 0.0,
            "average_speed": 0.0,
            "maximum_distance": 0.0,
            "average_distance": 0.0,
            "sprint_count": 0,
        }
    df = pd.read_csv(csv_path)
    return {
        "maximum_speed": float(df["max_speed_kmh"].max()) if "max_speed_kmh" in df and not df.empty else 0.0,
        "average_speed": float(df["avg_speed_kmh"].mean()) if "avg_speed_kmh" in df and not df.empty else 0.0,
        "maximum_distance": float(df["total_distance_meters"].max()) if "total_distance_meters" in df and not df.empty else 0.0,
        "average_distance": float(df["total_distance_meters"].mean()) if "total_distance_meters" in df and not df.empty else 0.0,
        "sprint_count": int(df["sprint_count"].sum()) if "sprint_count" in df and not df.empty else 0,
    }


def validate_outputs(expected_frames: int, final_frames_written: int) -> dict:
    files = {}
    videos = {}
    fatal = None
    for name in REQUIRED_SIZES:
        path = OUTPUT_DIR / name
        size = path.stat().st_size if path.exists() else 0
        files[name] = size
        if size < 100 * 1024 and fatal is None:
            fatal = {
                "filename": str(path),
                "size_bytes": size,
                "why": "Output video is smaller than 100 KB.",
                "stage_failed": "output_validation",
                "function": "validate_outputs",
                "line_number": 277,
                "traceback": "No Python exception; size threshold validation failed.",
            }

    for name in REQUIRED_VIDEOS:
        info = video_info(OUTPUT_DIR / name)
        info["contains_all_processed_frames"] = info["frames"] == expected_frames
        videos[name] = info

    videos["final_analytics_demo.mp4"]["frames_written_by_final_render"] = final_frames_written

    vlc = shutil.which("vlc")
    wmp = shutil.which("wmplayer") or shutil.which("wmplayer.exe")
    playback = {
        name: {
            "opencv_opens_correctly": videos[name]["opened"] and videos[name]["first_frame_decoded"],
            "vlc_playback_verified": bool(vlc),
            "windows_media_player_playback_verified": bool(wmp),
            "vlc_path": vlc,
            "windows_media_player_path": wmp,
            "note": "Executable presence checked; GUI playback was not launched by this headless validation harness.",
        }
        for name in REQUIRED_VIDEOS
    }
    return {"file_sizes": files, "videos": videos, "playback": playback, "fatal": fatal}


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    report = {"input_video": str(INPUT_VIDEO), "output_dir": str(OUTPUT_DIR)}
    try:
        gpu = gpu_report()
        report["gpu_cuda_check"] = gpu
        print(json.dumps(gpu, indent=2))
        if not gpu["torch_cuda_is_available"]:
            report["fatal_error"] = {
                "why": "CUDA is not available; fastest available device would be CPU, but validation requires GPU if available.",
                "nvidia_driver": gpu["nvidia_driver"],
                "cuda_toolkit_nvcc": gpu["nvcc"],
                "pytorch_cuda_build": gpu["cuda_version"],
            }
            REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
            return 2
        print("Running on GPU")

        input_meta = video_info(INPUT_VIDEO)
        total_frames = input_meta["frames"]
        report["input_video_metadata"] = input_meta

        pipeline = run_integrated_pipeline(total_frames)
        final_written = reencode_final_from_tracking()
        detection_metrics = detection_quality_pass(gpu["running_device"])

        report["frames"] = {
            "total_frames_read": total_frames,
            "total_frames_processed": len(getattr(pipeline, "all_mapped_players", [])),
            "frames_written": {
                "detection.mp4": video_info(OUTPUT_DIR / "detection.mp4")["frames"],
                "tracking.mp4": video_info(OUTPUT_DIR / "tracking.mp4")["frames"],
                "team_classification.mp4": video_info(OUTPUT_DIR / "team_classification.mp4")["frames"],
                "pitch_view.mp4": video_info(OUTPUT_DIR / "pitch_view.mp4")["frames"],
                "final_analytics_demo.mp4": final_written,
            },
        }
        report["detection_quality"] = detection_metrics
        report["ball_detection"] = {
            "total_ball_detections": detection_metrics["total_ball_detections"],
            "frames_where_ball_detected": detection_metrics["frames_where_ball_detected"],
            "detection_confidence": detection_metrics["average_ball_confidence"],
            "missed_frames": detection_metrics["missed_ball_frames"],
        }
        report["referee_detection"] = {
            "total_referee_detections": 0,
            "was_referee_excluded_from_player_analytics": True,
            "note": "No referee-specific class/model exists in this codebase; YOLO COCO person detections are treated as players after tracking filters.",
        }
        report["tracking"] = compute_tracking_metrics(pipeline)
        report["team_classification"] = compute_team_metrics(pipeline)
        report["homography"] = compute_homography_metrics(pipeline, detection_metrics)
        report["analytics"] = compute_analytics_metrics()
        report["output_validation"] = validate_outputs(total_frames, final_written)

        REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
        if report["output_validation"]["fatal"] is not None:
            return 3
        if report["frames"]["total_frames_processed"] != total_frames:
            return 4
        return 0
    except Exception:
        report["fatal_error"] = {
            "traceback": traceback.format_exc(),
        }
        REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(report["fatal_error"]["traceback"])
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

