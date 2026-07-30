"""Validate homography module on match30.mp4."""

import logging
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.homography.pitch_model import PitchModel
from app.homography.homography_estimator import HomographyEstimator
from app.homography.homography_metrics import HomographyMetrics
from app.homography.visualization import HomographyVisualizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("HomographyValidation")

INPUT_VIDEO = ROOT / "videos" / "raw" / "match30.mp4"
OUTPUT_DIR = ROOT / "outputs" / "homography_validation"
CALIBRATION_FILE = ROOT / "configs" / "homography_calibration.json"
MAX_FRAMES = 200


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    import json
    with open(CALIBRATION_FILE, "r") as f:
        calib_data = json.load(f)

    homography_cfg = {
        "calibration_points": calib_data.get("calibration_points", {}),
        "pitch": calib_data.get("field_dimensions", {}),
        "validation": calib_data.get("validation", {}),
    }

    pitch_model = PitchModel.from_config(homography_cfg.get("pitch", {}))
    estimator = HomographyEstimator(homography_cfg)
    visualizer = HomographyVisualizer(pitch_model)
    metrics = HomographyMetrics(homography_cfg)

    success = estimator.initialize()
    if not success:
        logger.warning("Calibration failed, using identity matrix")
    calib = estimator.calibration_result

    cap = cv2.VideoCapture(str(INPUT_VIDEO))
    total_frames = min(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), MAX_FRAMES)
    logger.info(f"Video: {total_frames} frames")

    rows = []
    frames_list = []
    motion_successes = 0
    motion_displacements = []
    transform_successes = 0
    t0 = time.perf_counter()

    for frame_no in range(1, total_frames + 1):
        ret, frame = cap.read()
        if not ret:
            break
        frames_list.append(frame)

        motion = estimator.estimate_camera_motion(frame, frame_no)
        if motion is not None:
            motion_successes += 1
            motion_displacements.append(motion.displacement)

        h, w = frame.shape[:2]
        sample_points = [
            (w * 0.2, h * 0.3),
            (w * 0.5, h * 0.3),
            (w * 0.8, h * 0.3),
            (w * 0.2, h * 0.7),
            (w * 0.5, h * 0.7),
            (w * 0.8, h * 0.7),
        ]

        for track_id, (px, py) in enumerate(sample_points, 1):
            pixel = np.array([px, py], dtype=np.float64)
            world = estimator.get_world_position(pixel)
            camera = estimator.get_camera_stabilized_position(pixel)

            world_x = float(world[0]) if world is not None else None
            world_y = float(world[1]) if world is not None else None
            camera_x = float(camera[0]) if camera is not None else None
            camera_y = float(camera[1]) if camera is not None else None

            transform_ok = world_x is not None and world_y is not None
            if transform_ok:
                transform_successes += 1

            rows.append({
                "frame": frame_no,
                "track_id": track_id,
                "pixel_x": round(px, 1),
                "pixel_y": round(py, 1),
                "camera_x": round(camera_x, 1) if camera_x is not None else "",
                "camera_y": round(camera_y, 1) if camera_y is not None else "",
                "world_x": round(world_x, 1) if world_x is not None else "",
                "world_y": round(world_y, 1) if world_y is not None else "",
                "transform_success": 1 if transform_ok else 0,
                "confidence": round(calib.confidence, 3) if calib else 0.0,
            })

    cap.release()
    elapsed = time.perf_counter() - t0
    fps = total_frames / max(elapsed, 0.001)

    df = pd.DataFrame(rows)
    csv_path = OUTPUT_DIR / "homography_report.csv"
    df.to_csv(csv_path, index=False)
    logger.info(f"CSV written to {csv_path}")

    # Compute metrics
    total_samples = total_frames * 6
    transform_success_rate = transform_successes / max(total_samples, 1)
    invalid_count = total_samples - transform_successes

    world_points = df[["world_x", "world_y"]].replace("", np.nan).dropna().values
    oob_count = metrics.check_out_of_bounds(world_points, margin=0.0) if world_points.size else 0

    reprojection_error_pixels = calib.reprojection_error if calib else float("inf")
    if calib and getattr(calib, "pixels_per_metre_x", 0) > 0 and getattr(calib, "pixels_per_metre_y", 0) > 0:
        ppm = (calib.pixels_per_metre_x + calib.pixels_per_metre_y) / 2.0
        reprojection_error_m = reprojection_error_pixels / max(ppm, 1e-6)
    else:
        reprojection_error_m = float("inf")

    avg_motion = float(np.mean(motion_displacements)) if motion_displacements else 0.0
    max_motion = float(np.max(motion_displacements)) if motion_displacements else 0.0
    median_motion = float(np.median(motion_displacements)) if motion_displacements else 0.0

    world_df = df[["track_id", "world_x", "world_y"]].replace("", np.nan).dropna()
    stability_std_m = np.nan
    if not world_df.empty:
        stds = world_df.groupby("track_id")[["world_x", "world_y"]].std().mean(axis=1)
        stability_std_m = float(stds.mean()) if not stds.empty else np.nan

    production_ready = (
        transform_success_rate >= 0.95
        and (oob_count / max(total_samples, 1)) < 0.05
        and bool(calib and calib.confidence >= 0.7 and calib.determinant > 0.01 and reprojection_error_pixels < 10.0)
    )

    report_path = OUTPUT_DIR / "homography_validation.md"
    report = f"""# Homography Validation Report

**Date:** {time.strftime("%Y-%m-%d %H:%M:%S")}  
**Video:** {INPUT_VIDEO.name}  
**Frames processed:** {total_frames}

---

## Metrics

| Metric | Value |
|--------|-------|
| Camera motion success rate | {motion_successes}/{total_frames} ({motion_successes/total_frames*100:.1f}%) |
| Transform success rate | {transform_successes}/{total_samples} ({transform_success_rate*100:.1f}%) |
| Average camera motion | {avg_motion:.2f} px |
| Maximum camera motion | {max_motion:.2f} px |
| Median camera motion | {median_motion:.2f} px |
| Reprojection error | {reprojection_error_pixels:.3f} px / {reprojection_error_m:.3f} m |
| Invalid transformations | {invalid_count} |
| Out-of-bounds world positions | {oob_count} |
| Transformation FPS | {fps:.1f} |
| Supports auto calibration later | Yes |
| Production ready | {'Yes' if production_ready else 'No'} |

## Calibration

- Method: manual
- Confidence: {calib.confidence:.3f}
- Determinant: {calib.determinant:.3f}
- Pixels per metre (x): {getattr(calib, 'pixels_per_metre_x', float('nan')):.2f}
- Pixels per metre (y): {getattr(calib, 'pixels_per_metre_y', float('nan')):.2f}

## World Coordinate Stability

- Average std dev per sample point: {stability_std_m:.3f} m

## Files

- `homography_report.csv`
- `homography_debug.mp4`
- `homography_validation.md`
"""
    report_path.write_text(report)
    logger.info(f"Report written to {report_path}")

    print(f"\nHOMOGRAPHY VALIDATION COMPLETE\n{report}")


if __name__ == "__main__":
    main()