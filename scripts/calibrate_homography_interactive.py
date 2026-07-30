"""Interactive homography calibration tool.

Usage:
    python scripts/calibrate_homography_interactive.py --video path/to/video.mp4 [--calibration configs/homography_calibration.json]

Controls:
    Left click        Add calibration point
    Backspace / U     Undo last point
    R                 Reset all points
    S                 Save calibration
    Q / Escape        Quit
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import cv2
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("HomographyCalibration")

DEFAULT_CALIBRATION = Path("configs/homography_calibration.json")
DEFAULT_VIDEO = Path("videos/raw/match30.mp4")


def load_calibration(path: Path) -> dict:
    if path.exists():
        with open(path, "r") as f:
            return json.load(f)
    return {
        "field_dimensions": {"length_m": 105.0, "width_m": 68.0},
        "image_dimensions": {"width_px": 1050, "height_px": 680},
        "calibration_points": {"source": [], "destination": []},
        "validation": {"mean_reprojection_error": 1.5, "m_per_px_cv": 0.1, "validation_passed": False},
        "method": "manual",
        "note": "Calibration points must be verified against actual video.",
    }


def save_calibration(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    logger.info(f"Saved calibration to {path}")


def compute_homography(source: np.ndarray, destination: np.ndarray):
    if source.shape[0] < 4 or destination.shape[0] < 4:
        return None, None, None
    H, mask = cv2.findHomography(source, destination, cv2.RANSAC, ransacReprojThreshold=3.0, maxIters=2000, confidence=0.99)
    if H is None:
        return None, None, None
    inliers = mask.ravel() == 1
    reprojected = cv2.perspectiveTransform(source[inliers].reshape(-1, 1, 2), H)
    errors = np.linalg.norm(destination[inliers] - reprojected.reshape(-1, 2), axis=1)
    return H, inliers, errors


def draw_pitch_overlay(image: np.ndarray, H: np.ndarray, pitch_m: tuple) -> np.ndarray:
    length_m, width_m = pitch_m
    h_img, w_img = image.shape[:2]

    # Simple pitch rectangle in meters
    pts = np.array(
        [
            [0, 0],
            [length_m, 0],
            [length_m, width_m],
            [0, width_m],
        ],
        dtype=np.float32,
    )

    if H is not None:
        transformed = cv2.perspectiveTransform(pts.reshape(-1, 1, 2), H).reshape(-1, 2)
        overlay = image.copy()
        cv2.fillPoly(overlay, [transformed.astype(int)], (0, 255, 0))
        alpha = 0.25
        cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)
        cv2.polylines(image, [transformed.astype(int)], True, (0, 255, 0), 2)
    return image


def main():
    parser = argparse.ArgumentParser(description="Interactive homography calibration")
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO, help="Path to video file")
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION, help="Calibration JSON path")
    args = parser.parse_args()

    calib = load_calibration(args.calibration)
    pitch_m = (calib["field_dimensions"]["length_m"], calib["field_dimensions"]["width_m"])

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        logger.error(f"Failed to open video: {args.video}")
        sys.exit(1)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        logger.error("Failed to read frame")
        sys.exit(1)

    image = frame.copy()
    source_points = np.array(calib["calibration_points"].get("source", []), dtype=np.float32)
    dest_points = np.array(calib["calibration_points"].get("destination", []), dtype=np.float32)
    if source_points.shape[0] >= 4 and dest_points.shape[0] >= 4:
        H, _, _ = compute_homography(source_points, dest_points)
    else:
        H = None

    window = "Homography Calibration"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            nonlocal source_points, dest_points, H
            new_point = np.array([[x, y]], dtype=np.float32)
            if source_points.size == 0:
                source_points = new_point
                dest_points = new_point
            else:
                source_points = np.vstack([source_points, new_point])
                dest_points = np.vstack([dest_points, new_point])
            H, _, _ = compute_homography(source_points, dest_points)

    cv2.setMouseCallback(window, mouse_callback)

    while True:
        display = image.copy()
        if H is not None:
            display = draw_pitch_overlay(display, H, pitch_m)

        for i, p in enumerate(source_points):
            cv2.circle(display, (int(p[0]), int(p[1])), 6, (0, 0, 255), -1)
            cv2.circle(display, (int(p[0]), int(p[1])), 8, (255, 255, 255), 2)
            cv2.putText(display, str(i + 1), (int(p[0]) + 10, int(p[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        status = f"Points: {len(source_points)}/4+"
        cv2.putText(display, status, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        if H is not None and source_points.shape[0] >= 4:
            _, _, errors = compute_homography(source_points, dest_points)
            if errors is not None and errors.size:
                err_text = f"Reproj error: {errors.mean():.2f} px"
                cv2.putText(display, err_text, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        cv2.imshow(window, display)
        key = cv2.waitKey(20) & 0xFF
        if key in (ord('q'), 27):
            break
        elif key in (ord('r'),):
            source_points = np.empty((0, 2), dtype=np.float32)
            dest_points = np.empty((0, 2), dtype=np.float32)
            H = None
        elif key in (ord('u'), 8):
            if source_points.size:
                source_points = source_points[:-1]
                dest_points = dest_points[:-1]
                H, _, _ = compute_homography(source_points, dest_points)
        elif key in (ord('s'),):
            if source_points.shape[0] < 4:
                logger.warning("Need at least 4 points to save")
                continue
            calib["calibration_points"]["source"] = source_points.tolist()
            calib["calibration_points"]["destination"] = dest_points.tolist()
            save_calibration(args.calibration, calib)
            logger.info("Calibration saved")

    cv2.destroyWindow(window)


if __name__ == "__main__":
    main()