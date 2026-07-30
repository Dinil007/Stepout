"""
Homography Utilities Module

Provides reusable, robust mathematical and computer vision utilities for computing,
validating, transforming, and visualizing homography perspective projections between 
broadcast video frame pixel coordinates and 2D football pitch coordinates.
"""

import logging
from typing import List, Tuple, Union, Optional
import cv2
import numpy as np

from app.homography.field_config import PITCH_IMAGE_WIDTH, PITCH_IMAGE_HEIGHT

# Configure logger for homography utilities
logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)


def validate_points(
    source_points: Union[List[Tuple[float, float]], np.ndarray],
    destination_points: Union[List[Tuple[float, float]], np.ndarray]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Validates input source and destination point sets for homography calculation.

    Args:
        source_points: Correspondences in source image space (N x 2).
        destination_points: Correspondences in target pitch space (N x 2).

    Returns:
        Tuple of (source_pts_array, destination_pts_array) as float32 NumPy arrays.

    Raises:
        TypeError: If point structures are not list, tuple, or numpy array.
        ValueError: If point counts mismatch or fewer than 4 point pairs are provided.
    """
    if source_points is None or destination_points is None:
        raise ValueError("Source points and destination points cannot be None.")

    try:
        src_arr = np.array(source_points, dtype=np.float32)
        dst_arr = np.array(destination_points, dtype=np.float32)
    except Exception as e:
        raise TypeError(f"Failed to convert point inputs to numeric NumPy arrays: {e}")

    if src_arr.ndim != 2 or src_arr.shape[1] != 2:
        raise ValueError(f"Source points must have shape (N, 2), got shape {src_arr.shape}.")

    if dst_arr.ndim != 2 or dst_arr.shape[1] != 2:
        raise ValueError(f"Destination points must have shape (N, 2), got shape {dst_arr.shape}.")

    if src_arr.shape[0] != dst_arr.shape[0]:
        raise ValueError(
            f"Point count mismatch: source has {src_arr.shape[0]} points, "
            f"destination has {dst_arr.shape[0]} points."
        )

    if src_arr.shape[0] < 4:
        raise ValueError(
            f"Homography calculation requires at least 4 point pairs. Provided: {src_arr.shape[0]}."
        )

    return src_arr, dst_arr


def compute_homography(
    source_points: Union[List[Tuple[float, float]], np.ndarray],
    destination_points: Union[List[Tuple[float, float]], np.ndarray],
    method: int = cv2.RANSAC,
    ransac_reproj_threshold: float = 5.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes the 3x3 Homography Matrix mapping source points to target destination points.

    Args:
        source_points: Source image coordinates (N x 2).
        destination_points: Destination pitch coordinates (N x 2).
        method: OpenCV homography method (e.g., cv2.RANSAC, cv2.LMEDS, 0).
        ransac_reproj_threshold: Maximum allowed reprojection error in RANSAC.

    Returns:
        Tuple of (homography_matrix, status_mask).

    Raises:
        RuntimeError: If OpenCV fails to compute a valid homography matrix.
    """
    src_pts, dst_pts = validate_points(source_points, destination_points)

    # Format points for cv2.findHomography (shape: N x 1 x 2)
    src_pts_reshaped = src_pts.reshape(-1, 1, 2)
    dst_pts_reshaped = dst_pts.reshape(-1, 1, 2)

    homography_matrix, mask = cv2.findHomography(
        src_pts_reshaped,
        dst_pts_reshaped,
        method=method,
        ransacReprojThreshold=ransac_reproj_threshold
    )

    if homography_matrix is None or homography_matrix.shape != (3, 3):
        logger.error("cv2.findHomography returned None or invalid matrix shape.")
        raise RuntimeError("Failed to compute valid 3x3 Homography Matrix.")

    logger.info("Homography matrix computed successfully.")
    return homography_matrix, mask


def transform_point(
    point: Tuple[float, float],
    homography_matrix: np.ndarray
) -> Tuple[float, float]:
    """
    Transforms a single (x, y) point from source space to target destination space using a 3x3 Homography.

    Args:
        point: Source coordinate (x, y).
        homography_matrix: 3x3 Homography transformation matrix.

    Returns:
        Transformed coordinate (x_prime, y_prime).

    Raises:
        ValueError: If input point or matrix format is invalid.
    """
    if homography_matrix is None or homography_matrix.shape != (3, 3):
        raise ValueError("Invalid 3x3 Homography matrix provided.")

    pt_arr = np.array([[[float(point[0]), float(point[1])]]], dtype=np.float32)
    transformed = cv2.perspectiveTransform(pt_arr, homography_matrix)

    res_x = float(transformed[0, 0, 0])
    res_y = float(transformed[0, 0, 1])

    return res_x, res_y


def transform_points(
    points: Union[List[Tuple[float, float]], np.ndarray],
    homography_matrix: np.ndarray
) -> List[Tuple[float, float]]:
    """
    Transforms multiple (x, y) points in batch using a 3x3 Homography matrix.

    Args:
        points: List or N x 2 array of source coordinates.
        homography_matrix: 3x3 Homography matrix.

    Returns:
        List of transformed (x_prime, y_prime) coordinate tuples.

    Raises:
        ValueError: If input points or matrix are invalid.
    """
    if homography_matrix is None or homography_matrix.shape != (3, 3):
        raise ValueError("Invalid 3x3 Homography matrix provided.")

    if len(points) == 0:
        return []

    pts_arr = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
    transformed = cv2.perspectiveTransform(pts_arr, homography_matrix)

    transformed_points = [
        (float(pt[0, 0]), float(pt[0, 1])) for pt in transformed
    ]

    return transformed_points


def inverse_transform(
    point: Tuple[float, float],
    homography_matrix: np.ndarray
) -> Tuple[float, float]:
    """
    Transforms a target pitch space coordinate back to image pixel space by computing H inverse.

    Args:
        point: Target coordinate in pitch space (x, y).
        homography_matrix: 3x3 Homography matrix (H).

    Returns:
        Original source coordinate in image pixel space (x_img, y_img).

    Raises:
        ValueError: If matrix is not invertible or invalid.
    """
    if homography_matrix is None or homography_matrix.shape != (3, 3):
        raise ValueError("Invalid 3x3 Homography matrix provided.")

    try:
        inv_h = np.linalg.inv(homography_matrix)
    except np.linalg.LinAlgError as e:
        raise ValueError(f"Homography matrix is singular and cannot be inverted: {e}")

    return transform_point(point, inv_h)


def draw_reference_points(
    image: np.ndarray,
    points: Union[List[Tuple[float, float]], np.ndarray],
    color: Tuple[int, int, int] = (0, 255, 0),
    radius: int = 6,
    thickness: int = -1
) -> np.ndarray:
    """
    Draws labeled reference keypoints on an OpenCV frame for calibration visual debugging.

    Args:
        image: Source OpenCV frame (BGR).
        points: List of keypoint coordinates (x, y).
        color: Circle color in BGR tuple format (default Green: (0, 255, 0)).
        radius: Circle radius in pixels.
        thickness: Line thickness (-1 for filled circle).

    Returns:
        Annotated BGR frame copy.
    """
    if image is None:
        raise ValueError("Input image cannot be None.")

    annotated = image.copy()

    for idx, (x, y) in enumerate(points):
        cx, cy = int(round(x)), int(round(y))

        # Draw point circle
        cv2.circle(annotated, (cx, cy), radius, color, thickness)

        # Draw outer ring
        cv2.circle(annotated, (cx, cy), radius + 2, (0, 0, 0), 1)

        # Render point label index
        label = f"P{idx + 1}"
        cv2.putText(
            annotated,
            label,
            (cx + 8, cy - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )
        cv2.putText(
            annotated,
            label,
            (cx + 8, cy - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA
        )

    return annotated
