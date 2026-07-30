import cv2
import numpy as np
import json
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass, asdict
import os

from app.homography.homography_utils import compute_homography


@dataclass
class LandmarkCalibrationResult:
    """Result of landmark-based homography calibration."""
    homography_matrix: np.ndarray
    image_points: np.ndarray
    world_points: np.ndarray
    reprojection_error: float
    determinant: float
    metres_per_pixel_x: float
    metres_per_pixel_y: float
    confidence: float
    num_landmarks: int
    success: bool
    message: str
    pitch_overlay: Optional[np.ndarray] = None


# Standard football pitch landmarks (6-20 can be selected)
PITCH_LANDMARKS = {
    "corner_bl": {"world": (0, 0), "description": "Bottom-left corner"},
    "corner_tl": {"world": (0, 68), "description": "Top-left corner"},
    "corner_tr": {"world": (105, 68), "description": "Top-right corner"},
    "corner_br": {"world": (105, 0), "description": "Bottom-right corner"},
    "penalty_bl_l": {"world": (0, 16.5), "description": "Bottom-left penalty area left post"},
    "penalty_bl_r": {"world": (0, 51.5), "description": "Bottom-left penalty area right post"},
    "penalty_tl_l": {"world": (16.5, 68), "description": "Top-left penalty area left post"},
    "penalty_tl_r": {"world": (52, 68), "description": "Top-left penalty area right post"},
    "penalty_tr_l": {"world": (52, 68), "description": "Top-right penalty area left post"},
    "penalty_tr_r": {"world": (105, 68), "description": "Top-right penalty area right post"},
    "penalty_br_l": {"world": (52, 0), "description": "Bottom-right penalty area left post"},
    "penalty_br_r": {"world": (16.5, 0), "description": "Bottom-right penalty area right post"},
    "goal_bl": {"world": (0, 34), "description": "Bottom-left goal center"},
    "goal_tl": {"world": (105, 34), "description": "Top-left goal center"},
    "goal_tr": {"world": (0, 34), "description": "Top-right goal center"},
    "goal_br": {"world": (105, 34), "description": "Bottom-right goal center"},
    "center": {"world": (52.5, 34), "description": "Center circle center"},
    "half_bl": {"world": (52.5, 0), "description": "Halfway line bottom"},
    "half_tl": {"world": (52.5, 68), "description": "Halfway line top"},
}


class CalibrationError(Exception):
    """Raised when homography calibration data is missing or invalid."""
    pass


class LandmarkHomographyCalibrator:
    """Calibrate homography using 6-20 user-selected landmarks."""
    
    MIN_LANDMARKS = 6
    MAX_LANDMARKS = 20
    
    def __init__(self, config_path: str = "configs/homography_calibration.json"):
        self.config_path = config_path
        self.config = self._load_config()
        
    def _load_config(self) -> Dict:
        """Load calibration configuration."""
        if os.path.exists(self.config_path):
            with open(self.config_path, 'r') as f:
                return json.load(f)
        return {
            "validation": {
                "max_reprojection_error": 3.0,
                "min_determinant": 0.01,
                "max_determinant": 10.0,
                "min_metres_per_pixel": 0.01,
                "max_metres_per_pixel": 0.5,
                "min_confidence": 0.7,
                "max_aspect_ratio_deviation": 0.5
            },
            "pitch": {"length": 105.0, "width": 68.0}
        }
    
    def generate_pitch_overlay(self, frame_shape: Tuple[int, int], output_size: Tuple[int, int] = (800, 600)) -> np.ndarray:
        """Generate a blank pitch overlay for preview."""
        h, w = output_size
        overlay = np.zeros((h, w, 3), dtype=np.uint8)
        overlay[:, :] = (34, 139, 34)
        
        margin = 20
        pitch_w = w - 2 * margin
        pitch_h = h - 2 * margin
        
        cv2.rectangle(overlay, (margin, margin), (margin + pitch_w, margin + pitch_h), (255, 255, 255), 2)
        cv2.line(overlay, (margin + pitch_w // 2, margin), (margin + pitch_w // 2, margin + pitch_h), (255, 255, 255), 2)
        radius = int(0.1 * min(pitch_w, pitch_h))
        cv2.circle(overlay, (margin + pitch_w // 2, margin + pitch_h // 2), radius, (255, 255, 255), 2)
        
        return overlay
    
    def validate_calibration(self, result: LandmarkCalibrationResult, frame_shape: Tuple[int, int]) -> Tuple[bool, str]:
        """Validate calibration quality and reject poor calibrations."""
        config = self.config["validation"]
        
        if result.reprojection_error > config["max_reprojection_error"]:
            return False, f"Reprojection error too high: {result.reprojection_error:.2f}px (max: {config['max_reprojection_error']})"
        
        if not (config["min_determinant"] < result.determinant < config["max_determinant"]):
            return False, f"Homography determinant out of range: {result.determinant:.3f} (expected positive)"
        
        if not (config["min_metres_per_pixel"] < result.metres_per_pixel_x < config["max_metres_per_pixel"]):
            return False, f"Metres-per-pixel X out of range: {result.metres_per_pixel_x:.4f}"
        if not (config["min_metres_per_pixel"] < result.metres_per_pixel_y < config["max_metres_per_pixel"]):
            return False, f"Metres-per-pixel Y out of range: {result.metres_per_pixel_y:.4f}"
        
        if result.confidence < config["min_confidence"]:
            return False, f"Confidence too low: {result.confidence:.2f} (min: {config['min_confidence']})"
        
        return True, "Validation passed"
    
    def calibrate(self, 
                  image_points: np.ndarray, 
                  world_points: np.ndarray,
                  frame_shape: Tuple[int, int]) -> LandmarkCalibrationResult:
        """
        Estimate homography using RANSAC on selected landmarks.
        
        Args:
            image_points: (N, 2) array of clicked image points (x, y)
            world_points: (N, 2) array of corresponding world coordinates (x, y) in metres
            frame_shape: (height, width) of the frame
            
        Returns:
            LandmarkCalibrationResult with homography and diagnostics
        """
        if len(image_points) < self.MIN_LANDMARKS:
            return LandmarkCalibrationResult(
                homography_matrix=np.eye(3),
                image_points=image_points,
                world_points=world_points,
                reprojection_error=float('inf'),
                determinant=0.0,
                metres_per_pixel_x=0.0,
                metres_per_pixel_y=0.0,
                confidence=0.0,
                num_landmarks=len(image_points),
                success=False,
                message=f"Need at least {self.MIN_LANDMARKS} landmarks, got {len(image_points)}"
            )
        
        if len(image_points) > self.MAX_LANDMARKS:
            return LandmarkCalibrationResult(
                homography_matrix=np.eye(3),
                image_points=image_points,
                world_points=world_points,
                reprojection_error=float('inf'),
                determinant=0.0,
                metres_per_pixel_x=0.0,
                metres_per_pixel_y=0.0,
                confidence=0.0,
                num_landmarks=len(image_points),
                success=False,
                message=f"Maximum {self.MAX_LANDMARKS} landmarks supported"
            )
        
        # Estimate homography using RANSAC
        homography, mask = cv2.findHomography(
            image_points, 
            world_points, 
            cv2.RANSAC, 
            ransacReprojThreshold=3.0,
            maxIters=2000,
            confidence=0.99
        )
        
        if homography is None:
            return LandmarkCalibrationResult(
                homography_matrix=np.eye(3),
                image_points=image_points,
                world_points=world_points,
                reprojection_error=float('inf'),
                determinant=0.0,
                metres_per_pixel_x=0.0,
                metres_per_pixel_y=0.0,
                confidence=0.0,
                num_landmarks=len(image_points),
                success=False,
                message="RANSAC failed to find homography"
            )
        
        # Compute reprojection error for inliers
        inlier_image_points = image_points[mask.ravel() == 1]
        inlier_world_points = world_points[mask.ravel() == 1]
        
        reprojected = cv2.perspectiveTransform(inlier_image_points.reshape(-1, 1, 2), homography)
        reprojection_errors = np.linalg.norm(
            inlier_world_points - reprojected.reshape(-1, 2), 
            axis=1
        )
        mean_reprojection_error = float(np.mean(reprojection_errors))
        
        # Compute determinant of top-left 2x2 (scale/shear part)
        det = np.linalg.det(homography[:2, :2])
        
        # Compute metres-per-pixel from homography scale
        sx = np.linalg.norm(homography[:2, 0])
        sy = np.linalg.norm(homography[:2, 1])
        
        metres_per_pixel_x = sx if sx > 0 else 0.01
        metres_per_pixel_y = sy if sy > 0 else 0.01
        
        # Compute confidence based on inlier ratio and reprojection error
        inlier_ratio = len(inlier_image_points) / max(len(image_points), 1)
        error_factor = max(0, 1.0 - mean_reprojection_error / 5.0)
        confidence = float(inlier_ratio * 0.6 + error_factor * 0.4)
        
        result = LandmarkCalibrationResult(
            homography_matrix=homography,
            image_points=image_points,
            world_points=world_points,
            reprojection_error=mean_reprojection_error,
            determinant=float(det),
            metres_per_pixel_x=metres_per_pixel_x,
            metres_per_pixel_y=metres_per_pixel_y,
            confidence=confidence,
            num_landmarks=len(image_points),
            success=False,
            message="Calibration computed, pending validation"
        )
        
        # Validate
        is_valid, message = self.validate_calibration(result, frame_shape)
        result.success = is_valid
        result.message = message
        
        return result
    
    def load_calibration(self, filename: str) -> bool:
        """
        Load calibration from JSON file.
        
        Supports two formats:
        1. Pre-computed homography_matrix
        2. calibration_points (source/destination) - computes matrix dynamically
        """
        if not os.path.exists(filename):
            raise FileNotFoundError(f"Calibration file not found: {filename}")
        
        with open(filename, 'r') as f:
            data = json.load(f)
        
        # Case 1: Pre-computed matrix
        if "homography_matrix" in data:
            self.homography_matrix = np.array(data["homography_matrix"], dtype=np.float64)
            self.calibration_method = data.get("validation_message", "pre-computed")
            return True
        
        # Case 2: Compute from calibration points
        elif "calibration_points" in data:
            src_pts = np.array(data["calibration_points"]["source"], dtype=np.float64)
            dst_pts = np.array(data["calibration_points"]["destination"], dtype=np.float64)
            
            if src_pts.shape[0] < 4 or dst_pts.shape[0] < 4:
                raise CalibrationError(
                    f"Need at least 4 calibration points, got {src_pts.shape[0]}"
                )
            
            matrix, mask = compute_homography(src_pts, dst_pts)
            self.homography_matrix = matrix
            self.calibration_method = "computed from calibration_points"
            return True
        
        # Case 3: Neither format found
        else:
            raise CalibrationError(
                "Calibration file must contain either 'homography_matrix' or 'calibration_points'"
            )

    def get_matrix(self) -> np.ndarray:
        """Get the current homography matrix."""
        if not hasattr(self, 'homography_matrix') or self.homography_matrix is None:
            return np.eye(3, dtype=np.float64)
        return self.homography_matrix

    def get_summary(self) -> Dict:
        """Get calibration summary."""
        return {
            "method": getattr(self, 'calibration_method', 'unknown'),
            "has_matrix": hasattr(self, 'homography_matrix') and self.homography_matrix is not None,
            "matrix_shape": self.homography_matrix.shape if hasattr(self, 'homography_matrix') else None
        }

    def save_calibration(self, result: LandmarkCalibrationResult, filename: str) -> bool:
        """Save calibration to JSON only if validation passed."""
        if not result.success:
            print(f"Refusing to save invalid calibration: {result.message}")
            return False
        
        calibration_data = {
            "homography_matrix": result.homography_matrix.tolist(),
            "image_points": result.image_points.tolist(),
            "world_points": result.world_points.tolist(),
            "reprojection_error": result.reprojection_error,
            "determinant": result.determinant,
            "metres_per_pixel_x": result.metres_per_pixel_x,
            "metres_per_pixel_y": result.metres_per_pixel_y,
            "confidence": result.confidence,
            "num_landmarks": result.num_landmarks,
            "success": True,
            "validation_message": result.message
        }
        
        os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else ".", exist_ok=True)
        with open(filename, 'w') as f:
            json.dump(calibration_data, f, indent=2)
        
        # Store the matrix for later use
        self.homography_matrix = result.homography_matrix
        self.calibration_method = result.message
        
        print(f"Calibration saved to {filename}")
        return True


class LandmarkClickSelector:
    """Interactive landmark selection for calibration."""
    
    def __init__(self, window_name: str = "Select Landmarks"):
        self.window_name = window_name
        self.clicked_points: List[Tuple[int, int]] = []
        self.frame: Optional[np.ndarray] = None
        
    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse clicks for landmark selection."""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.clicked_points.append((x, y))
            cv2.circle(self.frame, (x, y), 5, (0, 255, 0), -1)
            cv2.putText(self.frame, str(len(self.clicked_points)), (x+10, y-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    def select_landmarks(self, frame: np.ndarray, landmarks: Dict[str, Dict]) -> Tuple[np.ndarray, np.ndarray]:
        """Let user click on visible landmarks."""
        self.frame = frame.copy()
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)
        
        info_y = 20
        for i, (name, info) in enumerate(list(landmarks.items())[:20]):
            cv2.putText(self.frame, f"{i+1}: {name}", (10, info_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            info_y += 15
        
        cv2.putText(self.frame, f"Click at least 6 landmarks. Press 's' when done, 'c' to clear, 'q' to quit",
                   (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        self.clicked_points = []
        while True:
            cv2.imshow(self.window_name, self.frame)
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('c'):
                self.clicked_points = []
                self.frame = frame.copy()
            elif key == ord('s'):
                if len(self.clicked_points) >= LandmarkHomographyCalibrator.MIN_LANDMARKS:
                    break
        
        cv2.destroyWindow(self.window_name)
        
        if len(self.clicked_points) < LandmarkHomographyCalibrator.MIN_LANDMARKS:
            return np.array([]), np.array([])
        
        image_points = np.array(self.clicked_points, dtype=np.float32)
        world_points_list = [v["world"] for v in list(landmarks.values())[:len(self.clicked_points)]]
        world_points = np.array(world_points_list, dtype=np.float32)
        
        # Flip Y to match image orientation
        world_points[:, 1] = 68 - world_points[:, 1]
        
        return image_points, world_points