import pytest
import numpy as np
import cv2
import json
import os
from pathlib import Path

from app.homography.calibrator import (
    LandmarkHomographyCalibrator,
    LandmarkClickSelector,
    PITCH_LANDMARKS,
    LandmarkCalibrationResult
)


class TestLandmarkCalibrationResult:
    """Test the calibration result dataclass."""
    
    def test_creation(self):
        result = LandmarkCalibrationResult(
            homography_matrix=np.eye(3),
            image_points=np.array([[100, 100]]),
            world_points=np.array([[10, 10]]),
            reprojection_error=0.5,
            determinant=1.0,
            metres_per_pixel_x=0.05,
            metres_per_pixel_y=0.05,
            confidence=0.9,
            num_landmarks=6,
            success=True,
            message="OK"
        )
        assert result.success
        assert result.num_landmarks == 6
        assert result.determinant == 1.0


class TestLandmarkHomographyCalibrator:
    """Test homography calibrator with landmarks."""
    
    @pytest.fixture
    def calibrator(self, tmp_path):
        config = {
            "validation": {
                "max_reprojection_error": 3.0,
                "min_determinant": 1e-6,
                "max_determinant": 1e6,
                "min_metres_per_pixel": 0.01,
                "max_metres_per_pixel": 0.5,
                "min_confidence": 0.7,
                "max_aspect_ratio_deviation": 0.5
            },
            "pitch": {"length": 105.0, "width": 68.0}
        }
        config_file = tmp_path / "test_calibration_config.json"
        with open(config_file, 'w') as f:
            json.dump(config, f)
        return LandmarkHomographyCalibrator(config_path=str(config_file))
    
    def test_insufficient_landmarks(self, calibrator):
        image_points = np.array([[100, 100], [200, 200]], dtype=np.float32)
        world_points = np.array([[0, 0], [10, 10]], dtype=np.float32)
        result = calibrator.calibrate(image_points, world_points, (1080, 1920))
        
        assert not result.success
        assert result.num_landmarks == 2
        assert "6" in result.message  # mentions minimum
    
    def test_too_many_landmarks(self, calibrator):
        # 21 points exceeds MAX_LANDMARKS (20)
        n = 21
        image_points = np.random.rand(n, 2).astype(np.float32) * 1000
        world_points = np.random.rand(n, 2).astype(np.float32) * 100
        result = calibrator.calibrate(image_points, world_points, (1080, 1920))
        
        assert not result.success
        assert "20" in result.message
    
    def test_valid_calibration(self, calibrator):
        # Create a good synthetic calibration: 4 corners + extra landmarks
        h, w = 1080, 1920
        image_points = np.array([
            [400, 300],    # bottom-left
            [400, 800],    # top-left
            [1600, 800],   # top-right
            [1600, 300],   # bottom-right
            [1000, 550],   # center
            [1000, 300],   # halfway bottom
        ], dtype=np.float32)
        
        world_points = np.array([
            [0, 0],
            [0, 68],
            [105, 68],
            [105, 0],
            [52.5, 34],
            [52.5, 0],
        ], dtype=np.float32)
        
        result = calibrator.calibrate(image_points, world_points, (h, w))
        
        assert result.success
        assert result.reprojection_error < 3.0
        assert result.metres_per_pixel_x > 0.01
        assert result.metres_per_pixel_y > 0.01
        assert result.confidence >= 0.7
        assert result.num_landmarks == 6
    
    def test_calibration_with_noise(self, calibrator):
        """Calibration with slight noise should still pass."""
        h, w = 1080, 1920
        base_image = np.array([
            [400, 300],
            [400, 800],
            [1600, 800],
            [1600, 300],
            [1000, 550],
            [1000, 300],
        ], dtype=np.float32)
        
        # Add small Gaussian noise
        np.random.seed(42)
        noise = np.random.normal(0, 1.5, base_image.shape).astype(np.float32)
        image_points = base_image + noise
        
        world_points = np.array([
            [0, 0],
            [0, 68],
            [105, 68],
            [105, 0],
            [52.5, 34],
            [52.5, 0],
        ], dtype=np.float32)
        
        result = calibrator.calibrate(image_points, world_points, (h, w))
        
        # Should still be valid with small noise
        assert result.success
        assert result.reprojection_error < calibrator.config["validation"]["max_reprojection_error"]
    
    def test_calibration_with_bad_points_rejected(self, calibrator):
        """Calibration with collinear points should fail."""
        h, w = 1080, 1920
        image_points = np.array([
            [100, 100],
            [200, 200],
            [300, 300],
            [400, 400],
            [500, 500],
            [600, 600],
        ], dtype=np.float32)
        
        world_points = np.array([
            [0, 0],
            [10, 10],
            [20, 20],
            [30, 30],
            [40, 40],
            [50, 50],
        ], dtype=np.float32)
        
        result = calibrator.calibrate(image_points, world_points, (h, w))
        
        # Collinear points should cause RANSAC to fail or produce bad homography
        assert not result.success or result.confidence < 0.7
    
    def test_save_calibration_success(self, calibrator, tmp_path):
        output_file = tmp_path / "test_calibration.json"
        image_points = np.array([
            [400, 300],
            [400, 800],
            [1600, 800],
            [1600, 300],
            [1000, 550],
            [1000, 300],
        ], dtype=np.float32)
        world_points = np.array([
            [0, 0],
            [0, 68],
            [105, 68],
            [105, 0],
            [52.5, 34],
            [52.5, 0],
        ], dtype=np.float32)
        
        result = calibrator.calibrate(image_points, world_points, (1080, 1920))
        assert result.success
        
        saved = calibrator.save_calibration(result, str(output_file))
        assert saved
        assert output_file.exists()
        
        with open(output_file, 'r') as f:
            data = json.load(f)
        assert data["success"]
        assert "homography_matrix" in data
        assert data["num_landmarks"] == 6
    
    def test_save_calibration_rejects_invalid(self, calibrator, tmp_path):
        output_file = tmp_path / "test_invalid_calibration.json"
        
        # Create an invalid result
        invalid_result = LandmarkCalibrationResult(
            homography_matrix=np.eye(3),
            image_points=np.array([[0, 0]]),
            world_points=np.array([[0, 0]]),
            reprojection_error=100.0,
            determinant=0.001,
            metres_per_pixel_x=0.001,
            metres_per_pixel_y=0.001,
            confidence=0.1,
            num_landmarks=1,
            success=False,
            message="Fails validation"
        )
        
        saved = calibrator.save_calibration(invalid_result, str(output_file))
        assert not saved
        assert not output_file.exists()


class TestPitchOverlayGeneration:
    """Test pitch overlay preview generation."""
    
    def test_generate_pitch_overlay(self):
        calibrator = LandmarkHomographyCalibrator()
        overlay = calibrator.generate_pitch_overlay((1080, 1920))
        
        assert overlay.shape == (800, 600, 3)
        # Should have green background
        assert overlay[10, 10, 1] > 100  # green channel


class TestLandmarkClickSelector:
    """Test interactive landmark selection (without GUI)."""
    
    def test_selector_initialization(self):
        selector = LandmarkClickSelector()
        assert len(selector.clicked_points) == 0
    
    def test_pitch_landmarks_structure(self):
        """Verify PITCH_LANDMARKS has correct structure."""
        assert len(PITCH_LANDMARKS) >= 4
        for name, info in PITCH_LANDMARKS.items():
            assert "world" in info
            assert "description" in info
            assert len(info["world"]) == 2
            assert all(v >= 0 for v in info["world"])  # non-negative coordinates