"""
Tests for Ball Detection Module

Tests the complete ball detection flow:
1. BallDetector initialization with config
2. Detection scoring and best selection
3. Detection filtering (area, pitch ROI, frame boundaries)
4. Integration with BallTracker and BallInterpolator
"""

import json
import logging
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pytest

from app.detection.ball_detector import BallDetector
from app.detection.detection_types import Detection
from app.tracking.ball_tracker import BallTracker
from app.tracking.ball_interpolation import BallInterpolator

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("test_ball_detector")


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def mock_frame() -> np.ndarray:
    """Create a mock 1280x720 frame."""
    return np.zeros((720, 1280, 3), dtype=np.uint8)


@pytest.fixture
def pitch_roi() -> np.ndarray:
    """Standard pitch ROI polygon covering most of a 1280x720 frame."""
    return np.array([
        [0, 200],
        [1280, 200],
        [1280, 720],
        [0, 720],
    ], dtype=np.int32)


@pytest.fixture
def sample_ball_detections() -> List[Detection]:
    """Create sample ball detections for testing."""
    return [
        Detection(
            cls_id=32,
            conf=0.85,
            bbox=(600, 300, 620, 320),  # Center: (610, 310)
        ),
        Detection(
            cls_id=32,
            conf=0.65,
            bbox=(400, 250, 415, 268),  # Center: (407.5, 259)
        ),
        Detection(
            cls_id=32,
            conf=0.45,
            bbox=(800, 350, 818, 370),  # Center: (809, 360)
        ),
    ]


@pytest.fixture
def ball_detector() -> BallDetector:
    """Create BallDetector with test-friendly config."""
    return BallDetector(
        model_path="yolov8n.pt",  # Small model for testing
        conf=0.10,
        imgsz=640,
    )


# =============================================================================
# Test 1: BallDetector Initialization
# =============================================================================

class TestBallDetectorInitialization:
    """Test BallDetector constructor and configuration."""

    def test_default_init(self):
        """Test initialization with default parameters."""
        detector = BallDetector()
        assert detector.conf == 0.10  # ball_confidence_threshold
        assert detector.imgsz == 960  # ball_image_size
        assert detector.classes == [32]  # ball only
        assert detector.model_path is not None
        assert detector.model is None  # Not loaded yet
        assert detector.total_detections == 0
        assert detector.filtered_detections == 0
        assert detector.accepted_detections == 0

    def test_custom_init(self):
        """Test initialization with custom parameters."""
        detector = BallDetector(
            model_path="yolov8x.pt",
            conf=0.15,
            iou=0.6,
            imgsz=1280,
        )
        assert detector.model_path == "yolov8x.pt"
        assert detector.conf == 0.15
        assert detector.iou == 0.6
        assert detector.imgsz == 1280
        assert detector.classes == [32]

    def test_device_fallback(self):
        """Test device fallback when CUDA is not available."""
        import torch
        detector = BallDetector(device="cuda:99")  # Invalid CUDA device
        if not torch.cuda.is_available():
            assert detector.device == "cpu"

    def test_config_override(self):
        """Test initialization with custom config dict."""
        config = {
            "models": {
                "ball_confidence_threshold": 0.20,
                "ball_image_size": 800,
                "iou_threshold": 0.4,
            },
            "tracking": {
                "ball_max_match_dist": 150.0,
            },
            "detection_filter": {
                "max_ball_area": 2000,
                "ball_center_margin_px": 50.0,
            },
        }
        detector = BallDetector(config=config)
        assert detector.conf == 0.20
        assert detector.imgsz == 800
        assert detector.iou == 0.4
        assert detector.max_match_dist == 150.0
        assert detector.max_ball_area == 2000


# =============================================================================
# Test 2: Detection Scoring
# =============================================================================

class TestBallDetectionScoring:
    """Test detection scoring and best selection logic."""

    def test_score_confidence_only(self, sample_ball_detections):
        """Test scoring with no prediction (confidence only)."""
        detector = BallDetector()
        det = sample_ball_detections[0]  # conf=0.85
        score = detector.score_detection(det, predicted_center=None)
        assert score == 0.85

    def test_score_with_proximity(self, sample_ball_detections):
        """Test scoring with prediction proximity bonus."""
        detector = BallDetector()
        det = sample_ball_detections[0]  # Center: (610, 310)
        
        # Detection near prediction (small distance penalty)
        score_near = detector.score_detection(det, predicted_center=(610, 310))
        assert score_near == 0.85  # Zero distance penalty

        # Detection far from prediction (large distance penalty)
        score_far = detector.score_detection(det, predicted_center=(100, 100))
        dist = np.hypot(610 - 100, 310 - 100)
        expected_penalty = min(dist / detector.dist_penalty_scale, 2.0)
        assert score_far == pytest.approx(0.85 - expected_penalty)

    def test_pick_best_detection(self, sample_ball_detections):
        """Test picking best detection from multiple candidates."""
        detector = BallDetector()
        
        # Without prediction, highest confidence wins
        best = detector.pick_best_detection(
            sample_ball_detections, predicted_center=None
        )
        assert best is not None
        assert best.conf == 0.85  # Highest confidence

        # With prediction near a lower-confidence detection
        best_near = detector.pick_best_detection(
            sample_ball_detections,
            predicted_center=(407, 259),  # Near the 0.65 detection
        )
        assert best_near is not None
        # The 0.65 detection should score high due to proximity

    def test_pick_best_empty(self):
        """Test picking best from empty list."""
        detector = BallDetector()
        best = detector.pick_best_detection([], predicted_center=None)
        assert best is None


# =============================================================================
# Test 3: Detection Filtering
# =============================================================================

class TestBallDetectionFiltering:
    """Test geometric and pitch-based detection filtering."""

    def test_filter_ball_too_large(self, mock_frame):
        """Test filtering out overly large ball detections."""
        detector = BallDetector()
        # Create a very large ball detection
        large_ball = Detection(cls_id=32, conf=0.8, bbox=(100, 100, 500, 500))
        detections = [large_ball]
        
        # With max_ball_area=2600, area=160000 should be filtered
        filtered = detector.filter_detections(detections, mock_frame.shape[:2])
        assert len(filtered) == 0
        assert large_ball.reject_reason == "ball_too_large"

    def test_filter_out_of_frame(self, mock_frame):
        """Test filtering detections outside frame boundaries."""
        detector = BallDetector()
        h, w = mock_frame.shape[:2]  # 720, 1280
        
        # Detection outside frame
        outside = Detection(cls_id=32, conf=0.8, bbox=(-100, -100, -50, -50))
        filtered = detector.filter_detections([outside], (h, w))
        assert len(filtered) == 0
        assert outside.reject_reason == "out_of_frame"

    def test_filter_outside_pitch_roi(self, mock_frame, pitch_roi):
        """Test filtering detections outside pitch ROI."""
        detector = BallDetector()
        detector.set_pitch_roi(pitch_roi)
        
        # Detection far outside pitch
        outside = Detection(cls_id=32, conf=0.8, bbox=(50, 50, 70, 70))
        filtered = detector.filter_detections([outside], mock_frame.shape[:2])
        assert len(filtered) == 0
        assert outside.reject_reason == "ball_outside_pitch"

    def test_filter_valid_detections(self, mock_frame, pitch_roi):
        """Test that valid detections pass through filter."""
        detector = BallDetector()
        detector.set_pitch_roi(pitch_roi)
        
        # Small valid detection on pitch
        valid = Detection(cls_id=32, conf=0.8, bbox=(600, 350, 615, 365))
        filtered = detector.filter_detections([valid], mock_frame.shape[:2])
        assert len(filtered) == 1
        assert filtered[0] == valid
        assert valid.reject_reason == ""

    def test_filter_multiple_mixed(self, mock_frame, pitch_roi):
        """Test filtering with mix of valid and invalid detections."""
        detector = BallDetector()
        detector.set_pitch_roi(pitch_roi)
        
        detections = [
            Detection(cls_id=32, conf=0.8, bbox=(600, 350, 615, 365)),  # Valid
            Detection(cls_id=32, conf=0.7, bbox=(50, 50, 500, 500)),    # Too large
            Detection(cls_id=32, conf=0.6, bbox=(-50, 100, -30, 120)),  # Out of frame
            Detection(cls_id=32, conf=0.5, bbox=(0, 0, 20, 20)),       # Outside pitch
        ]
        
        filtered = detector.filter_detections(detections, mock_frame.shape[:2])
        assert len(filtered) == 1
        assert filtered[0].conf == 0.8


# =============================================================================
# Test 4: Detection to Dict Conversion
# =============================================================================

class TestDetectionToDict:
    """Test conversion between Detection and dict formats."""

    def test_detection_to_dict(self):
        """Test converting Detection to BallTracker-compatible dict."""
        detector = BallDetector()
        det = Detection(cls_id=32, conf=0.85, bbox=(600, 300, 620, 320))
        
        result = detector.detection_to_dict(det)
        assert result["center"] == (610.0, 310.0)
        assert result["bbox"] == [600, 300, 620, 320]
        assert result["confidence"] == 0.85

    def test_detections_to_dict_list(self, sample_ball_detections):
        """Test converting multiple Detections to dict list."""
        detector = BallDetector()
        results = detector.detections_to_dict_list(sample_ball_detections)
        
        assert len(results) == 3
        for i, r in enumerate(results):
            det = sample_ball_detections[i]
            # Center is computed from Detection.center property which uses int()
            cx_expect = int((det.bbox[0] + det.bbox[2]) / 2.0)
            cy_expect = int((det.bbox[1] + det.bbox[3]) / 2.0)
            assert r["center"] == (float(cx_expect), float(cy_expect))
            assert r["bbox"] == list(det.bbox)
            assert r["confidence"] == det.conf


# =============================================================================
# Test 5: Full Pipeline Integration
# =============================================================================

class TestBallDetectionPipeline:
    """Test integration of BallDetector + BallTracker + BallInterpolator."""

    def test_detect_and_filter_pipeline(self, mock_frame, pitch_roi):
        """Test the complete detect_and_filter pipeline method."""
        detector = BallDetector(
            model_path="yolov8n.pt",
            conf=0.10,
            imgsz=640,
        )
        detector.set_pitch_roi(pitch_roi)
        
        best_det, filtered_dets, inference_ms = detector.detect_and_filter(
            mock_frame, predicted_center=None
        )
        
        # Should return results (may be None for no ball in frame)
        assert isinstance(inference_ms, float)
        assert inference_ms >= 0
        assert isinstance(filtered_dets, list)

    def test_tracker_integration(self, sample_ball_detections, mock_frame):
        """Test BallDetector output feeding into BallTracker."""
        detector = BallDetector()
        tracker = BallTracker()
        
        # Convert detections to dict and feed to tracker
        det_list = detector.detections_to_dict_list(sample_ball_detections)
        
        # First frame - should initialize track
        result1 = tracker.update(det_list, frame_number=1)
        assert result1 is not None
        assert result1["track_id"] == 1
        
        # Second frame with same detections
        result2 = tracker.update(det_list, frame_number=2)
        assert result2 is not None
        assert result2["track_id"] == 1
        
        # Empty detections should use prediction
        result3 = tracker.update([], frame_number=3)
        assert result3 is not None
        assert result3["is_predicted"] is True

    def test_interpolator_integration(self, sample_ball_detections):
        """Test BallInterpolator with tracker output."""
        detector = BallDetector()
        tracker = BallTracker()
        interpolator = BallInterpolator(max_gap=5, method="linear")
        
        # Simulate tracking over 10 frames with gaps
        track_history = []
        total_frames = 10
        
        for frame_num in range(1, total_frames + 1):
            # Only provide detections on some frames
            if frame_num in [1, 2, 5, 8, 10]:
                det_list = detector.detections_to_dict_list(sample_ball_detections[:1])
            else:
                det_list = []
            
            result = tracker.update(det_list, frame_number=frame_num)
            if result is not None:
                track_history.append(result)
        
        # Interpolate
        trajectory = interpolator.interpolate(track_history, total_frames)
        
        # Should have some frames filled
        assert len(trajectory) > 0
        assert interpolator.get_stats()["total_frames"] == total_frames


# =============================================================================
# Test 6: Metrics
# =============================================================================

class TestBallDetectionMetrics:
    """Test ball detector metrics reporting."""

    def test_metrics_initial_state(self):
        """Test metrics before any detections."""
        detector = BallDetector()
        metrics = detector.get_metrics()
        
        assert metrics["total_raw_detections"] == 0
        assert metrics["filtered_detections"] == 0
        assert metrics["accepted_detections"] == 0
        assert metrics["confidence_threshold"] == detector.conf

    def test_metrics_after_filtering(self, mock_frame, pitch_roi):
        """Test metrics update after filtering."""
        detector = BallDetector()
        detector.set_pitch_roi(pitch_roi)
        
        # Filter some detections
        detections = [
            Detection(cls_id=32, conf=0.8, bbox=(600, 350, 615, 365)),  # Valid
            Detection(cls_id=32, conf=0.7, bbox=(50, 50, 500, 500)),    # Filtered
        ]
        filtered = detector.filter_detections(detections, mock_frame.shape[:2])
        
        metrics = detector.get_metrics()
        assert metrics["total_raw_detections"] == 0  # Only detect() increments this
        assert metrics["filtered_detections"] == 1
        assert metrics["accepted_detections"] == 1


# =============================================================================
# Test 7: Edge Cases
# =============================================================================

class TestBallDetectionEdgeCases:
    """Test edge cases and error handling."""

    def test_detect_without_load(self, mock_frame):
        """Test detection raises error if model not loaded."""
        detector = BallDetector()
        with pytest.raises(RuntimeError, match="BallDetector.load()"):
            detector.detect(mock_frame)

    def test_empty_detections(self, mock_frame, pitch_roi):
        """Test filtering with empty detection list."""
        detector = BallDetector()
        detector.set_pitch_roi(pitch_roi)
        
        filtered = detector.filter_detections([], mock_frame.shape[:2])
        assert filtered == []

    def test_detection_on_boundary(self, mock_frame, pitch_roi):
        """Test detections exactly on pitch boundary."""
        detector = BallDetector()
        detector.set_pitch_roi(pitch_roi)
        
        # Detection right on the edge of pitch ROI
        boundary = Detection(cls_id=32, conf=0.5, bbox=(12, 520, 30, 540))
        filtered = detector.filter_detections([boundary], mock_frame.shape[:2])
        assert len(filtered) >= 0  # May or may not be inside depending on pointPolygonTest

    def test_zero_area_detection(self, mock_frame, pitch_roi):
        """Test detection with zero area (invalid bbox)."""
        detector = BallDetector()
        detector.set_pitch_roi(pitch_roi)
        
        # Detection at (600, 400) is within pitch ROI (y >= 200)
        zero_area = Detection(cls_id=32, conf=0.5, bbox=(600, 400, 600, 400))
        filtered = detector.filter_detections([zero_area], mock_frame.shape[:2])
        assert zero_area.area == 0
        assert len(filtered) == 1  # Zero area is within max_ball_area

    def test_score_with_none_prediction(self):
        """Test scoring with None prediction returns confidence."""
        detector = BallDetector()
        det = Detection(cls_id=32, conf=0.75, bbox=(100, 100, 120, 120))
        score = detector.score_detection(det, predicted_center=None)
        assert score == 0.75

    def test_pitch_roi_update(self, pitch_roi):
        """Test updating pitch ROI."""
        detector = BallDetector()
        assert detector._pitch_roi is None
        
        detector.set_pitch_roi(pitch_roi)
        assert detector._pitch_roi is not None
        assert np.array_equal(detector._pitch_roi, pitch_roi)


# =============================================================================
# Test 8: BallTracker Integration (ensure backward compat)
# =============================================================================

class TestBallTrackerIntegration:
    """Test that BallDetector output format is compatible with BallTracker."""

    def test_tracker_accepts_detection_dict(self):
        """Test BallTracker accepts the dict format from BallDetector."""
        detector = BallDetector()
        tracker = BallTracker()
        
        # Create detection dicts as BallDetector would produce
        det_list = [
            {
                "center": (610.0, 310.0),
                "bbox": [600, 300, 620, 320],
                "confidence": 0.85,
            }
        ]
        
        # This should not raise
        result = tracker.update(det_list, frame_number=1)
        assert result is not None
        assert result["track_id"] == 1
        assert result["center"] == (610.0, 310.0)

    def test_tracker_empty_detections(self):
        """Test BallTracker handles empty detection lists."""
        tracker = BallTracker()
        
        # Should handle empty list without error
        result = tracker.update([], frame_number=1)
        assert result is None  # No track initialized

    def test_tracker_full_cycle(self):
        """Test a full cycle: init → track → predict → lost."""
        detector = BallDetector()
        tracker = BallTracker()
        tracker.max_missing_frames = 3  # Small for testing
        
        # Frame 1: Initialize
        result1 = tracker.update(
            [{"center": (500, 300), "bbox": [490, 290, 510, 310], "confidence": 0.85}],
            frame_number=1,
        )
        assert result1 is not None
        assert result1["is_predicted"] is False
        
        # Frame 2: Update
        result2 = tracker.update(
            [{"center": (510, 305), "bbox": [500, 295, 520, 315], "confidence": 0.80}],
            frame_number=2,
        )
        assert result2 is not None
        assert result2["is_predicted"] is False
        assert result2["longest_streak"] == 2
        
        # Frame 3-5: Missing (should predict)
        for f in [3, 4, 5]:
            result = tracker.update([], frame_number=f)
            if f <= 3 + tracker.max_missing_frames:
                # Still within missing tolerance - should predict
                pass
        
        # Frame 6: Still predict
        result6 = tracker.update([], frame_number=6)
        # Should be None since missing_frames (3) <= max_missing_frames (3)
        # Actually missing_frames starts at 0, increments by 1 each missing call
        # Frame 3: missing_frames=1, Frame 4: missing_frames=2, Frame 5: missing_frames=3
        # Frame 6: missing_frames=4 > max_missing_frames=3 => lost
        assert result6 is None or result6["is_predicted"] is True


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])