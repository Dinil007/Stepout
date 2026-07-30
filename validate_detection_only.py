"""
Detection-Only Validation Script

Validates YOLO player detection quality without any downstream modules.
- YOLO Detection (GPU)
- Person Classification (EfficientNet-B0) for Referee/Coach detection
- Team Classification
- Ball Detection
- Visualization
- Export
"""

import os
import time
import cv2
import torch
import numpy as np
from ultralytics import YOLO
from pathlib import Path
from datetime import datetime
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from app.utils.roi_loader import load_pitch_roi_as_numpy
from app.team_classification.color_extractor import ColorExtractor
from app.team_classification.team_classifier import TeamClassifier
from app.team_classification.visualize_teams import TeamVisualizer
from app.detection.ball_detector import BallDetector
from app.tracking.ball_tracker import BallTracker
from app.tracking.ball_interpolation import BallInterpolator
from app.classification.inference import PersonClassifier
from app.classification.config import InferenceConfig, CLASS_NAMES
from app.analytics.ball_possession import BallPossessionAnalyzer

# Configuration
PROJECT_ROOT = Path(__file__).resolve().parent
INPUT_VIDEO = "D:/stepout/videos/raw/match30.mp4"
OUTPUT_VIDEO = "outputs/detected_video.mp4"
OUTPUT_REPORT = "outputs/detection_report.txt"
MODEL_PATH = "yolov8x.pt"
CONFIDENCE_THRESHOLD = 0.25
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CLASS_ID = 0  # Person (standard YOLO model)
ENABLE_TRACKING = True  # Enable YOLO native tracking with persist=True
TRACKER_CONFIG = "app/tracking/bytetrack_custom.yaml"

# Team Classification Configuration
ENABLE_TEAM_CLASSIFICATION = True
WARMUP_FRAMES = 0  # Frames to collect colors before training (0 for immediate classification)
TEAM_CLASSIFIER_HISTORY_LEN = 30
# Manual team overrides for problematic tracks (track_id: team_label)
# team_label: 0 = Red, 1 = Blue (matching team_classifier.py)
MANUAL_TEAM_OVERRIDES = {
    8: 0,   # Track 8 -> Red
    13: 1,  # Track 13 -> Blue
    44: 1,  # Track 44 -> Blue
}

# Ball Detection Configuration
ENABLE_BALL_DETECTION = True
BALL_CONFIDENCE_THRESHOLD = 0.05  # Lowered from 0.10 to detect ball in air
BALL_IMAGE_SIZE = 1280  # Increased from 960 for better detection
BALL_MAX_MATCH_DIST = 200.0  # Increased from 180.0 for better tracking
BALL_MAX_MISSING_FRAMES = 60  # Increased from 45 for better continuity
BALL_INTERPOLATION_MAX_GAP = 30  # Increased from 20 for better interpolation

# Ball Possession Configuration
ENABLE_POSSESSION_ANALYTICS = True
POSSESSION_RADIUS_M = 150.0  # Distance threshold for possession (pixels, using pixel coordinates as proxy)
POSSESSION_CONFIRMATION_FRAMES = 3  # Frames to confirm possession

# Person Classification Configuration (EfficientNet-B0 for Referee/Coach detection)
ENABLE_PERSON_CLASSIFICATION = False  # Disabled - using black-brightness heuristic instead
PERSON_CLASSIFIER_CONFIDENCE = 0.3  # Lowered from 0.7 to capture more predictions
PERSON_CLASSIFIER_MODEL_PATH = Path("models/classifier/efficientnet_b0_best.pth")
# Debug: save sample crops to verify extraction quality
SAVE_DEBUG_CROPS = True
DEBUG_CROP_DIR = Path("outputs/debug_crops")

# Ensure output directories exist
os.makedirs("outputs", exist_ok=True)
if SAVE_DEBUG_CROPS:
    DEBUG_CROP_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("DETECTION-ONLY VALIDATION")
print("=" * 60)
print(f"Device: {DEVICE}")
print(f"Model: {MODEL_PATH}")
print(f"Input: {INPUT_VIDEO}")
print(f"Confidence Threshold: {CONFIDENCE_THRESHOLD}")
print(f"Person Classification: {ENABLE_PERSON_CLASSIFICATION}")
print(f"  Classifier Confidence Threshold: {PERSON_CLASSIFIER_CONFIDENCE}")
print(f"  Model: {PERSON_CLASSIFIER_MODEL_PATH}")
print("=" * 60)

# Load model
print("\n[1/4] Loading YOLO model...")
model = YOLO(MODEL_PATH)
model.to(DEVICE)

if torch.cuda.is_available():
    try:
        model.fuse()
        model.model.half()
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"FP16: Enabled")
    except Exception as e:
        print(f"Warning: Could not enable FP16: {e}")

# Initialize Person Classifier (EfficientNet-B0)
person_classifier = None
track_person_types = {}  # Cache: {track_id: (person_type, confidence)}
person_classification_stats = {
    "total_attempts": 0,
    "successful_classifications": 0,
    "referee_count": 0,
    "coach_count": 0,
    "player_count": 0,
    "low_confidence_count": 0,
    "crop_failures": 0,
}

# YOLO-based referee detection stats
yolo_referee_count = 0
yolo_goalkeeper_count = 0
yolo_player_count = 0
yolo_ball_count = 0

if ENABLE_PERSON_CLASSIFICATION:
    print("\n[1.5/4] Initializing Person Classifier (EfficientNet-B0)...")
    try:
        infer_config = InferenceConfig(
            model_path=PERSON_CLASSIFIER_MODEL_PATH,
            confidence_threshold=PERSON_CLASSIFIER_CONFIDENCE,
            device=DEVICE,
        )
        person_classifier = PersonClassifier(config=infer_config)
        print(f"  PersonClassifier loaded successfully")
        print(f"  Classes: {CLASS_NAMES}")
        print(f"  Confidence threshold: {PERSON_CLASSIFIER_CONFIDENCE}")
    except Exception as e:
        print(f"  WARNING: Could not load PersonClassifier: {e}")
        print(f"  Falling back to black-brightness heuristic for referee detection")
        person_classifier = None

# Open video
print("\n[2/4] Opening video...")
cap = cv2.VideoCapture(INPUT_VIDEO)

if not cap.isOpened():
    raise RuntimeError(f"Cannot open video: {INPUT_VIDEO}")

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"Resolution: {width}x{height}")
print(f"FPS: {fps:.2f}")
print(f"Total Frames: {total_frames}")

# Setup video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
writer = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (width, height))

# Load ROI
pitch_roi, roi_source = load_pitch_roi_as_numpy(PROJECT_ROOT, verbose=True)
roi_enabled = True

# Initialize tracking if enabled
if ENABLE_TRACKING:
    print("\n[2.5/4] Initializing YOLO native tracking...")
    print(f"  Tracker Config: {TRACKER_CONFIG}")
    print(f"  persist=True enabled for ID stability")
    print(f"  Confidence Threshold: {CONFIDENCE_THRESHOLD}")
    print(f"  IoU Threshold: 0.5")
    print(f"  Image Size: 1280")

# Initialize team classification if enabled
color_extractor = None
team_classifier = None
team_visualizer = None
collected_colors = []
track_color_samples = {}
is_classifier_trained = False

if ENABLE_TEAM_CLASSIFICATION:
    print("\n[2.6/4] Initializing Team Classification...")
    color_extractor = ColorExtractor(jersey_ratio=0.5)
    team_classifier = TeamClassifier(history_len=TEAM_CLASSIFIER_HISTORY_LEN)
    team_visualizer = TeamVisualizer()
    print(f"  Color Extractor: jersey_ratio=0.5")
    print(f"  Team Classifier: history_len={TEAM_CLASSIFIER_HISTORY_LEN}")
    print(f"  Warmup Frames: {WARMUP_FRAMES}")
    
    # Apply manual team overrides if configured
    if MANUAL_TEAM_OVERRIDES:
        print(f"  Applying manual overrides for {len(MANUAL_TEAM_OVERRIDES)} tracks:")
        for track_id, team_label in MANUAL_TEAM_OVERRIDES.items():
            team_name = "Red" if team_label == 0 else "Blue"
            team_classifier.set_manual_override(track_id, team_label)
            print(f"    Track {track_id} -> {team_name}")

# Initialize ball detection if enabled
ball_detector = None
ball_tracker = None
ball_interpolator = None
ball_track_history = []
if ENABLE_BALL_DETECTION:
    print("\n[2.7/4] Initializing Ball Detection...")
    ball_detector = BallDetector(
        conf=BALL_CONFIDENCE_THRESHOLD,
        imgsz=BALL_IMAGE_SIZE,
    )
    ball_detector.load()
    ball_detector.set_pitch_roi(pitch_roi)
    print(f"  Ball Detector: conf={BALL_CONFIDENCE_THRESHOLD}, imgsz={BALL_IMAGE_SIZE}")
    
    ball_tracker = BallTracker()
    ball_tracker.max_missing_frames = BALL_MAX_MISSING_FRAMES
    ball_tracker.max_match_dist = BALL_MAX_MATCH_DIST
    print(f"  Ball Tracker: max_missing={BALL_MAX_MISSING_FRAMES}, max_match_dist={BALL_MAX_MATCH_DIST}")
    
    ball_interpolator = BallInterpolator(max_gap=BALL_INTERPOLATION_MAX_GAP)
    print(f"  Ball Interpolator: max_gap={BALL_INTERPOLATION_MAX_GAP}")

# Initialize ball possession analyzer if enabled
ball_possession = None
if ENABLE_POSSESSION_ANALYTICS:
    print("\n[2.8/4] Initializing Ball Possession Analyzer...")
    ball_possession = BallPossessionAnalyzer(
        possession_radius_m=POSSESSION_RADIUS_M,
        confirmation_frames=POSSESSION_CONFIRMATION_FRAMES,
        fps=fps
    )
    print(f"  Possession Radius: {POSSESSION_RADIUS_M}m")
    print(f"  Confirmation Frames: {POSSESSION_CONFIRMATION_FRAMES}")
    print(f"  Video FPS: {fps:.2f}")

# Metrics
total_detections_before_roi = 0
total_detections_after_roi = 0
total_detections_removed = 0
confidences = []
inference_times = []
frame_count = 0

# Detailed logging for discarded detections
discarded_log = []
confidence_below_threshold = []
roi_removed = []
confidence_distribution = []

# Track ID stability logging
track_id_history = {}  # {track_id: [frame_numbers]}
id_changes = []  # List of ID change events

print("\n[3/4] Running detection...")
print("-" * 60)

start_time = time.time()

with torch.inference_mode():
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_start = time.time()

        # YOLO Detection with native tracking (persist=True for ID stability)
        if ENABLE_TRACKING:
            results = model.track(
                source=frame,
                persist=True,  # Critical for ID stability
                tracker=TRACKER_CONFIG,
                classes=[CLASS_ID],
                conf=CONFIDENCE_THRESHOLD,
                iou=0.5,
                imgsz=1280,
                device=DEVICE,
                verbose=False
            )
        else:
            # Detection-only mode
            results = model(
                frame,
                classes=[CLASS_ID],
                conf=CONFIDENCE_THRESHOLD,
                device=DEVICE,
                verbose=False,
                imgsz=1280
            )

        inference_time = time.time() - frame_start
        inference_times.append(inference_time)

        # Process detections/tracks
        annotated_frame = frame.copy()
        frame_detections_before = 0
        frame_detections_after = 0
        frame_removed = 0
        
        # Initialize possession data before player loop
        possession_data = None
        
        # Ball Detection (move before player loop for possession data availability)
        ball_position_pixel = None
        if ENABLE_BALL_DETECTION and ball_detector is not None:
            # Get predicted center from tracker for scoring proximity
            predicted_center = None
            if ball_tracker.is_active() and ball_tracker._track is not None:
                predicted_center = ball_tracker._track.predicted_center

            best_det, filtered_dets, inference_ms = ball_detector.detect_and_filter(
                frame, predicted_center
            )

            # Ball Tracking
            if best_det is not None:
                detection_list = [ball_detector.detection_to_dict(best_det)]
            else:
                detection_list = []

            track_result = ball_tracker.update(detection_list, frame_count)

            # Store track result for interpolation
            if track_result is not None:
                ball_track_history.append(track_result)

            # Ball Visualization
            if track_result is not None:
                center = track_result["center"]
                cx, cy = int(center[0]), int(center[1])
                ball_position_pixel = (cx, cy)
                is_predicted = track_result["is_predicted"]

                # Tracked ball (yellow if predicted, white if detected)
                color = (0, 255, 255) if is_predicted else (255, 255, 255)
                cv2.circle(annotated_frame, (cx, cy), 8, color, 2)
                cv2.circle(annotated_frame, (cx, cy), 3, color, -1)

                # Ball label (always show "BALL" regardless of prediction status)
                cv2.putText(
                    annotated_frame,
                    "BALL",
                    (cx + 10, cy),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2,
                )

                # Ball trajectory history removed as requested
                
                # Draw arrow above ball
                if possession_data is not None and possession_data.get("state") == "In Possession":
                    team_name = possession_data.get("team_name", "Unknown")
                    if team_name == "Blue":
                        arrow_color = (255, 0, 0)  # Blue in BGR
                    elif team_name == "Red":
                        arrow_color = (0, 0, 255)  # Red in BGR
                    else:
                        arrow_color = (128, 128, 128)  # Gray for unknown
                    
                    # Draw arrow above ball
                    arrow_x = cx
                    arrow_y_start = cy - 15
                    arrow_y_end = cy - 5
                    
                    cv2.arrowedLine(
                        annotated_frame,
                        (arrow_x, arrow_y_start),
                        (arrow_x, arrow_y_end),
                        arrow_color,
                        3,
                        tipLength=0.4
                    )
        
        # Ball Possession Analysis (move before player loop for arrow availability)
        if ENABLE_POSSESSION_ANALYTICS and ball_possession is not None:
            # Collect player positions (using pixel coordinates as proxy for meters)
            player_positions = {}
            team_assignments = {}
            
            if len(results) > 0 and results[0].boxes is not None:
                for box in results[0].boxes:
                    if int(box.cls[0]) == CLASS_ID:  # Person class
                        track_id = int(box.id[0]) if box.id is not None else None
                        if track_id is not None:
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            # Use feet position (bottom center of bbox)
                            player_x = (x1 + x2) / 2
                            player_y = y2
                            player_positions[track_id] = (player_x, player_y)
                            
                            # Get team assignment
                            if ENABLE_TEAM_CLASSIFICATION and team_classifier is not None:
                                if track_id in team_classifier.player_teams:
                                    team_label = team_classifier.player_teams[track_id]
                                    team_assignments[track_id] = team_label
            
            # Update possession analyzer
            ball_position_m = ball_position_pixel if ball_position_pixel else None
            possession_data = ball_possession.update(
                ball_position_m,
                player_positions,
                team_assignments,
                frame_count,
                ball_position_pixel
            )
        
        # Process YOLO results
        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            
            for box in boxes:
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                bbox = (x1, y1, x2, y2)
                class_id = int(box.cls[0])  # Get YOLO class ID
                
                # Get track ID if tracking is enabled
                track_id = None
                if ENABLE_TRACKING and box.id is not None:
                    track_id = int(box.id[0])
                    # Log track ID history
                    if track_id not in track_id_history:
                        track_id_history[track_id] = []
                    track_id_history[track_id].append(frame_count)
                
                # Compute center point
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                
                frame_detections_before += 1
                
                # Only process person detections (class_id=0)
                if class_id != CLASS_ID:
                    continue
                
                # ROI filtering
                keep_detection = True
                if roi_enabled:
                    inside = cv2.pointPolygonTest(pitch_roi, (center_x, center_y), False)
                    if inside < 0:
                        keep_detection = False
                        frame_removed += 1
                        roi_removed.append({
                            'frame': frame_count,
                            'confidence': conf,
                            'bbox': [x1, y1, x2, y2],
                            'center': [center_x, center_y],
                            'track_id': track_id,
                            'reason': 'ROI'
                        })
                
                if keep_detection:
                    # =========================================================
                    # BLACK-BRIGHTNESS HEURISTIC FOR REFEREE DETECTION
                    # Referees typically wear dark uniforms (black)
                    # =========================================================
                    is_referee = False
                    person_type = "Player"
                    
                    # Check cache first
                    if track_id is not None and track_id in track_person_types:
                        person_type, _ = track_person_types[track_id]
                        is_referee = (person_type == "Referee")
                    else:
                        # Extract player crop and check brightness
                        player_crop = color_extractor.extract_player_crop(frame, bbox)
                        if player_crop is not None and player_crop.size > 0:
                            jersey_crop = color_extractor.extract_jersey(player_crop)
                            if jersey_crop is not None and jersey_crop.size > 0:
                                gray = cv2.cvtColor(jersey_crop, cv2.COLOR_BGR2GRAY)
                                avg_brightness = np.mean(gray)
                                
                                # Log brightness values for first few frames
                                if frame_count < 50 and track_id is not None:
                                    print(f"  [BRIGHTNESS] Frame {frame_count} Track {track_id}: brightness={avg_brightness:.1f}")
                                
                                # Dark jersey = likely referee (adjusted threshold)
                                if avg_brightness < 110:
                                    is_referee = True
                                    person_type = "Referee"
                                    if track_id is not None:
                                        track_person_types[track_id] = (person_type, 0.5)
                                    person_classification_stats["referee_count"] += 1
                                    if frame_count < 50:
                                        print(f"  [REFEREE] Frame {frame_count} Track {track_id}: brightness={avg_brightness:.1f}")
                                else:
                                    person_classification_stats["player_count"] += 1
                                    if track_id is not None:
                                        track_person_types[track_id] = (person_type, 0.5)
                    
                    # Referee visualization
                    if is_referee:
                        color = (0, 255, 255)  # Yellow for referees
                        label = f"REF {track_id}" if track_id else "REF"
                        
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(
                            annotated_frame,
                            label,
                            (x1, max(15, y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            color,
                            2
                        )
                        continue  # Skip team classification for referees
                    
                    # Player: Team Classification
                    # Phase 1 - Warmup color collection
                    if ENABLE_TEAM_CLASSIFICATION and frame_count <= WARMUP_FRAMES and track_id is not None:
                        color = color_extractor.get_player_color(frame, bbox)
                        if color is not None:
                            collected_colors.append(color)
                            if track_id not in track_color_samples:
                                track_color_samples[track_id] = []
                            track_color_samples[track_id].append(color)
                    
                    # Phase 2 - Online prediction
                    team_name = "Unknown"
                    if ENABLE_TEAM_CLASSIFICATION and is_classifier_trained and track_id is not None:
                        if track_id in team_classifier.player_teams:
                            team_label = team_classifier.player_teams[track_id]
                        else:
                            color = color_extractor.get_player_color(frame, bbox)
                            team_label = team_classifier.assign_player(track_id, color)
                        team_name = team_classifier.get_team_name(team_label)
                    
                    # Visualization with team classification
                    if ENABLE_TEAM_CLASSIFICATION:
                        annotated_frame = team_visualizer.draw_player(
                            annotated_frame,
                            bbox,
                            track_id if track_id is not None else -1,
                            team_name
                        )
                        
                        # Draw possession arrow if player has ball
                        if ENABLE_POSSESSION_ANALYTICS and possession_data is not None:
                            possessor_id = possession_data.get("possessor_id")
                            if possessor_id == track_id and possession_data.get("state") == "In Possession":
                                # Team color for arrow
                                if team_name == "Blue":
                                    arrow_color = (255, 0, 0)  # Blue in BGR
                                elif team_name == "Red":
                                    arrow_color = (0, 0, 255)  # Red in BGR
                                else:
                                    arrow_color = (128, 128, 128)  # Gray for unknown
                                
                                # Draw enhanced arrow above bbox (larger and more visible)
                                arrow_x = int((x1 + x2) / 2)
                                arrow_y_start = y1 - 35  # Increased from 20
                                arrow_y_end = y1 - 10   # Increased from 5
                                
                                cv2.arrowedLine(
                                    annotated_frame,
                                    (arrow_x, arrow_y_start),
                                    (arrow_x, arrow_y_end),
                                    arrow_color,
                                    5,              # Increased thickness from 3
                                    tipLength=0.5    # Increased from 0.4
                                )
                    else:
                        # Original visualization without team classification
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        label = f"{conf:.2f}"
                        if track_id is not None:
                            label += f" ID:{track_id}"
                        cv2.putText(
                            annotated_frame,
                            label,
                            (x1, max(15, y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 255, 0),
                            2
                        )
                    
                    confidences.append(conf)
                    frame_detections_after += 1
        
        # Train team classifier after warmup phase
        if ENABLE_TEAM_CLASSIFICATION and frame_count == WARMUP_FRAMES and len(collected_colors) >= 2:
            print(f"\n[INFO] Training TeamClassifier on {len(collected_colors)} color samples...")
            team_classifier.fit(collected_colors)
            is_classifier_trained = True
            
            # Pre-assign teams for tracks seen during warmup
            for track_id, colors in track_color_samples.items():
                if colors:
                    avg_color = np.mean(colors, axis=0)
                    team_classifier.assign_player(track_id, avg_color)
            
            print("[INFO] TeamClassifier trained successfully.")

        # Ball Possession Visualization (stats box)
        if ENABLE_POSSESSION_ANALYTICS and ball_possession is not None and possession_data:
            # Draw possession stats box (top-left corner)
            stats_x, stats_y = 10, 10
            cv2.rectangle(annotated_frame, (stats_x, stats_y), (stats_x + 220, stats_y + 100), (0, 0, 0), -1)
            cv2.rectangle(annotated_frame, (stats_x, stats_y), (stats_x + 220, stats_y + 100), (255, 255, 255), 2)
            
            # Get possession percentages
            possession_pct = ball_possession.get_possession_percentage()
            
            # Draw team possession percentages
            y_offset = 25
            for team_key, pct in possession_pct.items():
                if team_key.endswith("_pct"):
                    team_name = team_key.replace("_pct", "")
                    if team_name == "Free_Ball":
                        color = (128, 128, 128)  # Gray
                    elif team_name == "Red":
                        color = (0, 0, 255)  # Red in BGR
                    else:
                        color = (255, 0, 0)  # Blue in BGR
                    
                    text = f"{team_name}: {pct}%"
                    cv2.putText(annotated_frame, text, (stats_x + 10, stats_y + y_offset),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    y_offset += 20
            
            # Draw current possession info
            if possession_data.get("state") == "In Possession":
                possessor_id = possession_data.get("possessor_id")
                team = possession_data.get("team_name", "Unknown")
                
                cv2.putText(annotated_frame, f"Possessor: {possessor_id} ({team})", 
                           (stats_x + 10, stats_y + 95),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        total_detections_before_roi += frame_detections_before
        total_detections_after_roi += frame_detections_after
        total_detections_removed += frame_removed
        frame_count += 1

        # Write frame
        writer.write(annotated_frame)
        
        # Memory cleanup
        del annotated_frame
        if frame_count % 50 == 0:
            torch.cuda.empty_cache()

        # Progress
        if frame_count % 50 == 0:
            elapsed = time.time() - start_time
            current_fps = frame_count / elapsed
            roi_status = f"ROI: {frame_detections_after}/{frame_detections_before}" if roi_enabled else "ROI: disabled"
            team_status = ""
            if ENABLE_TEAM_CLASSIFICATION:
                if frame_count <= WARMUP_FRAMES:
                    team_status = f" | Warmup: {frame_count}/{WARMUP_FRAMES}"
                elif is_classifier_trained:
                    team_status = f" | Teams: {len(team_classifier.player_teams)}"
                else:
                    team_status = " | Training..."
            
            # Person classification status
            person_status = ""
            if ENABLE_PERSON_CLASSIFICATION:
                person_status = f" | Ref: {person_classification_stats['referee_count']} Coach: {person_classification_stats['coach_count']}"
            else:
                # Black-brightness heuristic stats
                person_status = f" | Ref: {person_classification_stats['referee_count']}"
            
            print(f"Frame {frame_count}/{total_frames} | Detections: {frame_detections_after} | {roi_status} | FPS: {current_fps:.2f}{team_status}{person_status}")

# Cleanup
cap.release()
writer.release()

# Ball Interpolation post-processing
if ENABLE_BALL_DETECTION and ball_interpolator is not None and ball_track_history:
    print("\n[INFO] Running ball trajectory interpolation...")
    interpolated_trajectory = ball_interpolator.interpolate(ball_track_history, frame_count)
    print(f"[INFO] Interpolated trajectory: {len(interpolated_trajectory)} frames")
else:
    interpolated_trajectory = []

total_time = time.time() - start_time
avg_fps = frame_count / total_time
avg_inference_time = np.mean(inference_times)
avg_confidence = np.mean(confidences) if confidences else 0.0
avg_detections_per_frame = total_detections_after_roi / frame_count if frame_count > 0 else 0.0
avg_detections_before_roi = total_detections_before_roi / frame_count if frame_count > 0 else 0.0
removal_rate = (total_detections_removed / total_detections_before_roi * 100) if total_detections_before_roi > 0 else 0.0

# Analyze confidence distribution of low-confidence detections
if confidence_below_threshold:
    low_conf_values = [d['confidence'] for d in confidence_below_threshold]
    print(f"\nLow Confidence Analysis:")
    print(f"  Total below threshold: {len(confidence_below_threshold)}")
    print(f"  Min confidence: {min(low_conf_values):.4f}")
    print(f"  Max confidence: {max(low_conf_values):.4f}")
    print(f"  Mean confidence: {np.mean(low_conf_values):.4f}")
    
    # Analyze bbox sizes of low-confidence detections
    low_conf_areas = [d['area'] for d in confidence_below_threshold]
    print(f"  Min bbox area: {min(low_conf_areas)}")
    print(f"  Max bbox area: {max(low_conf_areas)}")
    print(f"  Mean bbox area: {np.mean(low_conf_areas):.2f}")

# Analyze ROI removals
if roi_removed:
    roi_conf_values = [d['confidence'] for d in roi_removed]
    print(f"\nROI Removal Analysis:")
    print(f"  Total removed by ROI: {len(roi_removed)}")
    print(f"  Min confidence: {min(roi_conf_values):.4f}")
    print(f"  Max confidence: {max(roi_conf_values):.4f}")
    print(f"  Mean confidence: {np.mean(roi_conf_values):.4f}")
    
    # Check if high-confidence detections are being removed by ROI
    high_conf_roi_removed = [d for d in roi_removed if d['confidence'] > 0.5]
    if high_conf_roi_removed:
        print(f"  WARNING: {len(high_conf_roi_removed)} high-confidence (>0.5) detections removed by ROI")
        for d in high_conf_roi_removed[:5]:  # Show first 5
            print(f"    Frame {d['frame']}: conf={d['confidence']:.3f}, bbox={d['bbox']}, center={d['center']}")

# Analyze track ID stability if tracking enabled
if ENABLE_TRACKING and track_id_history:
    print(f"\nTRACK ID STABILITY ANALYSIS:")
    print(f"  Total Unique Tracks: {len(track_id_history)}")
    
    stable_tracks = 0
    unstable_tracks = 0
    
    for track_id, frames in track_id_history.items():
        if len(frames) > 1:
            gaps = [frames[i+1] - frames[i] for i in range(len(frames)-1)]
            max_gap = max(gaps) if gaps else 0
            if max_gap <= 5:  # Consider tracks with gaps <= 5 frames as stable
                stable_tracks += 1
            else:
                unstable_tracks += 1
                print(f"  Track {track_id}: {len(frames)} frames, max gap: {max_gap} frames")
    
    print(f"  Stable Tracks (gap <= 5): {stable_tracks}")
    print(f"  Unstable Tracks (gap > 5): {unstable_tracks}")

# Person Classification Summary
if ENABLE_PERSON_CLASSIFICATION:
    print(f"\nPERSON CLASSIFICATION SUMMARY:")
    print(f"  Total classification attempts: {person_classification_stats['total_attempts']}")
    print(f"  Successful classifications: {person_classification_stats['successful_classifications']}")
    print(f"  Low confidence (< {PERSON_CLASSIFIER_CONFIDENCE}): {person_classification_stats['low_confidence_count']}")
    print(f"  Crop extraction failures: {person_classification_stats['crop_failures']}")
    print(f"  Players classified: {person_classification_stats['player_count']}")
    print(f"  Referees detected: {person_classification_stats['referee_count']}")
    print(f"  Coaches detected: {person_classification_stats['coach_count']}")
    print(f"  Unique tracks cached: {len(track_person_types)}")
    
    # Show cached classifications
    if track_person_types:
        print(f"\n  Track Classification Cache:")
        for tid, (ptype, pconf) in sorted(track_person_types.items()):
            print(f"    Track {tid}: {ptype} (conf={pconf:.3f})")

# Ball Possession Summary
if ENABLE_POSSESSION_ANALYTICS and ball_possession is not None:
    print(f"\nBALL POSSESSION SUMMARY:")
    possession_pct = ball_possession.get_possession_percentage()
    possession_summary = ball_possession.get_team_possession_summary()
    
    print(f"  Total frames analyzed: {possession_summary['total_frames']}")
    print(f"  Total duration: {possession_summary['total_duration_seconds']:.2f}s")
    print(f"  Team Possession %:")
    for team, pct in possession_summary['team_possession_pct'].items():
        print(f"    {team}: {pct}%")
    print(f"  Possession Time (seconds):")
    for team, time_sec in possession_summary['total_possession_time_seconds'].items():
        print(f"    {team}: {time_sec}s")

print("\n[4/4] Generating report...")
print("-" * 60)

# Generate report
report = f"""
DETECTION VALIDATION REPORT
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

VIDEO INFORMATION
-----------------
Input Video: {INPUT_VIDEO}
Output Video: {OUTPUT_VIDEO}
Resolution: {width}x{height}
FPS: {fps:.2f}
Total Frames: {total_frames}
Processed Frames: {frame_count}

DETECTION METRICS
-----------------
Total Detections (Before ROI): {total_detections_before_roi}
Total Detections (After ROI): {total_detections_after_roi}
Detections Removed: {total_detections_removed}
Removal Rate: {removal_rate:.2f}%
Average Detections per Frame (Before ROI): {avg_detections_before_roi:.2f}
Average Detections per Frame (After ROI): {avg_detections_per_frame:.2f}
Average Confidence: {avg_confidence:.4f}
Min Confidence: {min(confidences) if confidences else 0:.4f}
Max Confidence: {max(confidences) if confidences else 0:.4f}

ROI FILTERING
------------
ROI Enabled: {roi_enabled}
ROI Source: {roi_source}
Detections Removed by ROI: {len(roi_removed) if 'roi_removed' in locals() else 0}
Detections Below Threshold: {len(confidence_below_threshold) if 'confidence_below_threshold' in locals() else 0}

CONFIDENCE DISTRIBUTION
----------------------
Total Detections (conf < 0.25): {len(confidence_below_threshold) if 'confidence_below_threshold' in locals() else 0}
Min Low Confidence: {min([d['confidence'] for d in confidence_below_threshold]) if confidence_below_threshold else 0:.4f}
Max Low Confidence: {max([d['confidence'] for d in confidence_below_threshold]) if confidence_below_threshold else 0:.4f}
Mean Low Confidence: {np.mean([d['confidence'] for d in confidence_below_threshold]) if confidence_below_threshold else 0:.4f}

TRACKING METRICS
---------------
Tracking Enabled: {ENABLE_TRACKING}
Tracking Method: YOLO native (persist=True)
Total Unique Tracks: {len(track_id_history) if track_id_history else 'N/A'}
Tracker Config: {TRACKER_CONFIG if ENABLE_TRACKING else 'N/A'}

PERSON CLASSIFICATION METRICS
----------------------------
Person Classification Enabled: {ENABLE_PERSON_CLASSIFICATION}
Classifier Model: {PERSON_CLASSIFIER_MODEL_PATH if ENABLE_PERSON_CLASSIFICATION else 'N/A'}
Confidence Threshold: {PERSON_CLASSIFIER_CONFIDENCE if ENABLE_PERSON_CLASSIFICATION else 'N/A'}
Total Classification Attempts: {person_classification_stats['total_attempts'] if ENABLE_PERSON_CLASSIFICATION else 'N/A'}
Successful Classifications: {person_classification_stats['successful_classifications'] if ENABLE_PERSON_CLASSIFICATION else 'N/A'}
Low Confidence Results: {person_classification_stats['low_confidence_count'] if ENABLE_PERSON_CLASSIFICATION else 'N/A'}
Crop Extraction Failures: {person_classification_stats['crop_failures'] if ENABLE_PERSON_CLASSIFICATION else 'N/A'}
Players Classified: {person_classification_stats['player_count'] if ENABLE_PERSON_CLASSIFICATION else 'N/A'}
Referees Detected: {person_classification_stats['referee_count'] if ENABLE_PERSON_CLASSIFICATION else 'N/A'}
Coaches Detected: {person_classification_stats['coach_count'] if ENABLE_PERSON_CLASSIFICATION else 'N/A'}
Unique Tracks Cached: {len(track_person_types) if ENABLE_PERSON_CLASSIFICATION else 'N/A'}

TEAM CLASSIFICATION METRICS
---------------------------
Team Classification Enabled: {ENABLE_TEAM_CLASSIFICATION}
Classifier Trained: {is_classifier_trained}
Color Samples Collected: {len(collected_colors) if ENABLE_TEAM_CLASSIFICATION else 'N/A'}
Players Classified: {len(team_classifier.player_teams) if team_classifier else 'N/A'}
Warmup Frames: {WARMUP_FRAMES if ENABLE_TEAM_CLASSIFICATION else 'N/A'}

BALL DETECTION METRICS
----------------------
Ball Detection Enabled: {ENABLE_BALL_DETECTION}
Ball Detector: {'Loaded' if ball_detector else 'N/A'}
Ball Tracker: {'Active' if ball_tracker else 'N/A'}
Ball Interpolator: {'Active' if ball_interpolator else 'N/A'}
Track History Entries: {len(ball_track_history) if ball_track_history else 'N/A'}
Interpolated Trajectory: {len(interpolated_trajectory) if interpolated_trajectory else 'N/A'}

PERFORMANCE METRICS
-------------------
Total Processing Time: {total_time:.2f} seconds
Average Inference Time: {avg_inference_time*1000:.2f} ms
Average FPS: {avg_fps:.2f}
Device: {DEVICE.upper()}
GPU Model: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}

CONFIGURATION
-------------
Model: {MODEL_PATH}
Confidence Threshold: {CONFIDENCE_THRESHOLD}
Class ID: {CLASS_ID} (Person/Player)
Image Size: 640
ROI Polygon: {pitch_roi.tolist()}
"""

with open(OUTPUT_REPORT, 'w') as f:
    f.write(report)

print(report)
print("\n" + "=" * 60)
print("DETECTION VALIDATION COMPLETE")
print("=" * 60)
print(f"Output Video: {OUTPUT_VIDEO}")
print(f"Report: {OUTPUT_REPORT}")