# Stepout - Technology Stack & Architecture

## 1. Core Frameworks & Libraries

### Deep Learning & Computer Vision
- **PyTorch** - Primary deep learning framework
- **YOLOv8x** - Object detection model (yolov8x.pt)
- **OpenCV (cv2)** - Image/video processing, frame extraction, visualization
- **PIL/Pillow** - Image manipulation and preprocessing
- **NumPy** - Numerical operations and array handling

### Tracking
- **ByteTrack** - Multi-object tracker for player tracking (bytetrack_custom.yaml)
- **Kalman Filter** - Ball position prediction (custom implementation in ball_tracker.py)

### Classification
- **EfficientNet-B0** - Person classification (referee/coach detection) via transfer learning
- **K-Means Clustering** - Team classification based on jersey colors

### Backend & API
- **FastAPI** - REST API framework
- **SQLAlchemy** - Database ORM
- **Streamlit** - Dashboard and visualization frontend
- **Docker** - Containerization

### Data Processing
- **scikit-learn** - Machine learning utilities (KMeans, metrics)
- **pandas** - Data manipulation (implied in analytics modules)

---

## 2. System Components

### A. Detection Pipeline (`app/detection/`)

#### YOLO Detector (`yolo_detector.py`)
- **Model:** YOLOv8x (ultralytics library)
- **Purpose:** Player and ball detection
- **Configuration:**
  - Confidence threshold: 0.25
  - Class ID: 0 (Person)
  - Image size: 1280
  - Device: CUDA/CPU

#### Ball Detector (`ball_detector.py`)
- **Model:** Specialized YOLO for ball detection
- **Configuration:**
  - Confidence threshold: 0.10
  - Image size: 960
  - ROI filtering enabled

---

### B. Tracking System (`app/tracking/`)

#### Player Tracker (`bytetrack.py`)
- **Algorithm:** ByteTrack (YOLO native tracking with persist=True)
- **Features:**
  - ID stability across frames
  - Association by IoU
  - Track management with confidence thresholds

#### Ball Tracker (`ball_tracker.py`)
- **Algorithm:** Custom Kalman Filter-based tracker
- **Components:**
  - `BallKalmanFilter`: State vector [x, y, vx, vy] with constant velocity model
  - `BallTrack`: Track object with history
  - `BallTracker`: Main tracker with motion-consistency gating
- **Key Parameters:**
  - `max_missing_frames`: 45
  - `max_match_dist`: 180.0 pixels
  - `trajectory_len`: 45 frames
  - `kalman_enabled`: True

#### ReID Module (`reid.py`)
- Person re-identification for maintaining track IDs

---

### C. Team Classification (`app/team_classification/`)

#### Color Extractor (`color_extractor.py`)
- Extracts jersey colors from player bounding boxes
- Uses HSV color space for robust color representation

#### Team Classifier (`team_classifier.py`)
- **Algorithm:** K-Means clustering (2 clusters)
- **Method:**
  - Collects color samples during warmup phase (100 frames)
  - Trains KMeans model to identify two dominant team colors
  - Assigns players to teams based on jersey color
- **Team Names:** Red (label 1), Blue (label 0)
- **Fallback:** Majority vote over recent frames for robustness

#### Jersey Classifier (`jersey_classifier.py`)
- Deep learning approach for jersey classification
- Uses EfficientNet-B0 backbone

#### Person Classifier (`inference.py` in `app/classification/`)
- **Model:** EfficientNet-B0 (transfer learning)
- **Purpose:** Referee/Coach detection
- **Method:**
  - Crops player bounding boxes
  - Extracts jersey region
  - Classifies as Player/Referee/Coach
- **Heuristic fallback:** Black-brightness heuristic for referee detection
  - Dark jersey (brightness < 110) → Referee

---

### D. Ball Analytics (`app/analytics/ball_analytics/`)

#### Possession Detection (`possession.py`)
- **Algorithm:** Distance-based possession detection
- **Logic:**
  - Player position: feet (bottom center of bbox)
  - Ball position: center from tracking
  - Distance threshold: 120 pixels
  - Confirmation frames: 10 consecutive frames (anti-flicker)
- **Output:**
  - Possessor ID and team
  - Possession percentages
  - State: "In Possession", "Free Ball", "Contested/Transitioning"

#### Visualization (`visualization.py`)
- Stats box (top-right corner)
- Team possession percentages
- Possessor display
- Possession line (player feet to ball)
- Ball trail (last 45 positions)

#### Other Analytics Modules
- `trajectory.py` - Ball trajectory analysis
- `smoothing.py` - Signal smoothing
- `ball_speed.py` - Ball speed estimation
- `acceleration.py` - Ball acceleration
- `touch_detection.py` - Ball touch detection
- `pass_detection.py` - Pass detection
- `pass_metrics.py` - Passing statistics

---

### E. Player Kinematics (`app/analytics/player_kinematics/`)

- `speed.py` - Player speed estimation
- `acceleration.py` - Acceleration calculation
- `direction.py` - Movement direction
- `trajectory.py` - Player path tracking
- `sprint_detection.py` - Sprint detection
- `smoothing.py` - Data smoothing

---

### F. Homography & Pitch Mapping (`app/homography/`)

- `homography_estimator.py` - Perspective transform estimation
- `coordinate_transform.py` - Pixel to pitch coordinates
- `pitch_model.py` - Pitch geometry model
- `field_config.py` - Pitch dimensions and configuration
- `visualization.py` - Pitch visualization
- `auto_calibration.py` - Automatic homography calibration
- `manual_calibration.py` - Manual calibration tool
- `camera_motion.py` - Camera motion compensation

---

---

### H. AI/LLM Components (`app/ai/`)

- `match_analyst.py` - AI match analyst
- `prompt_builder.py` - LLM prompt construction
- `aggregator.py` - Data aggregation for AI
- `recommendations.py` - Tactical recommendations
- `report_generator.py` - Automated report generation
- `sql_agent.py` - Natural language to SQL

---

### I. API Layer (`app/api/`)

#### FastAPI Routers
- `matches.py` - Match endpoints
- `players.py` - Player endpoints
- `teams.py` - Team endpoints
- `reports.py` - Report generation
- `season.py` - Season analysis
- `admin.py` - Admin functions
- `auth.py` - Authentication

#### Services (`app/api/services/`)
- `match_service.py` - Match business logic
- `player_service.py` - Player business logic
- `team_service.py` - Team business logic

---

### J. Dashboard (`app/dashboard/`)

#### Pages
1. Match Analysis
2. Player Analysis
3. Team Analysis
4. Heatmaps
5. AI Chat
9. Formation Intelligence
10. Pressing Intelligence
11. Tactical Analytics

---

### K. Pipeline (`app/pipeline/`)

- `pipeline_manager.py` - Orchestration
- `stages.py` - Processing stages
- `data_models.py` - Data structures
- `pipeline_logger.py` - Logging

---

## 3. Anti-Flickering Mechanisms

### 1. Confirmation Frames (Ball Possession)
- **Parameter:** `POSSESSION_CONFIRMATION_FRAMES = 10`
- **Method:** Player must be closest to ball for N consecutive frames
- **Purpose:** Prevents rapid possession changes

### 2. Kalman Filter (Ball Tracking)
- **Method:** Predicts next position using constant velocity model
- **Purpose:** Smooths ball trajectory, handles missing detections

### 3. Team Classification History
- **Parameter:** `history_len = 30` frames
- **Method:** Majority vote over recent color samples
- **Purpose:** Stable team assignments despite lighting changes

### 4. Manual Overrides
- **Method:** Manual team assignment for problematic tracks
- **Example:** `MANUAL_TEAM_OVERRIDES = {8: 0, 13: 1, 44: 1}`

### 5. Track ID Stability
- **Method:** ByteTrack with persist=True
- **Metrics:**
  - Gap detection between frames
  - Stability threshold: ≤5 frames gap

---

## 4. Data Flow

```
Video Input (OpenCV)
    ↓
YOLO Detection (Players + Ball)
    ↓
ByteTrack (Player IDs)
    ↓
Ball Detector + Kalman Tracker
    ↓
Color Extraction + KMeans (Team Classification)
    ↓
EfficientNet-B0 (Referee Detection)
    ↓
Distance Calculation (Possession)
    ↓
Homography (Pitch Mapping)
    ↓
Analytics (Speed, Passes, Shots)
    ↓
Visualization (OpenCV + Streamlit)
    ↓
FastAPI (REST Endpoints)
    ↓
Dashboard (Streamlit)
```

---

## 5. Key Models & Their Roles

| Model | Purpose | Location |
|-------|---------|----------|
| YOLOv8x | Object detection (players, ball) | `app/detection/yolo_detector.py` |
| BallDetector (YOLO) | Ball-specific detection | `app/detection/ball_detector.py` |
| EfficientNet-B0 | Referee/Coach classification | `app/classification/` |
| KMeans | Team color clustering | `app/team_classification/team_classifier.py` |
| Kalman Filter | Ball position prediction | `app/tracking/ball_tracker.py` |
| ByteTrack | Multi-object tracking | `app/tracking/bytetrack.py` |

---

## 6. Visualization Stack

### OpenCV (`cv2`)
- Frame annotation
- Bounding boxes
- Arrows and lines
- Circles and trails
- Text rendering

### Streamlit
- Interactive dashboard
- Real-time video playback
- Charts and statistics
- Tactical visualizations

---

## 7. Configuration & Calibration

### ROI (Region of Interest)
- **File:** `configs/pitch_roi.json`
- **Purpose:** Filters detections to pitch area only
- **Format:** Quadrilateral polygon [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]

### Homography
- **File:** `configs/homography_calibration.json`
- **Purpose:** Camera to pitch coordinate mapping

### Tracker Config
- **File:** `app/tracking/bytetrack_custom.yaml`
- **Parameters:** Track buffer, match thresholds

---

## 8. Performance Optimizations

1. **GPU Acceleration**
   - CUDA for YOLO inference
   - FP16 half-precision
   - Model fusion

2. **Frame Processing**
   - Adaptive preprocessing
   - ROI filtering early in pipeline
   - Batch processing where possible

3. **Tracking Efficiency**
   - Kalman prediction reduces search space
   - Motion-consistency gating

4. **Memory Management**
   - Trajectory history limited to 45 frames
   - torch.cuda.empty_cache() every 50 frames

---

## 9. Testing & Validation

### Test Files
- `test_possession_implementation.py` - Possession logic
- `test_homography_speed_distance_integration.py` - Integration tests
- `tests/` - Unit tests for various modules

### Validation Scripts
- `validate_tracking.py` - Tracking accuracy
- `validate_team_classification.py` - Team assignment
- `run_validation_phases.py` - Multi-phase validation
- `master_validation.py` - End-to-end validation

---

## 10. Technology Summary

| Category | Technologies |
|----------|-------------|
| **Language** | Python 3.8+ |
| **Deep Learning** | PyTorch, YOLOv8x, EfficientNet-B0 |
| **Computer Vision** | OpenCV, PIL/Pillow |
| **Tracking** | ByteTrack, Kalman Filter |
| **ML/Analytics** | scikit-learn (KMeans), NumPy, pandas |
| **Backend** | FastAPI, SQLAlchemy |
| **Frontend** | Streamlit |
| **Database** | SQLite/PostgreSQL (configurable) |
| **Deployment** | Docker |
| **Visualization** | Matplotlib (implied), OpenCV |

---

## 11. Anti-Flickering Summary

1. **Possession Confirmation:** 10 consecutive frames required
2. **Kalman Prediction:** Smooths ball trajectory
3. **Team Voting:** 30-frame history majority vote
4. **Manual Overrides:** Human-in-the-loop corrections
5. **Track Stability:** ByteTrack persist=True with gap monitoring

---

*Generated for Stepout Football Analytics Platform*