# StepOut 🏃

> AI-powered sports analytics platform for computer vision and athlete performance tracking.

## Overview

StepOut is a comprehensive computer vision and video analytics platform for football/soccer analysis. It provides real-time player detection, ball tracking, team classification, possession analysis, and tactical intelligence using state-of-the-art deep learning models.

---

## Core Features

- **Real-time Player Detection** using YOLOv8x
- **Ball Detection & Tracking** with Kalman Filter
- **Multi-Object Player Tracking** with ByteTrack
- **Team Classification** using Jersey Color Clustering (K-Means)
- **Ball Possession Analysis** & Visualization
- **Referee Detection** using EfficientNet-B0
- **Pitch Mapping** via Homography Transformation
- **Player Kinematics** (Speed, Acceleration, Trajectory)
- **Tactical Formation Analysis**
- **Anti-Flickering Mechanisms** for Stable Tracking

---

## Technology Stack

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
- **pandas** - Data manipulation

---

## System Components

### Detection Pipeline (`app/detection/`)

#### YOLO Detector (`yolo_detector.py`)
- **Model:** YOLOv8/YOLO11 (ultralytics library)
- **Purpose:** Player and ball detection
- **Configuration:**
  - Confidence threshold: 0.25
  - Class ID: 0 (Person)
  - Image size: 1280
  - Device: CUDA/CPU

#### Ball Detector (`ball_detector.py`)
- **Model:** Specialized YOLO for ball detection
- **Configuration:**
  - Confidence threshold: 0.05
  - Image size: 1280
  - ROI filtering enabled

---

### Tracking System (`app/tracking/`)

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
  - `max_missing_frames`: 60
  - `max_match_dist`: 200.0 pixels
  - `trajectory_len`: 45 frames
  - `kalman_enabled`: True

#### ReID Module (`reid.py`)
- Person re-identification for maintaining track IDs

---

### Team Classification (`app/team_classification/`)

#### Color Extractor (`color_extractor.py`)
- Extracts jersey colors from player bounding boxes
- Uses HSV color space for robust color representation

#### Team Classifier (`team_classifier.py`)
- **Algorithm:** K-Means clustering (2 clusters)
- **Method:**
  - Collects color samples during warmup phase (0 frames for immediate classification)
  - Trains KMeans model to identify two dominant team colors
  - Assigns players to teams based on jersey color
- **Team Names:** Red (label 0), Blue (label 1)
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

### Ball Analytics (`app/analytics/ball_analytics/`)

#### Possession Detection (`possession.py`)
- **Algorithm:** Distance-based possession detection
- **Logic:**
  - Player position: feet (bottom center of bbox)
  - Ball position: center from tracking
  - Distance threshold: 150 pixels
  - Confirmation frames: 3 consecutive frames (anti-flicker)
- **Output:**
  - Possessor ID and team
  - Possession percentages
  - State: "In Possession", "Free Ball", "Contested/Transitioning"

#### Visualization (`visualization.py`)
- Stats box (top-left corner)
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

### Player Kinematics (`app/analytics/player_kinematics/`)

- `speed.py` - Player speed estimation
- `acceleration.py` - Acceleration calculation
- `direction.py` - Movement direction
- `trajectory.py` - Player path tracking
- `sprint_detection.py` - Sprint detection
- `smoothing.py` - Data smoothing

---

### Homography & Pitch Mapping (`app/homography/`)

- `homography_estimator.py` - Perspective transform estimation
- `coordinate_transform.py` - Pixel to pitch coordinates
- `pitch_model.py` - Pitch geometry model
- `field_config.py` - Pitch dimensions and configuration
- `visualization.py` - Pitch visualization
- `auto_calibration.py` - Automatic homography calibration
- `manual_calibration.py` - Manual calibration tool
- `camera_motion.py` - Camera motion compensation

---

## Data Flow

```text
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
Analytics (Speed, Passes, Shots, xG)
    ↓
Visualization (OpenCV + Streamlit)
    ↓
FastAPI (REST Endpoints)
    ↓
Dashboard (Streamlit)
```

---

## Anti-Flickering Mechanisms

### 1. Confirmation Frames (Ball Possession)
- **Parameter:** `POSSESSION_CONFIRMATION_FRAMES = 3`
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

## Key Models & Their Roles

| Model | Purpose | Location |
|-------|---------|----------|
| YOLOv8/YOLO11 | Object detection (players, ball) | `app/detection/yolo_detector.py` |
| BallDetector (YOLO) | Ball-specific detection | `app/detection/ball_detector.py` |
| EfficientNet-B0 | Referee/Coach classification | `app/classification/` |
| KMeans | Team color clustering | `app/team_classification/team_classifier.py` |
| Kalman Filter | Ball position prediction | `app/tracking/ball_tracker.py` |
| ByteTrack | Multi-object tracking | `app/tracking/bytetrack.py` |

---

## Project Structure

```
stepout/
│
├── app/                        # Core application logic
│   ├── detection/              # YOLO detection modules
│   │   ├── yolo_detector.py
│   │   ├── ball_detector.py
│   │   └── detection_filter.py
│   ├── tracking/               # Tracking modules
│   │   ├── bytetrack.py
│   │   ├── ball_tracker.py
│   │   ├── reid.py
│   │   └── tracker_config.py
│   ├── team_classification/    # Team classification
│   │   ├── team_classifier.py
│   │   ├── color_extractor.py
│   │   ├── jersey_classifier.py
│   │   └── visualize_teams.py
│   ├── analytics/              # Analytics modules
│   │   ├── ball_analytics/
│   │   ├── player_kinematics/
│   │   └── ball_possession.py
│   ├── homography/             # Pitch mapping
│   ├── classification/         # Person classification
│   ├── pose/                   # Pose estimation
│   ├── preprocessing/          # Video preprocessing
│   └── visualization/          # Visualization tools
├── backend/                    # FastAPI backend server
├── frontend/                   # Streamlit / UI frontend
├── models/                     # Trained ML/CV models
│   ├── yolov8x.pt
│   ├── yolov8n.pt
│   └── yolov8m.pt
├── datasets/                   # Training & evaluation datasets
│   └── person_classifier/
├── videos/                     # Input video files
│   └── raw/
├── outputs/                    # Processed outputs
│   ├── detected_video.mp4
│   └── detection_report.txt
├── configs/                    # Configuration files
│   ├── pitch_roi.json
│   ├── homography_calibration.json
│   └── bytetrack_custom.yaml
├── scripts/                    # Utility scripts
├── notebooks/                  # Jupyter notebooks
├── utils/                      # Shared utility functions
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── .env
└── README.md
```

---

## Setup

```bash
# Clone the repo
git clone <repo-url>
cd stepout

# Create virtual environment
python -m venv venv
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Copy env template and fill in values
cp .env.example .env
```

---

## Running

```bash
# Run detection and validation
python validate_detection_only.py

# Backend
uvicorn backend.main:app --reload

# Frontend
streamlit run frontend/app.py

# Docker
docker-compose up --build
```

---

## Configuration

### Ball Detection Configuration
```python
ENABLE_BALL_DETECTION = True
BALL_CONFIDENCE_THRESHOLD = 0.05
BALL_IMAGE_SIZE = 1280
BALL_MAX_MATCH_DIST = 200.0
BALL_MAX_MISSING_FRAMES = 60
BALL_INTERPOLATION_MAX_GAP = 30
```

### Team Classification Configuration
```python
ENABLE_TEAM_CLASSIFICATION = True
WARMUP_FRAMES = 0  # Immediate classification
TEAM_CLASSIFIER_HISTORY_LEN = 30
MANUAL_TEAM_OVERRIDES = {
    8: 0,   # Track 8 -> Red
    13: 1,  # Track 13 -> Blue
    44: 1,  # Track 44 -> Blue
}
```

### Ball Possession Configuration
```python
ENABLE_POSSESSION_ANALYTICS = True
POSSESSION_RADIUS_M = 150.0
POSSESSION_CONFIRMATION_FRAMES = 3
```

---

## Performance Optimizations

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

## Visualization Stack

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

## Testing & Validation

### Test Files
- `tests/test_ball_detector.py` - Ball detection tests
- `tests/test_possession_implementation.py` - Possession logic
- `tests/test_homography_speed_distance_integration.py` - Integration tests

### Validation Scripts
- `validate_tracking.py` - Tracking accuracy
- `validate_team_classification.py` - Team assignment
- `run_validation_phases.py` - Multi-phase validation
- `master_validation.py` - End-to-end validation

---

## Technology Summary

| Category | Technologies |
|----------|-------------|
| **Language** | Python 3.8+ |
| **Deep Learning** | PyTorch, YOLOv8/YOLO11, EfficientNet-B0 |
| **Computer Vision** | OpenCV, PIL/Pillow |
| **Tracking** | ByteTrack, Kalman Filter |
| **ML/Analytics** | scikit-learn (KMeans), NumPy, pandas |
| **Backend** | FastAPI, SQLAlchemy |
| **Frontend** | Streamlit |
| **Database** | SQLite/PostgreSQL (configurable) |
| **Deployment** | Docker |
| **Visualization** | Matplotlib, OpenCV |

---

## License

MIT License
