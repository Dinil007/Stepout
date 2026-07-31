# SPORTA VISTA PRO

<h3 align="center">Football Analytics System - Sports Intelligence Platform</h3>

<p align="center">
  <strong>Computer vision, tactical analytics, match intelligence, and interactive reporting for football video analysis.</strong>
</p>

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white">
  <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white">
  <img alt="OpenCV" src="https://img.shields.io/badge/OpenCV-Computer%20Vision-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white">
  <img alt="FastAPI" src="https://img.shields.io/badge/FastAPI-API-009688?style=for-the-badge&logo=fastapi&logoColor=white">
  <img alt="Streamlit" src="https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white">
</p>

---

## ⚽ Overview

SPORTA VISTA PRO is a football analytics system for processing match video and converting it into structured sports intelligence. It combines player detection, ball tracking, team classification, possession analysis, pitch mapping, tactical metrics, and dashboard-ready visualizations.

The platform is built for match analysts, coaches, sports scientists, and developers who want a complete computer vision pipeline for football performance analysis.

---

## 🎥 Processed Video Preview

Watch the processed Match 1 output here:

[▶ Play Match 1 processed video](https://github.com/Dinil007/Stepout/blob/master/outputs/match1.mp4)

Direct repository path:

`	ext
outputs/match1.mp4
`

> GitHub README pages do not show inline HTML video players, so the play button opens the MP4 file in GitHub's video viewer.

---

## ✨ Core Features

- 🎯 **Player Detection** with YOLO-based object detection
- 🏃 **Multi-Object Player Tracking** with ByteTrack
- ⚽ **Ball Detection & Kalman Tracking** for smoother ball paths
- 👕 **Team Classification** using jersey color clustering
- 🧍 **Referee & Staff Detection** with EfficientNet-B0 classification
- 📊 **Possession Analytics** with anti-flicker confirmation logic
- 🗺️ **Pitch Mapping** through homography transformation
- 🚀 **Player Kinematics** for speed, acceleration, sprinting, and trajectory analysis
- 🧠 **Tactical Intelligence** for formations, pressing, and match patterns
- 🖥️ **Interactive Dashboard** with Streamlit visual analytics
- 🔌 **REST API Layer** with FastAPI services

---

## 🧰 Technology Stack

| Area | Icons | Technologies |
|------|-------|--------------|
| Language | 🐍 | Python 3.8+ |
| Deep Learning | 🔥 🧠 | PyTorch, YOLOv8/YOLO11, EfficientNet-B0 |
| Computer Vision | 👁️ 🎞️ | OpenCV, PIL/Pillow |
| Tracking | 🛰️ 📍 | ByteTrack, Kalman Filter, ReID |
| Analytics | 📊 🧮 | NumPy, pandas, scikit-learn |
| Tactical Models | ⚽ 🗺️ | formations, pressing analytics |
| Backend | ⚡ 🗄️ | FastAPI, SQLAlchemy |
| Dashboard | 🖥️ 📈 | Streamlit, OpenCV visual overlays |
| Data & Config | 🧾 ⚙️ | YAML, JSON, SQLite/PostgreSQL |
| Deployment | 🐳 🚢 | Docker, docker-compose |
| Testing | ✅ 🧪 | pytest validation suite |

---

## 🏗️ System Architecture

```text
Video Input
    ↓
Frame Preprocessing
    ↓
YOLO Detection
    ↓
Player Tracking + Ball Tracking
    ↓
Team Classification + Referee Detection
    ↓
Possession + Kinematics + Tactical Analytics
    ↓
Pitch Mapping + Visual Overlays
    ↓
Processed Video + Reports + Dashboard
    ↓
FastAPI Services
```

---

## 🧩 Main Modules

### 🎯 Detection Pipeline

Located in `app/detection/`

- `yolo_detector.py` - player and object detection
- `ball_detector.py` - football-specific detection
- `detection_filter.py` - confidence, class, and ROI filtering

### 🛰️ Tracking System

Located in `app/tracking/`

- `bytetrack.py` - player ID tracking
- `ball_tracker.py` - Kalman-based ball tracking
- `reid.py` - person re-identification support
- `tracker_config.py` - tracker thresholds and behavior

### 👕 Team Classification

Located in `app/team_classification/`

- `color_extractor.py` - jersey color extraction
- `team_classifier.py` - K-Means team assignment
- `jersey_classifier.py` - deep learning jersey classification
- `visualize_teams.py` - team visualization utilities

### 📊 Analytics Engine

Located in `app/analytics/`

- `ball_analytics/` - possession, pass detection, speed, trajectory, touches
- `player_kinematics/` - speed, acceleration, sprinting, direction, smoothing
- `pressing_engine.py` - pressing intelligence
- `tactical_engine.py` - tactical match analysis

### 🗺️ Homography & Pitch Mapping

Located in `app/homography/`

- `homography_estimator.py` - perspective transform estimation
- `coordinate_transform.py` - pixel-to-pitch conversion
- `pitch_model.py` - pitch geometry
- `manual_calibration.py` - manual calibration workflow
- `auto_calibration.py` - automatic calibration support

### 🖥️ Dashboard & API

- `app/dashboard/` - Streamlit analytics dashboard
- `streamlit/pages/` - formation, pressing, and tactical intelligence pages
- `app/api/` - FastAPI routers, schemas, services, auth, and admin routes

---

## 📁 Project Structure

```text
stepout/
├── app/
│   ├── analytics/
│   ├── api/
│   ├── classification/
│   ├── dashboard/
│   ├── detection/
│   ├── homography/
│   ├── pipeline/
│   ├── pose/
│   ├── preprocessing/
│   ├── team_classification/
│   ├── tracking/
│   └── visualization/
├── configs/
├── docs/
├── outputs/
│   └── match1.mp4
├── scripts/
├── streamlit/
├── tests/
├── config.yaml
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
├── run_pipeline.py
├── streamlit_app.py
└── README.md
```

---

## 🚀 Quick Start

### 1. Clone and enter the project

```bash
git clone <repo-url>
cd stepout
```

### 2. Create a virtual environment

```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the pipeline

```bash
python run_pipeline.py
```

### 5. Launch the dashboard

```bash
streamlit run streamlit_app.py
```

### 6. Run with Docker

```bash
docker-compose up --build
```

---

## ⚙️ Configuration

### ⚽ Ball Detection

```python
ENABLE_BALL_DETECTION = True
BALL_CONFIDENCE_THRESHOLD = 0.05
BALL_IMAGE_SIZE = 1280
BALL_MAX_MATCH_DIST = 200.0
BALL_MAX_MISSING_FRAMES = 60
BALL_INTERPOLATION_MAX_GAP = 30
```

### 👕 Team Classification

```python
ENABLE_TEAM_CLASSIFICATION = True
WARMUP_FRAMES = 0
TEAM_CLASSIFIER_HISTORY_LEN = 30
MANUAL_TEAM_OVERRIDES = {
    8: 0,
    13: 1,
    44: 1,
}
```

### 📊 Possession Analytics

```python
ENABLE_POSSESSION_ANALYTICS = True
POSSESSION_RADIUS_M = 150.0
POSSESSION_CONFIRMATION_FRAMES = 3
```

---

## 🛡️ Anti-Flicker Logic

- ✅ **Possession confirmation** requires stable ownership across consecutive frames
- 🎯 **Kalman prediction** smooths missing or noisy ball detections
- 👕 **Team history voting** reduces jersey-classification noise
- 🧭 **Motion-consistency gating** rejects unrealistic ball jumps
- 🧑‍💻 **Manual overrides** allow analyst correction for difficult tracks

---

## 📈 Output Artifacts

| Output | Purpose |
|--------|---------|
| `outputs/match1.mp4` | Processed match video with visual overlays |
| `outputs/detection_report.txt` | Detection and tracking summary |
| `outputs/analytics.json` | Structured analytics export |
| `outputs/frames/` | Optional extracted or annotated frame outputs |

---

## 🧪 Testing & Validation

```bash
pytest
```

Important validation areas:

- 🧪 Detection accuracy
- 🛰️ Tracking stability
- ⚽ Ball tracking and interpolation
- 👕 Team classification
- 🗺️ Homography calibration
- 🧠 Tactical and pressing intelligence

---

## 📚 Documentation

- `TECHNOLOGY_STACK.md` - full technology and architecture notes
- `docs/ARCHITECTURE.md` - system architecture
- `docs/platform_architecture.md` - platform design
- `docs/AI_MATCH_ANALYST.md` - match analyst module notes
- `FINAL_500_FRAME_VALIDATION.md` - validation summary

---

## 📌 Status

SPORTA VISTA PRO is structured as a modular football analytics platform with production-ready components for detection, tracking, analytics, visualization, API delivery, and dashboard reporting.

---

## 📄 License

MIT License
