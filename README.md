# StepOut 🏃

> AI-powered sports analytics platform for computer vision and athlete performance tracking.

## Project Structure

```
stepout/
│
├── app/            # Core application logic
├── backend/        # FastAPI backend server
├── frontend/       # Streamlit / UI frontend
├── models/         # Trained ML/CV models
├── datasets/       # Training & evaluation datasets
├── videos/         # Input video files
├── outputs/        # Processed outputs (annotated videos, CSVs)
├── notebooks/      # Jupyter notebooks for experiments
├── configs/        # Configuration files (YAML, JSON)
├── utils/          # Shared utility functions
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── .env
└── README.md
```

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

## Running

```bash
# Backend
uvicorn backend.main:app --reload

# Frontend
streamlit run frontend/app.py

# Docker
docker-compose up --build
```

## Stack

- **Computer Vision**: OpenCV, MediaPipe, Supervision
- **ML/DL**: PyTorch, scikit-learn
- **Backend**: FastAPI, SQLAlchemy, PostgreSQL
- **Frontend**: Streamlit
- **DevOps**: Docker, Docker Compose
