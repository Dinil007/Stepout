"""
StepOut AI Football Analytics Platform - Enterprise Streamlit Dashboard
"""

import io
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st

try:
    st.set_page_config(
        page_title="StepOut AI Football Analytics Platform",
        page_icon="⚽",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(
        """
        <style>
            .main { background-color: #0b0f19; color: #f8fafc; }
            .stMetric { background-color: #1e293b; padding: 15px; border-radius: 10px; border: 1px solid #334155; }
            .stHeader { color: #38bdf8; }
            .success-box { background-color: #064e3b; padding: 15px; border-radius: 10px; border: 1px solid #065f46; }
            .warning-box { background-color: #78350f; padding: 15px; border-radius: 10px; border: 1px solid #92400e; }
            .error-box { background-color: #7f1d1d; padding: 15px; border-radius: 10px; border: 1px solid #991b1b; }
        </style>
        """,
        unsafe_allow_html=True,
    )
except AttributeError:
    pass

# Constants
OUTPUT_DIR = Path("outputs")
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# ------------------------------------------------------------------
# Authentication
# ------------------------------------------------------------------
USERS = {
    "admin": {"password": "admin123", "role": "Admin"},
    "coach": {"password": "coach123", "role": "Coach"},
    "scout": {"password": "scout123", "role": "Scout"},
    "analyst": {"password": "analyst123", "role": "Analyst"},
}

ROLE_PERMISSIONS = {
    "Admin": ["upload", "process", "view", "export", "settings"],
    "Coach": ["upload", "process", "view", "export"],
    "Scout": ["upload", "process", "view", "export"],
    "Analyst": ["view", "export"],
}


def init_session_state():
    defaults = {
        "authenticated": False,
        "username": None,
        "role": None,
        "page": "Login",
        "processing_status": None,
        "processing_log": [],
        "current_match": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _safe_init_session_state():
    try:
        init_session_state()
    except Exception:
        pass


_safe_init_session_state()


def login(username: str, password: str) -> bool:
    user = USERS.get(username)
    if user and user["password"] == password:
        st.session_state.authenticated = True
        st.session_state.username = username
        st.session_state.role = user["role"]
        st.session_state.page = "Home"
        return True
    return False


def logout():
    st.session_state.authenticated = False
    st.session_state.username = None
    st.session_state.role = None
    st.session_state.page = "Login"
    st.session_state.current_match = None


def has_permission(permission: str) -> bool:
    role = st.session_state.get("role")
    if not role:
        return False
    return permission in ROLE_PERMISSIONS.get(role, [])


# ------------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------------
@st.cache_data
def load_json_artifact(filename: str):
    p = OUTPUT_DIR / filename
    if p.exists():
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def get_video_info(video_path: Path) -> dict:
    """Extract video metadata using OpenCV."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {}
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps if fps > 0 else 0.0
    filesize = video_path.stat().st_size if video_path.exists() else 0
    cap.release()
    return {
        "filename": video_path.name,
        "fps": round(fps, 2),
        "width": width,
        "height": height,
        "frames": frame_count,
        "duration_sec": round(duration, 2),
        "filesize_mb": round(filesize / (1024 * 1024), 2),
    }


def get_first_frame(video_path: Path):
    """Extract first frame as numpy array."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    ret, frame = cap.read()
    cap.release()
    if ret:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return None


def run_pipeline_process(video_path: Path, output_dir: Path, max_frames: int = 500):
    """Run the pipeline in a subprocess with live progress."""
    script_path = Path("run_pipeline.py")
    cmd = [
        sys.executable,
        str(script_path),
        "--video", str(video_path),
        "--output", str(output_dir),
        "--max_frames", str(max_frames),
    ]
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True,
    )
    return process


# ------------------------------------------------------------------
# Pages
# ------------------------------------------------------------------

def page_login():
    st.title("🔐 StepOut AI Platform Login")
    st.markdown("Enter your credentials to access the analytics dashboard.")

    with st.form("login_form"):
        username = st.text_input("Username", placeholder="admin / coach / scout / analyst")
        password = st.text_input("Password", type="password", placeholder="Enter password")
        submit = st.form_submit_button("Sign In", use_container_width=True)

        if submit:
            if login(username, password):
                st.success(f"Welcome, {username}! Role: {st.session_state.role}")
                st.rerun()
            else:
                st.error("Invalid username or password.")


def page_home():
    st.title("🏠 Home Dashboard")
    st.markdown(f"**Welcome back, {st.session_state.username}** | Role: `{st.session_state.role}`")

    # Stats
    analytics = load_json_artifact("analytics.json")
    processed_videos = len(list(OUTPUT_DIR.glob("analytics.json")))

    c1, c2 = st.columns(2)
    with c1:
        st.metric("Total Uploaded Matches", processed_videos)
    with c2:
        st.metric("Total Processed Videos", processed_videos)

    st.markdown("---")
    st.subheader("Quick Actions")
    q1, q2, q3 = st.columns(3)
    with q1:
        if st.button("📹 Upload Video", use_container_width=True):
            if has_permission("upload"):
                st.session_state.page = "Upload Video"
                st.rerun()
            else:
                st.error("Permission denied.")
    with q2:
        if st.button("📊 Results Dashboard", use_container_width=True):
            st.session_state.page = "Results Dashboard"
            st.rerun()
    with q3:
        if st.button("⚙️ Settings", use_container_width=True):
            st.session_state.page = "Settings"
            st.rerun()

    st.markdown("---")
    st.subheader("Recent Matches")
    recent = sorted(OUTPUT_DIR.glob("analytics.json"), key=lambda p: p.stat().st_mtime, reverse=True)[:5]
    if recent:
        for p in recent:
            st.write(f"📄 `{p.parent.name}` — {time.ctime(p.stat().st_mtime)}")
    else:
        st.info("No processed matches yet. Upload a video to begin.")


def page_upload():
    st.title("📹 Upload Video")
    if not has_permission("upload"):
        st.error("You do not have permission to upload videos.")
        return

    with st.form("upload_form"):
        uploaded = st.file_uploader("Choose a video file", type=["mp4", "mov", "avi", "mkv"])
        submitted = st.form_submit_button("Upload")

    if submitted and uploaded is not None:
        save_path = UPLOAD_DIR / uploaded.name
        with open(save_path, "wb") as f:
            f.write(uploaded.getbuffer())
        st.success(f"Uploaded: {uploaded.name}")
        st.session_state.current_match = save_path
        st.session_state.page = "Process Video"
        st.rerun()

    if st.session_state.get("current_match"):
        video_path = Path(st.session_state.current_match)
        if video_path.exists():
            info = get_video_info(video_path)
            st.subheader("Video Details")
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("Filename", info.get("filename", "N/A"))
                st.metric("Duration", f"{info.get('duration_sec', 0)}s")
            with c2:
                st.metric("Resolution", f"{info.get('width')}x{info.get('height')}")
                st.metric("FPS", info.get("fps", 0))
            with c3:
                st.metric("Size", f"{info.get('filesize_mb', 0)} MB")
                st.metric("Frames", info.get("frames", 0))

            st.subheader("Preview (First Frame)")
            frame = get_first_frame(video_path)
            if frame is not None:
                st.image(frame, caption="First Frame", use_column_width=True)

            if st.button("▶ Process Video"):
                st.session_state.page = "Processing"
                st.rerun()


def page_processing():
    st.title("⚙️ Video Processing")
    video_path = st.session_state.get("current_match")
    if not video_path or not Path(video_path).exists():
        st.warning("No video selected. Please upload a video first.")
        if st.button("Go to Upload"):
            st.session_state.page = "Upload Video"
            st.rerun()
        return

    if not st.session_state.get("processing_status"):
        st.info("Ready to process. Click Start to begin.")
        if st.button("Start Processing"):
            st.session_state.processing_status = "running"
            st.session_state.processing_log = []
            st.rerun()
        return

    status = st.session_state.processing_status
    if status == "running":
        progress_bar = st.progress(0, text="Initializing pipeline...")
        log_container = st.container()
        video_path = Path(video_path)
        output_dir = OUTPUT_DIR / f"match_{int(time.time())}"
        output_dir.mkdir(parents=True, exist_ok=True)

        stages = [
            "Loading Video",
            "Detecting Players",
            "Tracking Players",
            "Homography",
            "Pose Estimation",
            "Biomechanics",
            "Football Analytics",
            "AI Report",
            "Completed",
        ]

        log_container.text("🚀 Starting pipeline...")
        process = run_pipeline_process(video_path, output_dir)

        current_stage = 0
        for line in process.stdout:
            line = line.strip()
            if line:
                st.session_state.processing_log.append(line)
                log_container.text(line)
                for idx, stage in enumerate(stages):
                    if stage.lower() in line.lower():
                        current_stage = idx
                        progress_bar.progress(int((idx + 1) / len(stages) * 100), text=f"✓ {stage}")

        process.wait()
        if process.returncode == 0:
            st.session_state.processing_status = "completed"
            progress_bar.progress(100, text="✅ Completed")
            st.success("Processing completed successfully!")
            time.sleep(1)
            st.session_state.page = "Results Dashboard"
            st.rerun()
        else:
            st.session_state.processing_status = "failed"
            st.error("Processing failed. Check logs for details.")
            for log_line in st.session_state.processing_log[-10:]:
                st.text(log_line)


def page_results_dashboard():
    st.title("📊 Results Dashboard")
    analytics = load_json_artifact("analytics.json")
    if not analytics:
        st.warning("No analytics results found. Please process a video first.")
        return

    summary = analytics.get("summary_metrics", {})

    # Team Analytics
    st.subheader("Team Analytics")
    team_analytics = load_json_artifact("team_analytics.json")
    if team_analytics:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("### Team A (Red)")
            team_a = team_analytics.get("Red", team_analytics.get("0", {}))
            st.metric("Ball Possession", f"{team_a.get('possession_pct', 0)}%")
            st.metric("Total Distance", f"{team_a.get('total_distance_m', 0)} m")
            st.metric("Average Speed", f"{team_a.get('avg_speed_kmh', 0)} km/h")
            st.metric("Top Speed", f"{team_a.get('top_speed_kmh', 0)} km/h")
        with c2:
            st.markdown("### Team B (Blue)")
            team_b = team_analytics.get("Blue", team_analytics.get("1", {}))
            st.metric("Ball Possession", f"{team_b.get('possession_pct', 0)}%")
            st.metric("Total Distance", f"{team_b.get('total_distance_m', 0)} m")
            st.metric("Average Speed", f"{team_b.get('avg_speed_kmh', 0)} km/h")
            st.metric("Top Speed", f"{team_b.get('top_speed_kmh', 0)} km/h")
    else:
        st.info("Team analytics not available.")

    # Annotated Video
    st.subheader("Annotated Video")
    annotated_path = OUTPUT_DIR / "annotated_video.mp4"
    if annotated_path.exists():
        st.video(str(annotated_path))

    # Visualizations
    st.subheader("Match Heat Map")
    match_hm = OUTPUT_DIR / "match_heatmap.png"
    if match_hm.exists():
        st.image(str(match_hm), caption="Match Heat Map", use_column_width=True)

    st.subheader("Team Density Maps")
    c1, c2 = st.columns(2)
    with c1:
        team_a_hm = OUTPUT_DIR / "team_a_density_map.png"
        if team_a_hm.exists():
            st.image(str(team_a_hm), caption="Team A Density Map", use_column_width=True)
    with c2:
        team_b_hm = OUTPUT_DIR / "team_b_density_map.png"
        if team_b_hm.exists():
            st.image(str(team_b_hm), caption="Team B Density Map", use_column_width=True)

    st.subheader("Ball Density Map")
    ball_hm = OUTPUT_DIR / "ball_density_map.png"
    if ball_hm.exists():
        st.image(str(ball_hm), caption="Ball Density Map", use_column_width=True)

    st.subheader("Ball Trajectory")
    traj_path = OUTPUT_DIR / "ball_trajectory.png"
    if traj_path.exists():
        st.image(str(traj_path), caption="Complete Ball Trajectory", use_column_width=True)


def page_settings():
    st.title("⚙️ Settings")
    st.subheader("User Profile")
    st.write(f"**Username:** {st.session_state.username}")
    st.write(f"**Role:** {st.session_state.role}")
    st.write(f"**Permissions:** {', '.join(ROLE_PERMISSIONS.get(st.session_state.role, []))}")

    st.markdown("---")
    st.subheader("Application Settings")
    st.info("Additional settings can be added here (model preferences, output directories, display options, etc.)")


# ------------------------------------------------------------------
# Main Navigation Router
# ------------------------------------------------------------------
def _run_app():
    """Execute the main Streamlit app routing."""
    if not st.session_state.get("authenticated"):
        page_login()
    else:
        with st.sidebar:
            st.header("Navigation")
            if st.button("🏠 Home"):
                st.session_state.page = "Home"
            if has_permission("upload") and st.button("📹 Upload Video"):
                st.session_state.page = "Upload Video"
            if has_permission("process") and st.button("⚙️ Process Video"):
                st.session_state.page = "Processing"
            if has_permission("view") and st.button("📊 Results Dashboard"):
                st.session_state.page = "Results Dashboard"
            if st.button("⚙️ Settings"):
                st.session_state.page = "Settings"
            st.markdown("---")
            if st.button("🚪 Logout"):
                logout()
                st.rerun()

        page = st.session_state.page
        if page == "Home":
            page_home()
        elif page == "Upload Video":
            page_upload()
        elif page == "Process Video":
            page_processing()
        elif page == "Processing":
            page_processing()
        elif page == "Results Dashboard":
            page_results_dashboard()
        elif page == "Settings":
            page_settings()
        else:
            page_home()

try:
    _run_app()
except Exception:
    pass
