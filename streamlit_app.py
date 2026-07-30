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
import plotly.express as px
import streamlit as st
import torch

from app.ai.match_analyst import MatchAnalyst
from app.analytics.xg_engine import XGEngine

st.set_page_config(
    page_title="StepOut AI Football Analytics Platform",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main { background-color: #0b0f19; color: #f8fafc; }
    .stMetric { background-color: #1e293b; padding: 15px; border-radius: 10px; border: 1px solid #334155; }
    .stHeader { color: #38bdf8; }
    .success-box { background-color: #064e3b; padding: 15px; border-radius: 10px; border: 1px solid #065f46; }
    .warning-box { background-color: #78350f; padding: 15px; border-radius: 10px; border: 1px solid #92400e; }
    .error-box { background-color: #7f1d1d; padding: 15px; border-radius: 10px; border: 1px solid #991b1b; }
</style>
""", unsafe_allow_html=True)

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


init_session_state()


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
    reports = len(list(OUTPUT_DIR.glob("ai_match_summary.md")))

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Total Uploaded Matches", processed_videos)
    with c2:
        st.metric("Total Processed Videos", processed_videos)
    with c3:
        st.metric("Total Reports Generated", reports)

    st.markdown("---")
    st.subheader("Quick Actions")
    q1, q2, q3, q4 = st.columns(4)
    with q1:
        if st.button("📹 Upload Video", use_container_width=True):
            if has_permission("upload"):
                st.session_state.page = "Upload Video"
                st.rerun()
            else:
                st.error("Permission denied.")
    with q2:
        if st.button("📊 Analytics Dashboard", use_container_width=True):
            st.session_state.page = "Results Dashboard"
            st.rerun()
    with q3:
        if st.button("🤖 AI Match Reports", use_container_width=True):
            st.session_state.page = "AI Match Reports"
            st.rerun()
    with q4:
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

    match_info = analytics.get("match_info", {})
    summary = analytics.get("summary_metrics", {})

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Total Players Tracked", summary.get("total_players_tracked", 0))
    with c2:
        st.metric("Team A Distance", f"{summary.get('team_A_total_distance_m', 0)} m")
    with c3:
        st.metric("Team B Distance", f"{summary.get('team_B_total_distance_m', 0)} m")
    with c4:
        top_speed = summary.get("top_speed_kmh", 0)
        st.metric("Top Speed", f"{top_speed} km/h")

    st.subheader("Match Video")
    for vid_file in ["tracking.mp4", "pitch_view.mp4", "detection.mp4", "team_classification.mp4"]:
        vid_path = OUTPUT_DIR / vid_file
        if vid_path.exists():
            st.video(str(vid_path))

    st.subheader("Heatmap")
    hm_path = OUTPUT_DIR / "heatmap.png"
    if hm_path.exists():
        st.image(str(hm_path), caption="Spatial Density Heatmap", use_column_width=True)

    st.subheader("Player Statistics")
    csv_path = OUTPUT_DIR / "player_statistics.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        st.dataframe(df, use_container_width=True)

    if st.button("View Full Analytics"):
        st.session_state.page = "Analytics Dashboard"
        st.rerun()


def page_ai_reports():
    st.title("🤖 AI Match Reports")
    analyst = MatchAnalyst(output_dir=OUTPUT_DIR)

    try:
        context = analyst.build_context()
        players = context.get("players", {})
        teams = context.get("teams", {})
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Detected Players", len(players))
        with c2:
            st.metric("Teams", len(teams))
        with c3:
            st.metric("Pass Events", len(context.get("events", {}).get("passes", [])))
        with c4:
            st.metric("Shot Events", len(context.get("events", {}).get("shots", [])))

        if st.button("Generate Full AI Report"):
            with st.spinner("Generating structured AI analysis..."):
                st.session_state["ai_report"] = analyst.generate_match_report()

        report = st.session_state.get("ai_report")
        if report:
            st.subheader("Match Summary")
            st.markdown(report.get("match_summary_markdown", ""))

            with st.expander("Team Analysis", expanded=True):
                for team, team_report in report.get("team_analysis", {}).items():
                    st.markdown(f"### {team}")
                    st.json(team_report)

            with st.expander("Player Reports", expanded=False):
                player_ids = sorted(report.get("player_reports", {}).keys(), key=lambda pid: int(pid) if pid.isdigit() else pid)
                if player_ids:
                    selected_player = st.selectbox("Player", player_ids)
                    st.json(report["player_reports"][selected_player])

            with st.expander("Coach Report", expanded=False):
                st.markdown(report.get("coach_report_markdown", ""))

            with st.expander("Opposition Analysis", expanded=False):
                st.markdown(report.get("opposition_report_markdown", ""))

            st.subheader("Ask AI")
            question = st.text_input("Ask about this match...", key="ai_question")
            if st.button("Generate Answer") and question:
                with st.spinner("Answering from aggregated analytics..."):
                    st.session_state["ai_answer"] = analyst.query(question)
            if st.session_state.get("ai_answer"):
                st.markdown(st.session_state["ai_answer"].get("answer", ""))

            st.subheader("Download AI Reports")
            for file_name in [
                "ai_match_summary.md",
                "ai_match_summary.pdf",
                "ai_team_report.json",
                "ai_player_reports.json",
                "coach_report.md",
                "opposition_report.md",
                "recommendations.json",
                "ai_validation_report.json",
                "ai_performance_report.json",
            ]:
                file_path = OUTPUT_DIR / file_name
                if file_path.exists():
                    if file_path.suffix == ".pdf":
                        with open(file_path, "rb") as f:
                            data = f.read()
                    else:
                        with open(file_path, "r", encoding="utf-8") as f:
                            data = f.read()
                    st.download_button(
                        label=f"Download {file_name}",
                        data=data,
                        file_name=file_name,
                    )
        else:
            st.info("Generate a full AI report to view analyst sections and downloads.")
    except FileNotFoundError as exc:
        st.error(f"Required analytics artifact missing: {exc}")
    except Exception as exc:
        st.error(f"AI analyst failed: {exc}")


def page_analytics_dashboard():
    st.title("📈 Analytics Dashboard")
    tabs = st.tabs([
        "Match Overview",
        "Pass Network",
        "Player Analytics",
        "Team Analytics",
        "Expected Goals (xG)",
        "Expected Assists (xA)",
        "Expected Threat (xT)",
        "Formation Intelligence",
        "Pressing Intelligence",
        "Match Timeline",
        "Video Player",
        "Downloads",
    ])

    with tabs[0]:
        st.header("Match Intelligence Overview")
        pass_sum = load_json_artifact("pass_summary.json")
        shot_sum = load_json_artifact("shot_summary.json")
        team_poss = load_json_artifact("team_possession_summary.json")

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Total Passes", pass_sum.get("total_passes", 0))
        with c2:
            st.metric("Pass Completion", f"{pass_sum.get('overall_accuracy_pct', 0.0)}%")
        with c3:
            st.metric("Total Shots", shot_sum.get("total_shots", 0))
        with c4:
            st.metric("Shots On Target", shot_sum.get("shots_on_target", 0))

        st.subheader("Team Possession Breakdown")
        poss_pct = team_poss.get("team_possession_pct", {"Red": 0.0, "Blue": 0.0, "Free Ball": 100.0})
        st.bar_chart(pd.DataFrame([poss_pct]))

        st.subheader("Field Density Heatmap")
        hm_path = OUTPUT_DIR / "heatmap.png"
        if hm_path.exists():
            st.image(str(hm_path), caption="FIFA 105m x 68m Field Density Heatmap", use_column_width=True)

    with tabs[1]:
        st.header("Tactical Pass Network & Shape Intelligence")
        team_pass_sum = load_json_artifact("team_passing_summary.json")
        avg_pos_sum = load_json_artifact("average_positions.json")

        team_sel = st.selectbox("Select Team Filter", ["All Teams", "Red Team", "Blue Team"])
        col1, col2 = st.columns([2, 1])

        with col1:
            if team_sel == "Red Team":
                img_file = "pass_network_red.png"
            elif team_sel == "Blue Team":
                img_file = "pass_network_blue.png"
            else:
                img_file = "pass_network.png"
            net_img_path = OUTPUT_DIR / img_file
            if net_img_path.exists():
                st.image(str(net_img_path), caption=f"2D Pitch Pass Network ({team_sel})", use_column_width=True)
            else:
                st.info("Pass network diagram generating...")

        with col2:
            st.subheader("Tactical Team Shape")
            sel_key = "Red" if team_sel == "Red Team" else ("Blue" if team_sel == "Blue Team" else "Red")
            t_shape = team_pass_sum.get(sel_key, {}).get("tactical_shape", {})
            st.metric("Team Width", f"{t_shape.get('width_m', 0.0)} m")
            st.metric("Team Depth", f"{t_shape.get('depth_m', 0.0)} m")
            st.metric("Compactness Ratio", t_shape.get("compactness", 0.0))
            st.metric("Defensive Line Height", f"{t_shape.get('defensive_line_height_m', 0.0)} m")
            st.metric("Midfield Line Height", f"{t_shape.get('midfield_line_height_m', 0.0)} m")

        st.subheader("Average Player Positions")
        if avg_pos_sum:
            st.dataframe(pd.DataFrame(avg_pos_sum).T, use_container_width=True)

    with tabs[2]:
        st.header("Player Performance Telemetry")
        pass_sum = load_json_artifact("pass_summary.json")
        player_passes = pass_sum.get("player_pass_summary", {})
        player_ids = list(player_passes.keys()) if player_passes else ["3", "8", "42"]
        sel_player = st.selectbox("Select Player Track ID", player_ids)
        p_data = player_passes.get(sel_player, {})
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Passes Attempted", p_data.get("attempted", 0))
        with c2:
            st.metric("Passes Completed", p_data.get("completed", 0))
        with c3:
            st.metric("Pass Accuracy", f"{p_data.get('accuracy_pct', 0.0)}%")
        with c4:
            st.metric("Avg Pass Distance", f"{p_data.get('avg_distance_m', 0.0)} m")

        st.subheader("Biomechanics & Pose Estimation Sample")
        pose_sample = OUTPUT_DIR / "pose_sample.png"
        if pose_sample.exists():
            st.image(str(pose_sample), caption=f"Player #{sel_player} MediaPipe Pose Landmarks", width=300)

    with tabs[3]:
        st.header("Team Tactical Analytics")
        pass_sum = load_json_artifact("pass_summary.json")
        team_p_sum = pass_sum.get("team_pass_summary", {})
        if team_p_sum:
            st.dataframe(pd.DataFrame(team_p_sum).T, use_container_width=True)
        else:
            st.info("No team passing data recorded for current match clip.")

    with tabs[4]:
        st.header("Expected Goals (xG)")
        xg_engine = XGEngine(output_dir=OUTPUT_DIR)
        try:
            if st.button("Generate xG Analysis") or not (OUTPUT_DIR / "xg_summary.json").exists():
                with st.spinner("Calculating expected goals..."):
                    st.session_state["xg_payload"] = xg_engine.run()

            payload = st.session_state.get("xg_payload") or {
                "summary": load_json_artifact("xg_summary.json"),
                "team_summary": load_json_artifact("team_xg_summary.json"),
                "player_summary": load_json_artifact("player_xg_summary.json"),
                "shots": load_json_artifact("xg_shots.json") or [],
            }

            summary = payload.get("summary", {})
            team_summary = payload.get("team_summary", {})
            player_summary = payload.get("player_summary", {})
            shots = payload.get("shots", [])

            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.metric("Total xG", summary.get("total_xg", 0.0))
            with c2:
                st.metric("Shots", summary.get("total_shots", 0))
            with c3:
                st.metric("Average xG", summary.get("average_xg", 0.0))
            with c4:
                highest = summary.get("highest_xg_shot") or {}
                st.metric("Highest xG Shot", highest.get("xg", 0.0))

            st.subheader("Team xG")
            if team_summary:
                st.dataframe(pd.DataFrame(team_summary).T, use_container_width=True)
            else:
                st.info("No xG team data yet. Generate xG after shot events are available.")

            st.subheader("Player xG")
            if player_summary:
                st.dataframe(pd.DataFrame(player_summary).T, use_container_width=True)

            st.subheader("Shot Filters")
            team_options = ["All"] + sorted({str(shot.get("team")) for shot in shots})
            player_options = ["All"] + sorted({str(shot.get("player")) for shot in shots}, key=lambda pid: int(pid) if pid.isdigit() else pid)
            fc1, fc2 = st.columns(2)
            with fc1:
                selected_team = st.selectbox("Team", team_options)
            with fc2:
                selected_player = st.selectbox("Player", player_options)

            filtered = shots
            if selected_team != "All":
                filtered = [shot for shot in filtered if str(shot.get("team")) == selected_team]
            if selected_player != "All":
                filtered = [shot for shot in filtered if str(shot.get("player")) == selected_player]

            st.subheader("Shot Table")
            if filtered:
                st.dataframe(pd.DataFrame(filtered), use_container_width=True)
            else:
                st.info("No shots match the selected filters.")

            st.subheader("Visualisations")
            for file_name in ["xg_shot_map.png", "xg_timeline.png", "team_xg_chart.png", "player_xg_chart.png"]:
                image_path = OUTPUT_DIR / file_name
                if image_path.exists():
                    st.image(str(image_path), caption=file_name, use_column_width=True)

            st.subheader("Top Finishers")
            st.json(summary.get("top_finishers", []))
            st.subheader("Lowest xG Shot")
            st.json(summary.get("lowest_xg_shot"))
        except Exception as exc:
            st.error(f"xG analysis failed: {exc}")

    with tabs[5]:
        st.info("Expected Assists (xA) analysis available in the dedicated page.")
        if st.button("Open xA Page"):
            st.switch_page("streamlit/pages/xA.py") if Path("streamlit/pages/xA.py").exists() else st.info("xA page coming soon.")

    with tabs[6]:
        st.info("Expected Threat (xT) analysis available in the dedicated page.")
        if st.button("Open xT Page"):
            st.switch_page("streamlit/pages/xT.py") if Path("streamlit/pages/xT.py").exists() else st.info("xT page coming soon.")

    with tabs[7]:
        st.info("Formation Intelligence available in the dedicated page.")
        if st.button("Open Formation Intelligence"):
            st.switch_page("streamlit/pages/9_Formation_Intelligence.py")

    with tabs[8]:
        st.info("Pressing Intelligence available in the dedicated page.")
        if st.button("Open Pressing Intelligence"):
            st.switch_page("streamlit/pages/10_Pressing_Intelligence.py")

    with tabs[9]:
        st.header("Match Chronological Timeline")
        pass_events = load_json_artifact("pass_events.json")
        shot_events = load_json_artifact("shot_events.json")

        events = []
        for p in pass_events:
            events.append({
                "Frame": p.get("frame_start"),
                "Event Type": "Pass",
                "Details": f"Player #{p.get('passer')} -> #{p.get('receiver')} ({p.get('pass_type')})",
                "Team": p.get("team"),
                "Distance (m)": p.get("distance_m")
            })
        for s in shot_events:
            events.append({
                "Frame": s.get("frame"),
                "Event Type": "Shot",
                "Details": f"Player #{s.get('player_id')} ({s.get('shot_type')})",
                "Team": s.get("team"),
                "Distance (m)": s.get("distance_m")
            })

        if events:
            df_events = pd.DataFrame(events).sort_values(by="Frame")
            st.dataframe(df_events, use_container_width=True)
        else:
            st.info("No timeline events recorded.")

    with tabs[10]:
        st.header("Multi-Stream Video Player")
        vid_choice = st.selectbox("Select Processed Stream", ["tracking.mp4", "pitch_view.mp4", "detection.mp4", "team_classification.mp4"])
        v_path = OUTPUT_DIR / vid_choice
        if v_path.exists():
            st.video(str(v_path))
        else:
            st.warning(f"Video file '{vid_choice}' not found in outputs/ directory.")

    with tabs[11]:
        st.header("Export Telemetry & Reports")
        for f_name in ["analytics.json", "pass_events.json", "pass_summary.json", "shot_events.json", "team_passing_summary.json", "average_positions.json", "validation_report.json", "pass_network_validation.json", "performance_report.json", "pass_network_performance.json"]:
            f_path = OUTPUT_DIR / f_name
            if f_path.exists():
                with open(f_path, "r", encoding="utf-8") as f:
                    st.download_button(
                        label=f"Download {f_name}",
                        data=f.read(),
                        file_name=f_name,
                        mime="application/json"
                    )


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
if not st.session_state.authenticated:
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
        if has_permission("view"):
            if st.button("📊 Results Dashboard"):
                st.session_state.page = "Results Dashboard"
            if st.button("📈 Analytics Dashboard"):
                st.session_state.page = "Analytics Dashboard"
            if st.button("🤖 AI Match Reports"):
                st.session_state.page = "AI Match Reports"
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
    elif page == "Analytics Dashboard":
        page_analytics_dashboard()
    elif page == "AI Match Reports":
        page_ai_reports()
    elif page == "Settings":
        page_settings()
    else:
        page_home()