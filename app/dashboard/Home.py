from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.dashboard.analytics_v1 import ball_points, player_analytics, possession_timeline, team_analytics, telemetry_points
from app.dashboard.data_v1 import ROOT_OUTPUT_DIR, fps_from_artifacts, latest_output_dir, load_match_artifacts
from app.dashboard.visualizations_v1 import ball_trajectory, comparison_bar, density_pitch, possession_pie, scatter_pitch, speed_line

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)
ROOT_OUTPUT_DIR.mkdir(exist_ok=True)

st.set_page_config(page_title="StepOut Football Analytics", page_icon="SO", layout="wide", initial_sidebar_state="collapsed")

st.markdown(
    """
<style>
:root { color-scheme: dark; }
.stApp { background: #080c12; color: #e5edf7; }
.block-container { padding-top: 1.2rem; max-width: 1440px; }
[data-testid="stHeader"] { background: rgba(8, 12, 18, 0.78); }
.stepout-topbar { display:flex; align-items:center; justify-content:space-between; gap:16px; padding:18px 0 12px; border-bottom:1px solid #1f2937; }
.stepout-brand { display:flex; align-items:center; gap:14px; }
.stepout-logo { width:44px; height:44px; border-radius:8px; display:grid; place-items:center; background:#16a34a; color:#03120a; font-weight:900; letter-spacing:0; }
.stepout-title { font-size:24px; line-height:1; font-weight:800; color:#f8fafc; }
.stepout-subtitle { color:#94a3b8; margin-top:4px; font-size:14px; }
.stepout-pill { border:1px solid #334155; color:#cbd5e1; padding:7px 10px; border-radius:999px; font-size:13px; background:#0f172a; }
.upload-panel, .metric-card, .video-shell { border:1px solid #1f2937; background:#0d141f; border-radius:8px; padding:18px; }
.metric-card { min-height:138px; }
.metric-label { color:#94a3b8; font-size:13px; text-transform:uppercase; letter-spacing:.06em; }
.metric-value { color:#f8fafc; font-size:30px; font-weight:800; margin-top:10px; }
.metric-small { color:#9ca3af; font-size:13px; margin-top:4px; }
.stage-row { display:flex; align-items:center; gap:10px; padding:10px 0; border-bottom:1px solid #172033; color:#cbd5e1; }
.stage-dot { width:20px; height:20px; border-radius:50%; display:grid; place-items:center; background:#14532d; color:#bbf7d0; font-size:13px; font-weight:800; }
.section-title { font-size:18px; color:#f8fafc; font-weight:800; margin:16px 0 8px; }
.stTabs [data-baseweb="tab-list"] { gap: 8px; }
.stTabs [data-baseweb="tab"] { background:#0f172a; border:1px solid #1f2937; border-radius:8px; padding:10px 16px; }
.stTabs [aria-selected="true"] { background:#14532d; border-color:#22c55e; }
button[kind="primary"] { border-radius:8px; }
</style>
""",
    unsafe_allow_html=True,
)


def init_state() -> None:
    defaults = {
        "screen": "home",
        "uploaded_video": None,
        "result_output_dir": str(latest_output_dir()),
        "processing_log": [],
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def topbar() -> None:
    st.markdown(
        """
<div class="stepout-topbar">
  <div class="stepout-brand">
    <div class="stepout-logo">SO</div>
    <div>
      <div class="stepout-title">StepOut Football Analytics</div>
      <div class="stepout-subtitle">Computer vision match intelligence for coaches, scouts, and analysts.</div>
    </div>
  </div>
  <div class="stepout-pill">Version 1 Analytics Platform</div>
</div>
""",
        unsafe_allow_html=True,
    )


def save_upload(uploaded_file) -> Path:
    target = UPLOAD_DIR / uploaded_file.name
    with target.open("wb") as handle:
        handle.write(uploaded_file.getbuffer())
    return target


def run_pipeline(video_path: Path, output_dir: Path, max_frames: int) -> tuple[int, list[str]]:
    cmd = [sys.executable, "run_pipeline.py", "--video", str(video_path), "--output", str(output_dir), "--max_frames", str(max_frames)]
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    logs: list[str] = []
    status = st.empty()
    progress = st.progress(0, text="Analyzing match...")
    stages = [
        ("Detecting Players", ["Detection Completed", "Detection Prep"]),
        ("Tracking Players", ["Tracking Completed", "Tracking + Teams"]),
        ("Classifying Teams", ["Team Classification Completed"]),
        ("Tracking Ball", ["Final Video Frame"]),
        ("Calculating Speed", ["Speed Estimation Completed"]),
        ("Calculating Distance", ["Distance Tracking Completed"]),
        ("Calculating Ball Possession", ["Possession"]),
        ("Generating Analytics", ["Analytics metadata exported", "Pipeline Completed"]),
    ]
    completed: set[str] = set()

    def render() -> None:
        rows = []
        for stage, _ in stages:
            mark = "OK" if stage in completed else ".."
            rows.append(f"<div class='stage-row'><div class='stage-dot'>{mark}</div><div>{stage}</div></div>")
        status.markdown("<div class='upload-panel'><div class='section-title'>Analyzing Match...</div>" + "".join(rows) + "</div>", unsafe_allow_html=True)
        progress.progress(int(len(completed) / len(stages) * 100), text=f"{len(completed)} of {len(stages)} stages complete")

    render()
    if process.stdout:
        for line in process.stdout:
            clean = line.strip()
            if not clean:
                continue
            logs.append(clean)
            for stage, needles in stages:
                if any(needle.lower() in clean.lower() for needle in needles):
                    completed.add(stage)
            render()
    return_code = process.wait()
    if return_code == 0:
        completed.update(stage for stage, _ in stages)
        render()
        progress.progress(100, text="Analysis complete")
    return return_code, logs


def home_screen() -> None:
    topbar()
    st.write("")
    left, right = st.columns([1.15, 0.85], gap="large")
    with left:
        st.markdown("<div class='section-title'>Upload Match Video</div>", unsafe_allow_html=True)
        st.markdown("<div class='upload-panel'>", unsafe_allow_html=True)
        uploaded = st.file_uploader("Supported formats: mp4, mov, avi", type=["mp4", "mov", "avi"], label_visibility="visible")
        max_frames = st.slider("Processing frame limit", min_value=50, max_value=1000, value=500, step=50)
        col_a, col_b = st.columns([1, 1])
        with col_a:
            start = st.button("Upload and Analyze", type="primary", use_container_width=True, disabled=uploaded is None)
        with col_b:
            open_results = st.button("Open Latest Results", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
        if start and uploaded is not None:
            video_path = save_upload(uploaded)
            st.session_state.uploaded_video = str(video_path)
            output_dir = ROOT_OUTPUT_DIR / f"match_{int(time.time())}"
            output_dir.mkdir(parents=True, exist_ok=True)
            st.session_state.result_output_dir = str(output_dir)
            st.session_state.max_frames = max_frames
            st.session_state.screen = "processing"
            st.rerun()
        if open_results:
            st.session_state.result_output_dir = str(latest_output_dir())
            st.session_state.screen = "results"
            st.rerun()
    with right:
        artifacts = load_match_artifacts(Path(st.session_state.result_output_dir))
        team_df = team_analytics(artifacts)
        player_df = player_analytics(artifacts)
        st.markdown("<div class='section-title'>Current Workspace</div>", unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            metric_card("Players", len(player_df), "tracked IDs")
        with c2:
            metric_card("Teams", team_df["team"].nunique() if not team_df.empty else 0, "classified squads")
        metric_card("Output Directory", artifacts.output_dir.name, "latest analytics artifact")


def metric_card(label: str, value, subtext: str = "") -> None:
    st.markdown(
        f"<div class='metric-card'><div class='metric-label'>{label}</div><div class='metric-value'>{value}</div><div class='metric-small'>{subtext}</div></div>",
        unsafe_allow_html=True,
    )


def processing_screen() -> None:
    topbar()
    video_path = Path(st.session_state.uploaded_video) if st.session_state.uploaded_video else None
    if not video_path or not video_path.exists():
        st.warning("No uploaded video found.")
        st.session_state.screen = "home"
        return
    output_dir = Path(st.session_state.result_output_dir)
    return_code, logs = run_pipeline(video_path, output_dir, int(st.session_state.get("max_frames", 500)))
    st.session_state.processing_log = logs
    if return_code == 0:
        st.success("Processing completed. Opening results...")
        time.sleep(1)
        st.session_state.screen = "results"
        st.rerun()
    else:
        st.error("Processing failed. The last pipeline messages are shown below.")
        st.code("\n".join(logs[-30:]))
        if st.button("Back to Upload"):
            st.session_state.screen = "home"
            st.rerun()


def results_screen() -> None:
    topbar()
    artifacts = load_match_artifacts(Path(st.session_state.result_output_dir))
    team_df = team_analytics(artifacts)
    player_df = player_analytics(artifacts)
    ball_df = ball_points(artifacts)
    all_points = telemetry_points(artifacts)

    st.write("")
    tabs = st.tabs(["Annotated Video", "Analytics"])
    with tabs[0]:
        video = artifacts.annotated_video
        st.markdown("<div class='section-title'>Annotated Match Video</div>", unsafe_allow_html=True)
        if video:
            st.video(str(video))
        else:
            st.info("No annotated video artifact is available yet. Run processing to generate it.")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            metric_card("Players", len(player_df), "persistent IDs")
        with c2:
            metric_card("Top Speed", f"{player_df['max_speed_kmh'].max():.1f} km/h" if not player_df.empty else "0 km/h", "player maximum")
        with c3:
            metric_card("Distance", f"{team_df['total_distance_m'].sum():.0f} m" if not team_df.empty else "0 m", "team total")
        with c4:
            metric_card("FPS", fps_from_artifacts(artifacts), "source metadata")

    with tabs[1]:
        st.markdown("<div class='section-title'>Team Analytics</div>", unsafe_allow_html=True)
        if team_df.empty:
            st.info("No team analytics artifact is available.")
        else:
            cols = st.columns(max(1, min(3, len(team_df))))
            for idx, row in enumerate(team_df.itertuples()):
                with cols[idx % len(cols)]:
                    metric_card(row.team, f"{row.possession_pct:.1f}%", "possession")
                    st.metric("Total Distance", f"{row.total_distance_m:.1f} m")
                    st.metric("Average Speed", f"{row.avg_speed_kmh:.1f} km/h")
                    st.metric("Maximum Speed", f"{row.max_speed_kmh:.1f} km/h")

        st.markdown("<div class='section-title'>Player Analytics</div>", unsafe_allow_html=True)
        query = st.text_input("Search player or team", placeholder="Player ID or team name")
        table = player_df.copy()
        if query and not table.empty:
            mask = table.astype(str).apply(lambda col: col.str.contains(query, case=False, na=False)).any(axis=1)
            table = table[mask]
        st.dataframe(table, use_container_width=True, hide_index=True)

        st.markdown("<div class='section-title'>Heatmaps</div>", unsafe_allow_html=True)
        h1, h2 = st.columns(2)
        with h1:
            st.plotly_chart(scatter_pitch(all_points, "Player Heatmap"), use_container_width=True)
            st.plotly_chart(scatter_pitch(telemetry_points(artifacts, "Team A"), "Team A Heatmap"), use_container_width=True)
        with h2:
            st.plotly_chart(scatter_pitch(telemetry_points(artifacts, "Team B"), "Team B Heatmap"), use_container_width=True)
            st.plotly_chart(scatter_pitch(ball_df, "Ball Heatmap", color=None), use_container_width=True)

        st.markdown("<div class='section-title'>Density Maps</div>", unsafe_allow_html=True)
        d1, d2, d3 = st.columns(3)
        with d1:
            st.plotly_chart(density_pitch(telemetry_points(artifacts, "Team A"), "Team A Density"), use_container_width=True)
        with d2:
            st.plotly_chart(density_pitch(telemetry_points(artifacts, "Team B"), "Team B Density"), use_container_width=True)
        with d3:
            st.plotly_chart(density_pitch(ball_df, "Ball Density"), use_container_width=True)

        st.markdown("<div class='section-title'>Ball Analytics</div>", unsafe_allow_html=True)
        b1, b2 = st.columns(2)
        with b1:
            st.plotly_chart(ball_trajectory(ball_df), use_container_width=True)
        with b2:
            st.plotly_chart(speed_line(ball_df), use_container_width=True)
            timeline = possession_timeline(artifacts)
            if timeline.empty:
                st.info("No possession timeline artifact is available yet.")
            else:
                st.dataframe(timeline, use_container_width=True, hide_index=True)

        st.markdown("<div class='section-title'>Charts</div>", unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            st.plotly_chart(possession_pie(team_df), use_container_width=True)
        with c2:
            st.plotly_chart(comparison_bar(team_df, "total_distance_m", "Distance Comparison"), use_container_width=True)
        with c3:
            st.plotly_chart(comparison_bar(team_df, "avg_speed_kmh", "Average Speed Comparison"), use_container_width=True)

    nav1, nav2 = st.columns([1, 1])
    with nav1:
        if st.button("Analyze Another Video", use_container_width=True):
            st.session_state.screen = "home"
            st.rerun()
    with nav2:
        if st.button("Refresh Results", use_container_width=True):
            st.rerun()


init_state()
if st.session_state.screen == "processing":
    processing_screen()
elif st.session_state.screen == "results":
    results_screen()
else:
    home_screen()
