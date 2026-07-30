from __future__ import annotations

import io
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from app.analytics.formation_engine import FormationEngine
from app.analytics.formation_types import PlayerPosition
from app.analytics.formation_validation import FormationValidator
from app.analytics.formation_visualizer import FormationVisualizer, VisualizerConfig

st.set_page_config(page_title="Formation Intelligence", layout="wide")

# Session state initialization
if "formation_engine" not in st.session_state:
    st.session_state.formation_engine = FormationEngine()
if "validator" not in st.session_state:
    st.session_state.validator = FormationValidator()
if "visualizer" not in st.session_state:
    st.session_state.visualizer = FormationVisualizer()
if "analysis_history" not in st.session_state:
    st.session_state.analysis_history = []

# Sidebar
with st.sidebar:
    st.header("Controls")
    match_id = st.text_input("Match ID", value="demo_match")
    frame_number = st.slider("Frame", min_value=0, max_value=1000, value=0)
    confidence_threshold = st.slider("Min Confidence", min_value=0.0, max_value=1.0, value=0.7)
    show_overlay = st.checkbox("Template Overlay", value=False)
    show_hull = st.checkbox("Convex Hull", value=True)
    show_labels = st.checkbox("Labels", value=True)
    show_defensive_line = st.checkbox("Defensive Line", value=True)
    show_midfield_line = st.checkbox("Midfield Line", value=True)
    show_forward_line = st.checkbox("Forward Line", value=True)
    run_analysis = st.button("Run Analysis")

# Header
st.title("Formation Intelligence")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Match", match_id)
with col2:
    st.metric("Frame", frame_number)
with col3:
    st.metric("Timestamp", datetime.now(timezone.utc).strftime("%H:%M:%S"))

# Placeholder player generation for demo
def _demo_players(formation_name: str, team_id: int = 1) -> list[PlayerPosition]:
    from app.analytics.formation_templates import default_registry
    template = default_registry.get_template(formation_name)
    players = []
    for idx, (x, y) in enumerate(template.normalized_positions):
        players.append(PlayerPosition(
            player_id=idx + 1,
            team_id=team_id,
            team_name="Home" if team_id == 1 else "Away",
            jersey_number=idx + 1,
            x=x,
            y=y,
            frame_number=frame_number,
            timestamp=datetime.now(timezone.utc),
            confidence=1.0,
            is_goalkeeper=False,
            is_visible=True,
        ))
    return players

# Analysis
if run_analysis:
    home_players = _demo_players("4-3-3", team_id=1)
    away_players = _demo_players("4-4-2", team_id=2)
    engine = st.session_state.formation_engine
    try:
        home_result = engine.analyze(home_players, frame_number=frame_number)
        away_result = engine.analyze(away_players, frame_number=frame_number)
        st.session_state.current_home = home_result
        st.session_state.current_away = away_result
        st.session_state.analysis_history.append({
            "frame": frame_number,
            "home_formation": home_result.detected_formation,
            "home_confidence": home_result.confidence,
            "away_formation": away_result.detected_formation,
            "away_confidence": away_result.confidence,
        })
    except Exception as e:
        st.error(f"Analysis failed: {e}")

# Display results
if "current_home" in st.session_state and "current_away" in st.session_state:
    home = st.session_state.current_home
    away = st.session_state.current_away

    # Formation Summary
    st.subheader("Formation Summary")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Home Formation", home.detected_formation, f"{home.confidence:.2%}")
    with c2:
        st.metric("Away Formation", away.detected_formation, f"{away.confidence:.2%}")
    with c3:
        st.metric("Home Width", f"{home.metrics.team_width:.2f}")
    with c4:
        st.metric("Away Width", f"{away.metrics.team_width:.2f}")

    # Tactical Pitch
    st.subheader("Tactical Pitch")
    visualizer = st.session_state.visualizer
    visualizer.config.show_labels = show_labels
    try:
        home_frame = visualizer.create_frame(
            _demo_players(home.detected_formation, team_id=1),
            home.metrics,
            home,
        )
        st.image(home_frame, caption="Home Team Tactical View", use_column_width=True)
    except Exception as e:
        st.warning(f"Visualization failed: {e}")

    # Metrics Panel
    st.subheader("Metrics Panel")
    metrics_df = pd.DataFrame({
        "Metric": ["Width", "Length", "Compactness", "Density", "Vertical Stretch", "Horizontal Stretch"],
        "Home": [
            home.metrics.team_width,
            home.metrics.team_length,
            home.metrics.compactness,
            home.metrics.team_width * home.metrics.team_length,
            home.metrics.vertical_stretch,
            home.metrics.horizontal_stretch,
        ],
        "Away": [
            away.metrics.team_width,
            away.metrics.team_length,
            away.metrics.compactness,
            away.metrics.team_width * away.metrics.team_length,
            away.metrics.vertical_stretch,
            away.metrics.horizontal_stretch,
        ],
    })
    fig = px.bar(metrics_df, x="Metric", y=["Home", "Away"], barmode="group", title="Team Metrics Comparison")
    st.plotly_chart(fig, use_container_width=True)

    # Formation Timeline
    st.subheader("Formation Timeline")
    if st.session_state.analysis_history:
        timeline_df = pd.DataFrame(st.session_state.analysis_history)
        fig = px.line(timeline_df, x="frame", y=["home_confidence", "away_confidence"], title="Confidence Over Time")
        st.plotly_chart(fig, use_container_width=True)

    # Team Comparison
    st.subheader("Team Comparison")
    comparison_df = pd.DataFrame({
        "Metric": ["Formation", "Width", "Length", "Compactness", "Convex Hull"],
        "Home": [
            home.detected_formation,
            home.metrics.team_width,
            home.metrics.team_length,
            home.metrics.compactness,
            home.metrics.convex_hull_area,
        ],
        "Away": [
            away.detected_formation,
            away.metrics.team_width,
            away.metrics.team_length,
            away.metrics.compactness,
            away.metrics.convex_hull_area,
        ],
    })
    st.table(comparison_df)

    # Validation Panel
    st.subheader("Validation Panel")
    report = st.session_state.validator.validate_analysis(home)
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Overall Valid", str(report.overall_valid))
        st.metric("Checked Items", report.checked_items)
        st.metric("Passed Items", report.passed_items)
    with col2:
        if report.errors:
            st.error("Errors: " + ", ".join(report.errors))
        if report.warnings:
            st.warning("Warnings: " + ", ".join(report.warnings))

    # Export
    st.subheader("Export")
    export_col1, export_col2, export_col3 = st.columns(3)
    with export_col1:
        if st.button("Download Visualization"):
            try:
                img_bytes = io.BytesIO()
                import cv2
                _, img_encoded = cv2.imencode(".png", home_frame)
                img_bytes.write(img_encoded)
                st.download_button("Download PNG", data=img_bytes.getvalue(), file_name="formation.png", mime="image/png")
            except Exception as e:
                st.error(f"Export failed: {e}")
    with export_col2:
        if st.button("Export Metrics CSV"):
            csv = metrics_df.to_csv(index=False)
            st.download_button("Download CSV", data=csv, file_name="metrics.csv", mime="text/csv")
    with export_col3:
        if st.button("Export Analysis JSON"):
            analysis_json = json.dumps({
                "home": {
                    "formation": home.detected_formation,
                    "confidence": home.confidence,
                    "metrics": home.metrics.__dict__,
                },
                "away": {
                    "formation": away.detected_formation,
                    "confidence": away.confidence,
                    "metrics": away.metrics.__dict__,
                },
            }, default=str)
            st.download_button("Download JSON", data=analysis_json, file_name="analysis.json", mime="application/json")

else:
    st.info("Configure controls and click 'Run Analysis' to begin.")