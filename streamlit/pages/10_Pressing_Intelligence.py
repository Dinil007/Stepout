from __future__ import annotations

import io
import json
import random
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from app.analytics.pressing_config import PressingConfig
from app.analytics.pressing_detector import PressingDetector
from app.analytics.pressing_engine import PressingEngine
from app.analytics.pressing_metrics import PressingMetricsEngine
from app.analytics.pressing_types import (
    PPDAWindow,
    PressingDetection,
    PressingMetrics,
    PressingSequence,
    PressingZone,
    PressureEvent,
)
from app.analytics.pressing_validation import PressingValidator
from app.analytics.pressing_visualizer import PressingVisualizer, VisualizerConfig

st.set_page_config(page_title="Pressing Intelligence", layout="wide")

# ------------------------------------------------------------------
# Session state initialisation
# ------------------------------------------------------------------
if "pressing_engine" not in st.session_state:
    st.session_state.pressing_engine = PressingEngine()
if "pressing_detector" not in st.session_state:
    st.session_state.pressing_detector = PressingDetector(config=PressingConfig())
if "pressing_metrics_engine" not in st.session_state:
    st.session_state.pressing_metrics_engine = PressingMetricsEngine(config=PressingConfig())
if "pressing_validator" not in st.session_state:
    st.session_state.pressing_validator = PressingValidator()
if "pressing_visualizer" not in st.session_state:
    st.session_state.pressing_visualizer = PressingVisualizer()
if "pressing_history" not in st.session_state:
    st.session_state.pressing_history: list[dict[str, Any]] = []
if "current_result" not in st.session_state:
    st.session_state.current_result: dict[str, Any] | None = None

# ------------------------------------------------------------------
# Sidebar controls
# ------------------------------------------------------------------
with st.sidebar:
    st.header("Controls")
    match_id = st.text_input("Match ID", value="demo_match")
    team_name = st.selectbox("Team", ["Home", "Away"])
    frame_number = st.slider("Frame", min_value=0, max_value=1000, value=0)
    confidence_threshold = st.slider("Confidence Threshold", min_value=0.0, max_value=1.0, value=0.7, step=0.05)
    show_pressure_lines = st.checkbox("Pressure Lines", value=True)
    show_zones = st.checkbox("Pressing Zones", value=True)
    show_labels = st.checkbox("Labels", value=True)
    show_sequences = st.checkbox("Sequences", value=True)
    theme = st.selectbox("Theme", ["Default", "Dark", "Light"])
    run_analysis = st.button("Run Analysis")

# ------------------------------------------------------------------
# Header
# ------------------------------------------------------------------
st.title("Pressing Intelligence")
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Match", match_id)
with col2:
    st.metric("Team", team_name)
with col3:
    st.metric("Frame", frame_number)
with col4:
    st.metric("Timestamp", datetime.now(timezone.utc).strftime("%H:%M:%S"))

# ------------------------------------------------------------------
# Demo data generation helpers
# ------------------------------------------------------------------

def _demo_attackers(count: int = 4) -> list[tuple[float, float, float, float]]:
    """Generate random attacker positions with velocities."""
    attackers: list[tuple[float, float, float, float]] = []
    for _ in range(count):
        x = random.uniform(0.3, 0.7)
        y = random.uniform(0.1, 0.4)
        vx = random.uniform(0.5, 3.0)
        vy = random.uniform(-1.0, 1.0)
        attackers.append((x, y, vx, vy))
    return attackers


def _demo_defenders(count: int = 4) -> list[tuple[float, float, float, float]]:
    """Generate random defender positions with velocities."""
    defenders: list[tuple[float, float, float, float]] = []
    for _ in range(count):
        x = random.uniform(0.3, 0.7)
        y = random.uniform(0.5, 0.8)
        vx = random.uniform(-2.0, 0.5)
        vy = random.uniform(-1.0, 1.0)
        defenders.append((x, y, vx, vy))
    return defenders


def _demo_attacker_positions(count: int = 4) -> dict[int, tuple[float, float]]:
    """Generate attacker position mapping for visualizer."""
    return {i: (random.uniform(0.3, 0.7), random.uniform(0.1, 0.4)) for i in range(count)}


def _demo_defender_positions(count: int = 4) -> dict[int, tuple[float, float]]:
    """Generate defender position mapping for visualizer."""
    return {i: (random.uniform(0.3, 0.7), random.uniform(0.5, 0.8)) for i in range(count)}


def _demo_player_positions(team_id: int = 1) -> list[tuple[int, float, float, int]]:
    """Generate player positions for visualizer."""
    players: list[tuple[int, float, float, int]] = []
    for i in range(11):
        x = random.uniform(0.1, 0.9)
        y = random.uniform(0.1, 0.9)
        players.append((i + 1, x, y, team_id))
    return players


def _generate_timeline_data(frames: int = 50) -> pd.DataFrame:
    """Generate synthetic pressing timeline data."""
    data: list[dict[str, Any]] = []
    for f in range(frames):
        ppda = round(random.uniform(3.0, 15.0), 2)
        pressures = random.randint(1, 20)
        success_rate = round(random.uniform(0.2, 0.9), 2)
        zone = random.choice(["High Press", "Mid Block", "Low Block"])
        data.append({
            "frame": f,
            "ppda": ppda,
            "pressures": pressures,
            "success_rate": success_rate,
            "style": zone,
        })
    return pd.DataFrame(data)


# ------------------------------------------------------------------
# Analysis execution
# ------------------------------------------------------------------
if run_analysis:
    engine = st.session_state.pressing_engine
    detector = st.session_state.pressing_detector
    metrics_engine = st.session_state.pressing_metrics_engine
    validator = st.session_state.pressing_validator
    visualizer = st.session_state.pressing_visualizer

    # Generate demo data
    attackers = _demo_attackers()
    defenders = _demo_defenders()

    try:
        # Run full analysis
        result = engine.analyze(
            attackers=attackers,
            defenders=defenders,
            frame_number=frame_number,
            timestamp=float(frame_number) / 25.0,
        )

        # Build a comprehensive result dict for the UI
        metrics_dict: dict[str, Any] | None = None
        if result.pressing_metrics:
            m = result.pressing_metrics
            metrics_dict = {
                "total_pressures": m.total_pressures,
                "successful_pressures": m.successful_pressures,
                "pressure_success_rate": m.pressure_success_rate,
                "average_pressure_time": m.average_pressure_time,
                "average_closing_speed": m.average_closing_speed,
                "ppda": m.ppda,
                "high_press_count": m.high_press_count,
                "mid_block_count": m.mid_block_count,
                "low_block_count": m.low_block_count,
            }

        detection_dict: dict[str, Any] | None = None
        if result.pressing_detection:
            d = result.pressing_detection
            detection_dict = {
                "pressing_style": d.pressing_style.value,
                "confidence": d.confidence,
                "frame_number": d.frame_number,
            }

        st.session_state.current_result = {
            "result": result,
            "metrics": metrics_dict,
            "detection": detection_dict,
            "attackers": attackers,
            "defenders": defenders,
            "frame": frame_number,
            "team": team_name,
        }

        # Append to history
        st.session_state.pressing_history.append({
            "frame": frame_number,
            "team": team_name,
            "total_pressures": metrics_dict["total_pressures"] if metrics_dict else 0,
            "success_rate": metrics_dict["pressure_success_rate"] if metrics_dict else 0.0,
            "ppda": metrics_dict["ppda"] if metrics_dict else 0.0,
            "style": detection_dict["pressing_style"] if detection_dict else "unknown",
            "confidence": detection_dict["confidence"] if detection_dict else 0.0,
        })

        st.success("Analysis completed successfully.")
    except Exception as e:
        st.error(f"Analysis failed: {e}")

# ------------------------------------------------------------------
# Display results
# ------------------------------------------------------------------
current = st.session_state.current_result

if current is not None:
    result = current["result"]
    metrics = current["metrics"]
    detection = current["detection"]

    # ------------------------------------------------------------------
    # KPI Cards
    # ------------------------------------------------------------------
    st.subheader("Key Performance Indicators")
    kpi_cols = st.columns(7)
    with kpi_cols[0]:
        ppda_val = metrics["ppda"] if metrics else 0.0
        st.metric("PPDA", f"{ppda_val:.2f}")
    with kpi_cols[1]:
        total_p = metrics["total_pressures"] if metrics else 0
        st.metric("Total Pressures", total_p)
    with kpi_cols[2]:
        succ_p = metrics["successful_pressures"] if metrics else 0
        st.metric("Successful", succ_p)
    with kpi_cols[3]:
        sr = metrics["pressure_success_rate"] if metrics else 0.0
        st.metric("Success Rate", f"{sr:.1%}")
    with kpi_cols[4]:
        spd = metrics["average_closing_speed"] if metrics else 0.0
        st.metric("Avg Closing Speed", f"{spd:.2f} m/s")
    with kpi_cols[5]:
        style_label = detection["pressing_style"].replace("_", " ").title() if detection else "N/A"
        st.metric("Pressing Style", style_label)
    with kpi_cols[6]:
        conf = detection["confidence"] if detection else 0.0
        st.metric("Confidence", f"{conf:.1%}")

    # ------------------------------------------------------------------
    # Tactical Pitch
    # ------------------------------------------------------------------
    st.subheader("Tactical Pitch")
    visualizer = st.session_state.pressing_visualizer
    visualizer.config.show_pressing_lines = show_pressure_lines
    visualizer.config.show_pressing_zones = show_zones
    visualizer.config.show_labels = show_labels
    visualizer.config.show_sequences = show_sequences

    try:
        # Build data for visualizer
        player_positions = _demo_player_positions(team_id=1 if team_name == "Home" else 2)
        ball_pos = (random.uniform(0.4, 0.6), random.uniform(0.3, 0.7))
        att_pos = _demo_attacker_positions()
        def_pos = _demo_defender_positions()

        # Build metrics object for annotate_metrics
        metrics_obj = None
        if metrics:
            metrics_obj = PressingMetrics(
                total_pressures=metrics["total_pressures"],
                successful_pressures=metrics["successful_pressures"],
                pressure_success_rate=metrics["pressure_success_rate"],
                average_pressure_time=metrics["average_pressure_time"],
                average_closing_speed=metrics["average_closing_speed"],
                ppda=metrics["ppda"],
                high_press_count=metrics["high_press_count"],
                mid_block_count=metrics["mid_block_count"],
                low_block_count=metrics["low_block_count"],
            )

        detection_obj = None
        if detection:
            try:
                zone = PressingZone(detection["pressing_style"])
            except ValueError:
                zone = PressingZone.LOW_BLOCK
            detection_obj = PressingDetection(
                pressing_style=zone,
                confidence=detection["confidence"],
                frame_number=detection["frame_number"],
                timestamp=datetime.now(timezone.utc),
            )

        frame_img = visualizer.render_frame(
            player_positions=player_positions,
            ball_position=ball_pos,
            pressure_events=result.pressure_events if result.pressure_events else None,
            sequences=result.pressing_sequences if result.pressing_sequences else None,
            metrics=metrics_obj,
            detection=detection_obj,
            attacker_positions=att_pos,
            defender_positions=def_pos,
        )
        st.image(frame_img, caption=f"{team_name} Pressing View", use_column_width=True)
    except Exception as e:
        st.warning(f"Visualisation failed: {e}")

    # ------------------------------------------------------------------
    # Pressing Timeline
    # ------------------------------------------------------------------
    st.subheader("Pressing Timeline")
    timeline_data = _generate_timeline_data(frames=50)

    tab1, tab2, tab3, tab4 = st.tabs(["Style", "PPDA", "Pressure Count", "Success Rate"])

    with tab1:
        style_counts = timeline_data["style"].value_counts().reset_index()
        style_counts.columns = ["Style", "Count"]
        fig_style = px.pie(style_counts, values="Count", names="Style", title="Pressing Style Distribution")
        st.plotly_chart(fig_style, use_container_width=True)

    with tab2:
        fig_ppda = px.line(
            timeline_data, x="frame", y="ppda",
            title="PPDA Trend",
            labels={"frame": "Frame", "ppda": "PPDA"},
        )
        fig_ppda.add_hline(y=10, line_dash="dash", line_color="red", annotation_text="Threshold")
        st.plotly_chart(fig_ppda, use_container_width=True)

    with tab3:
        fig_pressures = px.bar(
            timeline_data, x="frame", y="pressures",
            title="Pressure Count per Frame",
            labels={"frame": "Frame", "pressures": "Pressures"},
        )
        st.plotly_chart(fig_pressures, use_container_width=True)

    with tab4:
        fig_sr = px.line(
            timeline_data, x="frame", y="success_rate",
            title="Pressure Success Rate Trend",
            labels={"frame": "Frame", "success_rate": "Success Rate"},
            range_y=[0, 1],
        )
        st.plotly_chart(fig_sr, use_container_width=True)

    # ------------------------------------------------------------------
    # Zone Analysis
    # ------------------------------------------------------------------
    st.subheader("Zone Analysis")
    if metrics:
        zone_data = pd.DataFrame({
            "Zone": ["High Press", "Mid Block", "Low Block"],
            "Count": [
                metrics["high_press_count"],
                metrics["mid_block_count"],
                metrics["low_block_count"],
            ],
        })
        zone_data["Percentage"] = (
            zone_data["Count"] / zone_data["Count"].sum() * 100
            if zone_data["Count"].sum() > 0
            else 0.0
        )

        zcol1, zcol2 = st.columns(2)
        with zcol1:
            fig_zone_bar = px.bar(
                zone_data, x="Zone", y="Count",
                color="Zone",
                title="Pressure Events by Zone",
                color_discrete_map={
                    "High Press": "#FF4444",
                    "Mid Block": "#FFA500",
                    "Low Block": "#FFDD44",
                },
            )
            st.plotly_chart(fig_zone_bar, use_container_width=True)
        with zcol2:
            fig_zone_pie = px.pie(
                zone_data, values="Count", names="Zone",
                title="Zone Distribution",
                color_discrete_map={
                    "High Press": "#FF4444",
                    "Mid Block": "#FFA500",
                    "Low Block": "#FFDD44",
                },
            )
            st.plotly_chart(fig_zone_pie, use_container_width=True)

        # Zone percentages table
        st.dataframe(
            zone_data.style.format({"Percentage": "{:.1f}%"}),
            use_container_width=True,
        )
    else:
        st.info("No zone data available. Run analysis to generate zone metrics.")

    # ------------------------------------------------------------------
    # Team Comparison (Home vs Away)
    # ------------------------------------------------------------------
    st.subheader("Team Comparison")
    # Generate synthetic away metrics for comparison
    away_metrics = {
        "ppda": round(random.uniform(3.0, 15.0), 2),
        "success_rate": round(random.uniform(0.2, 0.9), 2),
        "closing_speed": round(random.uniform(0.5, 3.0), 2),
        "total_pressures": random.randint(1, 20),
        "high_press": random.randint(0, 10),
        "mid_block": random.randint(0, 10),
        "low_block": random.randint(0, 10),
    }

    home_ppda = metrics["ppda"] if metrics else 0.0
    home_sr = metrics["pressure_success_rate"] if metrics else 0.0
    home_spd = metrics["average_closing_speed"] if metrics else 0.0
    home_total = metrics["total_pressures"] if metrics else 0

    comparison_data = pd.DataFrame({
        "Metric": ["PPDA", "Pressure Success Rate", "Avg Closing Speed", "Total Pressures"],
        "Home": [home_ppda, home_sr, home_spd, home_total],
        "Away": [
            away_metrics["ppda"],
            away_metrics["success_rate"],
            away_metrics["closing_speed"],
            away_metrics["total_pressures"],
        ],
    })

    fig_comp = px.bar(
        comparison_data,
        x="Metric",
        y=["Home", "Away"],
        barmode="group",
        title="Home vs Away Pressing Comparison",
    )
    st.plotly_chart(fig_comp, use_container_width=True)

    # Zone distribution comparison
    if metrics:
        zone_comp = pd.DataFrame({
            "Zone": ["High Press", "Mid Block", "Low Block"],
            "Home": [
                metrics["high_press_count"],
                metrics["mid_block_count"],
                metrics["low_block_count"],
            ],
            "Away": [
                away_metrics["high_press"],
                away_metrics["mid_block"],
                away_metrics["low_block"],
            ],
        })
        fig_zone_comp = px.bar(
            zone_comp,
            x="Zone",
            y=["Home", "Away"],
            barmode="group",
            title="Zone Distribution Comparison",
        )
        st.plotly_chart(fig_zone_comp, use_container_width=True)

    # ------------------------------------------------------------------
    # Validation Panel
    # ------------------------------------------------------------------
    st.subheader("Validation Panel")
    validator = st.session_state.pressing_validator
    try:
        report = validator.validate_analysis(result)
        vcol1, vcol2 = st.columns(2)
        with vcol1:
            st.metric("Overall Valid", str(report.overall_valid))
            st.metric("Checked Items", report.checked_items)
            st.metric("Passed Items", report.passed_items)
        with vcol2:
            if report.errors:
                st.error("Errors: " + ", ".join(report.errors))
            else:
                st.success("No errors.")
            if report.warnings:
                st.warning("Warnings: " + ", ".join(report.warnings))
            else:
                st.info("No warnings.")
    except Exception as e:
        st.warning(f"Validation failed: {e}")

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------
    st.subheader("Export")
    export_col1, export_col2, export_col3 = st.columns(3)

    with export_col1:
        if st.button("Download Visualization (PNG)"):
            try:
                import cv2
                img_bytes = io.BytesIO()
                _, img_encoded = cv2.imencode(".png", frame_img)
                img_bytes.write(img_encoded)
                st.download_button(
                    "Download PNG",
                    data=img_bytes.getvalue(),
                    file_name=f"pressing_{frame_number}.png",
                    mime="image/png",
                )
            except Exception as e:
                st.error(f"PNG export failed: {e}")

    with export_col2:
        if st.button("Export Metrics (CSV)"):
            try:
                csv_data: dict[str, Any] = {}
                if metrics:
                    csv_data = {
                        "Metric": list(metrics.keys()),
                        "Value": list(metrics.values()),
                    }
                csv_df = pd.DataFrame(csv_data)
                csv = csv_df.to_csv(index=False)
                st.download_button(
                    "Download CSV",
                    data=csv,
                    file_name=f"pressing_metrics_{frame_number}.csv",
                    mime="text/csv",
                )
            except Exception as e:
                st.error(f"CSV export failed: {e}")

    with export_col3:
        if st.button("Export Analysis (JSON)"):
            try:
                export_data: dict[str, Any] = {
                    "match_id": match_id,
                    "team": team_name,
                    "frame": frame_number,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "metrics": metrics,
                    "detection": detection,
                    "pressure_events": [
                        {
                            "attacker_id": e.attacker_id,
                            "defender_id": e.defender_id,
                            "team_id": e.team_id,
                            "frame_number": e.frame_number,
                            "distance": e.distance,
                            "closing_speed": e.closing_speed,
                            "pressure_angle": e.pressure_angle,
                            "successful": e.successful,
                        }
                        for e in result.pressure_events
                    ],
                    "pressing_sequences": [
                        {
                            "sequence_id": s.sequence_id,
                            "team_id": s.team_id,
                            "start_frame": s.start_frame,
                            "end_frame": s.end_frame,
                            "duration_seconds": s.duration_seconds,
                            "event_count": s.event_count(),
                        }
                        for s in result.pressing_sequences
                    ],
                }
                analysis_json = json.dumps(export_data, default=str, indent=2)
                st.download_button(
                    "Download JSON",
                    data=analysis_json,
                    file_name=f"pressing_analysis_{frame_number}.json",
                    mime="application/json",
                )
            except Exception as e:
                st.error(f"JSON export failed: {e}")

else:
    st.info("Configure controls in the sidebar and click 'Run Analysis' to begin.")