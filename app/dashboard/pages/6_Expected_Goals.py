"""
StepOut AI Football Analytics — Expected Goals (xG) Dashboard Page

Provides:
  - Team xG overview
  - Player xG breakdown
  - Shot map visualisation
  - xG timeline
  - Highest / lowest xG shots
  - Top finishers
  - Shot table with filters
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.analytics.xg_engine import XGEngine

LOGGER = logging.getLogger(__name__)

OUTPUT_DIR = Path("outputs")


@st.cache_resource
def _xg_engine() -> XGEngine:
    return XGEngine(output_dir=OUTPUT_DIR)


@st.cache_data(ttl=60)
def _run_xg() -> Dict[str, Any]:
    return _xg_engine().run()


def _img_path(name: str) -> str | None:
    path = OUTPUT_DIR / name
    return str(path) if path.exists() else None


st.set_page_config(
    page_title="Expected Goals (xG)",
    page_icon="⚽",
    layout="wide",
)

st.title("⚽ Expected Goals (xG)")
st.markdown("Production-grade xG engine — shot-quality evaluation and finishing analysis.")

# ---------------------------------------------------------------------------
# Load xG data
# ---------------------------------------------------------------------------
try:
    payload = _run_xg()
except Exception as exc:
    st.error(f"Could not compute xG: {exc}")
    st.info("Run the main pipeline first (produces shot_events.json in outputs/).")
    st.stop()

summary: Dict = payload.get("summary", {})
team_summary: Dict = payload.get("team_summary", {})
player_summary: Dict = payload.get("player_summary", {})
shots: list = payload.get("shots", [])
validation: Dict = payload.get("validation", {})
performance: Dict = payload.get("performance", {})

# ---------------------------------------------------------------------------
# 1.  Key Metrics Row
# ---------------------------------------------------------------------------
cols = st.columns(6)
cols[0].metric("Total Shots", summary.get("total_shots", 0))
cols[1].metric("Total xG", f"{summary.get('total_xg', 0.0):.3f}")
cols[2].metric("Average xG/Shot", f"{summary.get('average_xg', 0.0):.3f}")
cols[3].metric("Model", summary.get("model", "N/A"))
cols[4].metric("Teams", summary.get("team_count", 0))
cols[5].metric("Players with Shots", summary.get("player_count", 0))

# ---------------------------------------------------------------------------
# 2.  Best & Worst Chance
# ---------------------------------------------------------------------------
best = summary.get("highest_xg_shot")
worst = summary.get("lowest_xg_shot")
if best:
    st.success(
        f"**Best Chance:** Player #{best.get('player')} ({best.get('team')}) "
        f"— {best.get('distance', 0):.1f}m, {best.get('angle', 0):.1f}° goal angle "
        f"— **xG {best.get('xg', 0):.3f}** "
        f"{'⚽ GOAL!' if best.get('goal') else '❌ Missed'}"
    )
if worst:
    st.info(
        f"**Lowest xG Shot:** Player #{worst.get('player')} ({worst.get('team')}) "
        f"— {worst.get('distance', 0):.1f}m — **xG {worst.get('xg', 0):.3f}**"
    )

# ---------------------------------------------------------------------------
# 3.  Team xG
# ---------------------------------------------------------------------------
st.subheader("📊 Team xG Summary")
team_data = []
for team_name, data in team_summary.items():
    team_data.append({
        "Team": team_name,
        "Total xG": data.get("total_xg", 0),
        "Avg xG/Shot": data.get("average_xg", 0),
        "Goals": data.get("goals", 0),
        "Goals − xG": data.get("goals_minus_xg", 0),
        "Shots": data.get("shots", 0),
    })
if team_data:
    st.dataframe(pd.DataFrame(team_data), use_container_width=True)

team_img = _img_path("team_xg_chart.png")
if team_img:
    st.image(team_img, caption="Team xG Comparison", use_container_width=True)

# ---------------------------------------------------------------------------
# 4.  Player xG
# ---------------------------------------------------------------------------
st.subheader("🏃 Player xG Summary")
player_rows = []
for pid, data in player_summary.items():
    player_rows.append({
        "Player": pid,
        "Total xG": data.get("total_xg", 0),
        "Goals": data.get("goals", 0),
        "Goals − xG": data.get("goals_minus_xg", 0),
        "Shots": data.get("shots", 0),
        "Shots on Target": data.get("shots_on_target", 0),
        "Conversion %": data.get("conversion_pct", 0),
        "Avg Distance (m)": data.get("average_shot_distance", 0),
        "Avg Angle (°)": data.get("average_shot_angle", 0),
    })
if player_rows:
    df_players = pd.DataFrame(player_rows)
    col_filter, _ = st.columns([2, 8])
    team_filter = col_filter.selectbox(
        "Filter by Player",
        ["All"] + sorted(df_players["Player"].astype(str).unique().tolist()),
    )
    if team_filter != "All":
        df_players = df_players[df_players["Player"].astype(str) == team_filter]
    st.dataframe(df_players, use_container_width=True)
else:
    st.info("No player xG data available.")

player_img = _img_path("player_xg_chart.png")
if player_img:
    st.image(player_img, caption="Player xG Comparison", use_container_width=True)

# ---------------------------------------------------------------------------
# 5.  Shot Map
# ---------------------------------------------------------------------------
st.subheader("🗺️ Shot Map")
shot_map = _img_path("xg_shot_map.png")
if shot_map:
    st.image(shot_map, caption="Shot locations with xG-proportional sizing", use_container_width=True)
else:
    st.warning("Shot map image not available.")

# ---------------------------------------------------------------------------
# 6.  xG Timeline
# ---------------------------------------------------------------------------
st.subheader("📈 xG Timeline")
timeline = _img_path("xg_timeline.png")
if timeline:
    st.image(timeline, caption="Cumulative xG over match time", use_container_width=True)
else:
    st.warning("xG timeline not available.")

# ---------------------------------------------------------------------------
# 7.  Top Finishers
# ---------------------------------------------------------------------------
st.subheader("🏆 Top Finishers (by Goals − xG)")
finishers = summary.get("top_finishers", [])
if finishers:
    finisher_rows = []
    for pid, data in finishers:
        finisher_rows.append({
            "Player": pid,
            "Goals − xG": data.get("goals_minus_xg", 0),
            "Goals": data.get("goals", 0),
            "Total xG": data.get("total_xg", 0),
            "Shots": data.get("shots", 0),
        })
    st.dataframe(pd.DataFrame(finisher_rows), use_container_width=True)
else:
    st.info("No finisher data.")

# ---------------------------------------------------------------------------
# 8.  Shot Table
# ---------------------------------------------------------------------------
st.subheader("🎯 All Shots")
if shots:
    shot_rows = []
    for shot in shots:
        shot_rows.append({
            "Shot ID": shot.get("shot_id"),
            "Frame": shot.get("frame"),
            "Player": shot.get("player"),
            "Team": str(shot.get("team", "?")),
            "Distance (m)": shot.get("distance_m", 0),
            "Angle (°)": shot.get("angle_deg", 0),
            "xG": shot.get("xg", 0),
            "Goal": "⚽" if shot.get("goal") else "❌",
            "Shot Type": shot.get("shot_type", ""),
        })
    df_shots = pd.DataFrame(shot_rows)

    cf1, cf2, cf3 = st.columns(3)
    team_f = cf1.selectbox(
        "Filter Team", ["All"] + sorted(df_shots["Team"].unique().tolist()),
        key="shot_team",
    )
    player_f = cf2.selectbox(
        "Filter Player", ["All"] + sorted(df_shots["Player"].astype(str).unique().tolist()),
        key="shot_player",
    )
    min_xg = cf3.slider("Min xG", 0.0, 1.0, 0.0, 0.01)

    if team_f != "All":
        df_shots = df_shots[df_shots["Team"] == team_f]
    if player_f != "All":
        df_shots = df_shots[df_shots["Player"].astype(str) == player_f]
    df_shots = df_shots[df_shots["xG"] >= min_xg]

    st.dataframe(df_shots.sort_values("xG", ascending=False), use_container_width=True)
else:
    st.info("No shot data available.")

# ---------------------------------------------------------------------------
# 9.  Validation & Performance
# ---------------------------------------------------------------------------
st.subheader("🔍 Validation Report")
st.json({k: v for k, v in validation.items() if not k.startswith("_")})

st.subheader("⚡ Performance")
st.json(performance)

# ---------------------------------------------------------------------------
# 10.  Reload
# ---------------------------------------------------------------------------
if st.button("🔄 Recompute xG"):
    st.cache_data.clear()
    st.cache_resource.clear()
    st.rerun()