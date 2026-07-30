"""
StepOut AI Football Analytics — Expected Assists (xA) Dashboard Page

Provides:
  - Team xA overview
  - Player xA breakdown
  - Pass map visualisation
  - xA timeline
  - Top chance creators
  - Linked pass-to-shot table with filters
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import streamlit as st

from app.analytics.xa_engine import XAEngine

LOGGER = logging.getLogger(__name__)

OUTPUT_DIR = Path("outputs")


@st.cache_resource
def _xa_engine() -> XAEngine:
    return XAEngine(output_dir=OUTPUT_DIR)


@st.cache_data(ttl=60)
def _run_xa() -> Dict[str, Any]:
    return _xa_engine().run()


def _img_path(name: str) -> str | None:
    path = OUTPUT_DIR / name
    return str(path) if path.exists() else None


st.set_page_config(page_title="Expected Assists (xA)", page_icon="🎯", layout="wide")

st.title("🎯 Expected Assists (xA)")
st.markdown("Production-grade xA engine — pass-to-shot linking and chance-creation evaluation.")

try:
    payload = _run_xa()
except Exception as exc:
    st.error(f"Could not compute xA: {exc}")
    st.info("Run the main pipeline first (produces pass_events.json and shot_events.json in outputs/).")
    st.stop()

summary: Dict = payload.get("summary", {})
team_summary: Dict = payload.get("team_summary", {})
player_summary: Dict = payload.get("player_summary", {})
passes: list = payload.get("passes", [])
validation: Dict = payload.get("validation", {})
performance: Dict = payload.get("performance", {})

# ---------------------------------------------------------------------------
# 1. Key Metrics Row
# ---------------------------------------------------------------------------
cols = st.columns(6)
cols[0].metric("Linked Passes", summary.get("total_passes", 0))
cols[1].metric("Total xA", f"{summary.get('total_xa', 0.0):.3f}")
cols[2].metric("Average xA/Pass", f"{summary.get('average_xa', 0.0):.3f}")
cols[3].metric("Model", summary.get("model", "N/A"))
cols[4].metric("Teams", summary.get("team_count", 0))
cols[5].metric("Creators", summary.get("player_count", 0))

# ---------------------------------------------------------------------------
# 2. Best Chance Creator
# ---------------------------------------------------------------------------
best = summary.get("best_chance_creator")
if best:
    st.success(
        f"**Best Chance Creator:** Player #{best.get('player')} ({best.get('team')}) "
        f"— **xA {best.get('xA', 0):.3f}** "
        f"(linked shot xG: {best.get('linked_shot_xG', 0):.3f}) "
        f"{'⚽ GOAL!' if best.get('goal') else '❌ No goal'}"
    )

# ---------------------------------------------------------------------------
# 3. Team xA
# ---------------------------------------------------------------------------
st.subheader("📊 Team xA Summary")
team_data = []
for team_name, data in team_summary.items():
    team_data.append({
        "Team": team_name,
        "Total xA": data.get("total_xa", 0),
        "Avg xA/Pass": data.get("average_xa", 0),
        "Progressive xA": data.get("progressive_xa", 0),
        "Assists": data.get("assists", 0),
        "Assists − xA": data.get("assists_minus_xa", 0),
        "Passes": data.get("passes", 0),
    })
if team_data:
    st.dataframe(pd.DataFrame(team_data), use_container_width=True)

team_img = _img_path("team_xa_chart.png")
if team_img:
    st.image(team_img, caption="Team xA Comparison", use_container_width=True)

# ---------------------------------------------------------------------------
# 4. Player xA
# ---------------------------------------------------------------------------
st.subheader("🏃 Player xA Summary")
player_rows = []
for pid, data in player_summary.items():
    player_rows.append({
        "Player": pid,
        "Total xA": data.get("total_xa", 0),
        "Assists": data.get("assists", 0),
        "Assists − xA": data.get("assists_minus_xa", 0),
        "Key Passes": data.get("key_passes", 0),
        "Progressive Passes": data.get("progressive_passes", 0),
        "Avg Pass Length (m)": data.get("average_pass_length", 0),
        "Chance Creation %": data.get("chance_creation_rate", 0),
    })
if player_rows:
    df_players = pd.DataFrame(player_rows)
    col_filter, _ = st.columns([2, 8])
    player_filter = col_filter.selectbox(
        "Filter by Player",
        ["All"] + sorted(df_players["Player"].astype(str).unique().tolist()),
    )
    if player_filter != "All":
        df_players = df_players[df_players["Player"].astype(str) == player_filter]
    st.dataframe(df_players, use_container_width=True)
else:
    st.info("No player xA data available.")

player_img = _img_path("player_xa_chart.png")
if player_img:
    st.image(player_img, caption="Player xA Comparison", use_container_width=True)

# ---------------------------------------------------------------------------
# 5. Pass Map
# ---------------------------------------------------------------------------
st.subheader("🗺️ Pass Map")
pass_map = _img_path("xa_pass_map.png")
if pass_map:
    st.image(pass_map, caption="Pass origins → destinations with xA-proportional arrows", use_container_width=True)
else:
    st.warning("Pass map image not available.")

# ---------------------------------------------------------------------------
# 6. xA Timeline
# ---------------------------------------------------------------------------
st.subheader("📈 xA Timeline")
timeline = _img_path("xa_timeline.png")
if timeline:
    st.image(timeline, caption="Cumulative xA over match time", use_container_width=True)
else:
    st.warning("xA timeline not available.")

# ---------------------------------------------------------------------------
# 7. Top Creators
# ---------------------------------------------------------------------------
st.subheader("🏆 Top Chance Creators (by Total xA)")
creators = summary.get("top_creators", [])
if creators:
    creator_rows = []
    for pid, data in creators:
        creator_rows.append({
            "Player": pid,
            "Total xA": data.get("total_xa", 0),
            "Key Passes": data.get("key_passes", 0),
            "Progressive Passes": data.get("progressive_passes", 0),
        })
    st.dataframe(pd.DataFrame(creator_rows), use_container_width=True)
else:
    st.info("No creator data.")

# ---------------------------------------------------------------------------
# 8. Linked Pass-to-Shot Table
# ---------------------------------------------------------------------------
st.subheader("🔗 Linked Pass-to-Shot Table")
if passes:
    pass_rows = []
    for pa in passes:
        pass_rows.append({
            "Pass ID": pa.get("pass_id"),
            "Passer": pa.get("player"),
            "Receiver": pa.get("receiver"),
            "Team": str(pa.get("team", "?")),
            "Pass Length (m)": pa.get("pass_length_m", 0),
            "Forward (m)": pa.get("forward_distance_m", 0),
            "xA": pa.get("xA", 0),
            "Linked Shot xG": pa.get("linked_shot_xG", 0),
            "Goal": "⚽" if pa.get("goal") else "❌",
            "Pass Type": pa.get("pass_type", ""),
        })
    df_passes = pd.DataFrame(pass_rows)

    cf1, cf2 = st.columns(2)
    team_f = cf1.selectbox(
        "Filter Team", ["All"] + sorted(df_passes["Team"].unique().tolist()),
        key="xa_team",
    )
    player_f = cf2.selectbox(
        "Filter Passer", ["All"] + sorted(df_passes["Passer"].astype(str).unique().tolist()),
        key="xa_passer",
    )
    if team_f != "All":
        df_passes = df_passes[df_passes["Team"] == team_f]
    if player_f != "All":
        df_passes = df_passes[df_passes["Passer"].astype(str) == player_f]

    st.dataframe(df_passes.sort_values("xA", ascending=False), use_container_width=True)
else:
    st.info("No linked pass data available.")

# ---------------------------------------------------------------------------
# 9. Validation & Performance
# ---------------------------------------------------------------------------
st.subheader("🔍 Validation Report")
st.json({k: v for k, v in validation.items() if not k.startswith("_")})

st.subheader("⚡ Performance")
st.json(performance)

# ---------------------------------------------------------------------------
# 10. Reload
# ---------------------------------------------------------------------------
if st.button("🔄 Recompute xA"):
    st.cache_data.clear()
    st.cache_resource.clear()
    st.rerun()