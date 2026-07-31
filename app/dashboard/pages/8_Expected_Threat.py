"""
StepOut AI Football Analytics — Expected Threat (xT) Dashboard Page

Provides:
  - Team xT overview
  - Player xT breakdown
  - Threat heatmap
  - xT timeline
  - Threat flow
  - Progressive actions table with filters
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.analytics.xt_engine import XTEngine

LOGGER = logging.getLogger(__name__)

OUTPUT_DIR = Path("outputs")


@st.cache_resource
def _xt_engine() -> XTEngine:
    return XTEngine(output_dir=OUTPUT_DIR)


@st.cache_data(ttl=60)
def _run_xt() -> Dict[str, Any]:
    return _xt_engine().run()


def _img_path(name: str) -> str | None:
    path = OUTPUT_DIR / name
    return str(path) if path.exists() else None


st.set_page_config(page_title="Expected Threat (xT)", page_icon="🔥", layout="wide")

st.title("🔥 Expected Threat (xT)")
st.markdown("Production-grade xT engine — pitch heatmap, threat progression, and action evaluation.")

try:
    payload = _run_xt()
except Exception as exc:
    st.error(f"Could not compute xT: {exc}")
    st.info("Run the main pipeline first (produces pass_events.json and ball_tracks.json in outputs/).")
    st.stop()

summary: Dict = payload.get("summary", {})
team_summary: Dict = payload.get("team_summary", {})
player_summary: Dict = payload.get("player_summary", {})
actions: list = payload.get("actions", [])
validation: Dict = payload.get("validation", {})
performance: Dict = payload.get("performance", {})

# ---------------------------------------------------------------------------
# 1. Key Metrics Row
# ---------------------------------------------------------------------------
cols = st.columns(7)
cols[0].metric("Actions", summary.get("total_actions", 0))
cols[1].metric("Total xT", f"{summary.get('total_xt', 0.0):.3f}")
cols[2].metric("Avg xT", f"{summary.get('average_xt', 0.0):.3f}")
cols[3].metric("Model", summary.get("model", "N/A"))
cols[4].metric("Grid", summary.get("grid", "N/A"))
cols[5].metric("Teams", summary.get("team_count", 0))
cols[6].metric("Players", summary.get("player_count", 0))

# ---------------------------------------------------------------------------
# 2. Highest xT Action
# ---------------------------------------------------------------------------
best = summary.get("highest_xt_action")
if best:
    st.success(
        f"**Highest xT Action:** {best.get('action').title()} by Player #{best.get('player_id')} "
        f"({best.get('team')}) — **xT {best.get('xt_added', 0):.4f}**"
    )

# ---------------------------------------------------------------------------
# 3. Team xT
# ---------------------------------------------------------------------------
st.subheader("📊 Team xT Summary")
team_data = []
for team_name, data in team_summary.items():
    team_data.append({
        "Team": team_name,
        "Total xT": data.get("total_xt", 0),
        "Pass xT": data.get("pass_xt", 0),
        "Carry xT": data.get("carry_xt", 0),
        "Avg xT": data.get("average_xt", 0),
        "Positive Actions": data.get("positive_actions", 0),
        "Negative Actions": data.get("negative_actions", 0),
        "Total Actions": data.get("total_actions", 0),
    })
if team_data:
    st.dataframe(pd.DataFrame(team_data), use_container_width=True)

team_img = _img_path("team_xt_chart.png")
if team_img:
    st.image(team_img, caption="Team xT Comparison", use_container_width=True)

# ---------------------------------------------------------------------------
# 4. Player xT
# ---------------------------------------------------------------------------
st.subheader("🏃 Player xT Summary")
player_rows = []
for pid, data in player_summary.items():
    player_rows.append({
        "Player": pid,
        "Total xT": data.get("total_xt", 0),
        "Pass xT": data.get("pass_xt", 0),
        "Carry xT": data.get("carry_xt", 0),
        "Avg xT/Action": data.get("average_xt_per_action", 0),
        "Progressive Actions": data.get("progressive_actions", 0),
    })
if player_rows:
    df_players = pd.DataFrame(player_rows)
    player_filter = st.selectbox(
        "Filter by Player",
        ["All"] + sorted(df_players["Player"].astype(str).unique().tolist()),
    )
    if player_filter != "All":
        df_players = df_players[df_players["Player"].astype(str) == player_filter]
    st.dataframe(df_players, use_container_width=True)
else:
    st.info("No player xT data available.")

player_img = _img_path("player_xt_chart.png")
if player_img:
    st.image(player_img, caption="Player xT Comparison", use_container_width=True)

# ---------------------------------------------------------------------------
# 5. Heatmap
# ---------------------------------------------------------------------------
st.subheader("🗺️ Threat Heatmap")
heatmap = _img_path("xt_heatmap.png")
if heatmap:
    st.image(heatmap, caption="xT pitch heatmap — brighter = higher threat probability", use_container_width=True)
else:
    st.warning("Heatmap not available.")

# ---------------------------------------------------------------------------
# 6. Timeline
# ---------------------------------------------------------------------------
st.subheader("📈 xT Timeline")
timeline = _img_path("xt_timeline.png")
if timeline:
    st.image(timeline, caption="Cumulative xT over actions", use_container_width=True)
else:
    st.warning("Timeline not available.")

# ---------------------------------------------------------------------------
# 7. Threat Flow
# ---------------------------------------------------------------------------
st.subheader("🌊 Threat Flow")
flow = _img_path("threat_flow.png")
if flow:
    st.image(flow, caption="Positive xT actions only", use_container_width=True)
else:
    st.warning("Threat flow not available.")

# ---------------------------------------------------------------------------
# 8. Top Players
# ---------------------------------------------------------------------------
st.subheader("🏆 Top Threat Creators (by Total xT)")
creators = summary.get("top_players", [])
if creators:
    creator_rows = []
    for pid, data in creators:
        creator_rows.append({
            "Player": pid,
            "Total xT": data.get("total_xt", 0),
            "Pass xT": data.get("pass_xt", 0),
            "Carry xT": data.get("carry_xt", 0),
        })
    st.dataframe(pd.DataFrame(creator_rows), use_container_width=True)
else:
    st.info("No creator data.")

# ---------------------------------------------------------------------------
# 9. Actions Table
# ---------------------------------------------------------------------------
st.subheader("🔗 Actions Table")
if actions:
    action_rows = []
    for a in actions:
        action_rows.append({
            "Event ID": a.get("event_id"),
            "Player": a.get("player_id"),
            "Team": str(a.get("team", "?")),
            "Action": a.get("action"),
            "Start Cell": f"[{a.get('start_cell_col')},{a.get('start_cell_row')}]",
            "End Cell": f"[{a.get('end_cell_col')},{a.get('end_cell_row')}]",
            "xT Start": a.get("xt_start", 0),
            "xT End": a.get("xt_end", 0),
            "xT Added": a.get("xt_added", 0),
            "Progressive": "✅" if a.get("progressive") else "❌",
            "Distance (m)": a.get("distance_m", 0),
        })
    df_actions = pd.DataFrame(action_rows)

    cf1, cf2 = st.columns(2)
    team_f = cf1.selectbox(
        "Filter Team", ["All"] + sorted(df_actions["Team"].unique().tolist()),
        key="xt_team",
    )
    action_f = cf2.selectbox(
        "Filter Action", ["All", "pass", "carry"],
        key="xt_action",
    )
    if team_f != "All":
        df_actions = df_actions[df_actions["Team"] == team_f]
    if action_f != "All":
        df_actions = df_actions[df_actions["Action"] == action_f]

    st.dataframe(df_actions.sort_values("xT Added", ascending=False), use_container_width=True)
else:
    st.info("No actions available.")

# ---------------------------------------------------------------------------
# 10. Validation & Performance
# ---------------------------------------------------------------------------
st.subheader("🔍 Validation Report")
st.json({k: v for k, v in validation.items() if not str(k).startswith("_")})

st.subheader("⚡ Performance")
st.json(performance)

# ---------------------------------------------------------------------------
# 11. Reload
# ---------------------------------------------------------------------------
if st.button("🔄 Recompute xT"):
    st.cache_data.clear()
    st.cache_resource.clear()
    st.rerun()