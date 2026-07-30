"""
Tactical Analytics Dashboard

Visualizes:
- Heatmaps (player, team, ball)
- Pass network graph
- Team shape
- Possession chart
- Territory map
- Pressing dashboard
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
import streamlit as st

# Add project root to path for imports
ROOT_DIR = Path(__file__).resolve().parents[2]
import sys
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.homography.field_config import FIELD_LENGTH_METERS, FIELD_WIDTH_METERS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path("outputs")


@st.cache_data
def load_json(path: Path) -> Optional[Dict]:
    if not path.exists():
        return None
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load {path}: {e}")
        return None


def render_heatmaps() -> None:
    st.header("Heatmaps")

    team_hm = load_json(OUTPUT_DIR / "team_heatmap.json")
    player_hm = load_json(OUTPUT_DIR / "player_heatmaps.json")
    territory = load_json(OUTPUT_DIR / "territory_control.json")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Team Heatmaps")
        if team_hm:
            for team_id, heatmap in team_hm.items():
                st.write(f"Team {team_id}")
                st._native_component(
                    tag_name="pre",
                    properties={},
                    content=str(pd.DataFrame(heatmap).describe().to_string())
                )
        else:
            st.warning("Team heatmaps not available.")

    with col2:
        st.subheader("Third Occupancy")
        if territory:
            for team_id, data in territory.items():
                st.write(f"Team {team_id}")
                st.metric("Attacking Third %", f"{data.get('attacking_third_pct', 0):.1f}%")
                st.metric("Defensive Third %", f"{data.get('defensive_third_pct', 0):.1f}%")
                st.metric("Penalty Area %", f"{data.get('penalty_area_pct', 0):.1f}%")
        else:
            st.warning("Territory control not available.")


def render_pass_network() -> None:
    st.header("Pass Network")

    pass_net = load_json(OUTPUT_DIR / "pass_network.json")
    if not pass_net:
        st.warning("Pass network data not available.")
        return

    nodes = pass_net.get("nodes", [])
    edges = pass_net.get("edges", [])
    most_connected = pass_net.get("most_connected_players", [])

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Network Summary")
        st.metric("Total Passes", pass_net.get("total_passes", 0))
        st.metric("Unique Players", len(nodes))

    with col2:
        st.subheader("Most Connected Players")
        for entry in most_connected:
            st.write(f"Player {entry['track_id']}: {entry['connections']} connections")

    st.subheader("Pass Frequency Matrix")
    if nodes:
        player_ids = sorted([n["track_id"] for n in nodes])
        matrix = pd.DataFrame(0, index=player_ids, columns=player_ids)
        for edge in edges:
            a, b = edge["from"], edge["to"]
            if a in matrix.index and b in matrix.columns:
                matrix.loc[a, b] = edge["count"]
                matrix.loc[b, a] = edge["count"]
        st.dataframe(matrix)


def render_team_shape() -> None:
    st.header("Team Shape")

    shape = load_json(OUTPUT_DIR / "team_shape.json")
    if not shape:
        st.warning("Team shape data not available.")
        return

    for team_id, metrics in shape.items():
        st.subheader(f"Team {team_id}")
        cols = st.columns(3)
        cols[0].metric("Avg Centroid X", f"{metrics.get('avg_centroid_x', 0):.1f} m")
        cols[1].metric("Avg Width", f"{metrics.get('avg_width_m', 0):.1f} m")
        cols[2].metric("Avg Length", f"{metrics.get('avg_length_m', 0):.1f} m")
        cols[0].metric("Compactness", f"{metrics.get('avg_compactness_m', 0):.1f} m")
        cols[1].metric("Avg Centroid Y", f"{metrics.get('avg_centroid_y', 0):.1f} m")


def render_possession() -> None:
    st.header("Possession")

    poss = load_json(OUTPUT_DIR / "possession_summary.json")
    if not poss:
        st.warning("Possession data not available.")
        return

    pct = poss.get("possession_pct", {})
    st.subheader("Possession %")
    for team, val in pct.items():
        st.metric(f"Team {team}", f"{val:.1f}%")

    st.subheader("Chains")
    st.metric("Total Chains", poss.get("num_possession_chains", 0))
    st.metric("Avg Duration (frames)", poss.get("avg_possession_duration_frames", 0))
    st.metric("Longest Chain (frames)", poss.get("longest_possession_chain_frames", 0))


def render_territory() -> None:
    st.header("Territory Control")

    territory = load_json(OUTPUT_DIR / "territory_control.json")
    if not territory:
        st.warning("Territory control data not available.")
        return

    for team_id, data in territory.items():
        st.subheader(f"Team {team_id}")
        st.metric("Final Third Entries", data.get("final_third_entries", 0))
        st.metric("Penalty Area Entries", data.get("penalty_area", 0))
        st.metric("Attacking Third %", f"{data.get('attacking_third_pct', 0):.1f}%")
        st.metric("Middle Third %", f"{data.get('middle_third_pct', 0):.1f}%")
        st.metric("Defensive Third %", f"{data.get('defensive_third_pct', 0):.1f}%")


def render_pressing() -> None:
    st.header("Pressing Metrics")

    pressing = load_json(OUTPUT_DIR / "pressing_metrics.json")
    if not pressing:
        st.warning("Pressing metrics not available.")
        return

    ppda = pressing.get("ppda", {})
    st.subheader("PPDA (Passes Per Defensive Action)")
    for team, val in ppda.items():
        st.metric(f"Team {team}", f"{val:.2f}")

    st.subheader("Defensive Actions Detail")
    details = pressing.get("details", {})
    for team, d in details.items():
        st.write(f"Team {team}: {d.get('defensive_actions', 0)} actions, {d.get('opponent_passes', 0)} opponent passes")


def main() -> None:
    st.set_page_config(page_title="Tactical Analytics", layout="wide")
    st.title("Team Tactical Analytics")

    page = st.sidebar.selectbox(
        "Select View",
        [
            "Heatmaps",
            "Pass Network",
            "Team Shape",
            "Possession",
            "Territory Control",
            "Pressing",
        ]
    )

    if page == "Heatmaps":
        render_heatmaps()
    elif page == "Pass Network":
        render_pass_network()
    elif page == "Team Shape":
        render_team_shape()
    elif page == "Possession":
        render_possession()
    elif page == "Territory Control":
        render_territory()
    elif page == "Pressing":
        render_pressing()


if __name__ == "__main__":
    main()