from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from app.dashboard.data_v1 import MatchArtifacts, fps_from_artifacts, format_team


def possession_percentages(artifacts: MatchArtifacts) -> dict[str, float]:
    summary = artifacts.analytics.get("possession", {}) or artifacts.analytics.get("team_possession", {})
    if isinstance(summary, dict):
        direct = {}
        for key, value in summary.items():
            if "pct" in str(key).lower() or "percent" in str(key).lower():
                direct[format_team(str(key).replace("_pct", "").replace("team_", ""))] = float(value or 0)
        if direct:
            return direct

    players = artifacts.players
    if "possession_frames" not in players.columns or players.empty:
        return {}
    grouped = players.groupby("team")["possession_frames"].sum()
    total = float(grouped.sum())
    if total <= 0:
        return {}
    return {team: round(float(frames) / total * 100, 1) for team, frames in grouped.items()}


def team_analytics(artifacts: MatchArtifacts) -> pd.DataFrame:
    players = artifacts.players.copy()
    if players.empty:
        return pd.DataFrame(columns=["team", "possession_pct", "total_distance_m", "avg_speed_kmh", "max_speed_kmh", "players"])

    possession = possession_percentages(artifacts)
    grouped = players.groupby("team", dropna=False).agg(
        total_distance_m=("distance_m", "sum"),
        avg_speed_kmh=("avg_speed_kmh", "mean"),
        max_speed_kmh=("max_speed_kmh", "max"),
        players=("player_id", "count"),
    ).reset_index()
    grouped["possession_pct"] = grouped["team"].map(possession).fillna(0.0)
    return grouped.round({"total_distance_m": 2, "avg_speed_kmh": 2, "max_speed_kmh": 2, "possession_pct": 1})


def player_analytics(artifacts: MatchArtifacts) -> pd.DataFrame:
    fps = fps_from_artifacts(artifacts)
    df = artifacts.players.copy()
    if df.empty:
        return df
    df["movement_time_s"] = df["frames_tracked"].astype(float).div(fps).round(2)
    if "possession_frames" not in df.columns:
        df["possession_frames"] = 0
    df["possession_time_s"] = df["possession_frames"].astype(float).div(fps).round(2)
    total_poss = float(df["possession_frames"].sum())
    df["possession_pct"] = np.where(total_poss > 0, df["possession_frames"].astype(float) / total_poss * 100, 0).round(1)
    return df[["player_id", "team", "distance_m", "avg_speed_kmh", "max_speed_kmh", "possession_time_s", "possession_pct", "movement_time_s"]]


def telemetry_points(artifacts: MatchArtifacts, team: str | None = None) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    team_lookup = {str(row.player_id): row.team for row in artifacts.players.itertuples()}
    for player_id, payload in artifacts.telemetry.items():
        label = team_lookup.get(str(player_id), format_team(payload.get("team_id")))
        if team and label != team:
            continue
        frames = payload.get("frames", [])
        for idx, pos in enumerate(payload.get("positions_m", [])):
            if len(pos) >= 2:
                rows.append({"player_id": player_id, "team": label, "frame": frames[idx] if idx < len(frames) else idx, "x": float(pos[0]), "y": float(pos[1])})
    return pd.DataFrame(rows, columns=["player_id", "team", "frame", "x", "y"])


def ball_points(artifacts: MatchArtifacts) -> pd.DataFrame:
    ball = artifacts.ball.copy()
    if ball.empty:
        return ball
    x_col = "x" if "x" in ball.columns else "center_x"
    y_col = "y" if "y" in ball.columns else "center_y"
    if x_col not in ball.columns or y_col not in ball.columns:
        return pd.DataFrame(columns=["frame", "x", "y", "speed_kmh"])
    out = pd.DataFrame({"frame": ball.get("frame", pd.Series(range(len(ball)))), "x": ball[x_col], "y": ball[y_col]})
    if "speed_kmh" in ball.columns:
        out["speed_kmh"] = ball["speed_kmh"]
    else:
        out["speed_kmh"] = 0.0
    return out.dropna(subset=["x", "y"])


def possession_timeline(artifacts: MatchArtifacts) -> pd.DataFrame:
    events = artifacts.analytics.get("possession_timeline", [])
    if events:
        return pd.DataFrame(events)
    return pd.DataFrame(columns=["start", "end", "team"])
