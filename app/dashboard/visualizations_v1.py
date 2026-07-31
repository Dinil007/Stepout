from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

FIELD_LENGTH = 105.0
FIELD_WIDTH = 68.0

TEAM_COLORS = {"Team A": "#38bdf8", "Team B": "#f97373", "Unknown": "#94a3b8"}


def empty_pitch(title: str = "Pitch") -> go.Figure:
    fig = go.Figure()
    fig.update_layout(title=title, template="plotly_dark", height=430, margin=dict(l=10, r=10, t=42, b=10), plot_bgcolor="#12351f", paper_bgcolor="rgba(0,0,0,0)")
    fig.update_xaxes(range=[0, FIELD_LENGTH], showgrid=False, zeroline=False, visible=False)
    fig.update_yaxes(range=[FIELD_WIDTH, 0], showgrid=False, zeroline=False, visible=False, scaleanchor="x", scaleratio=1)
    shapes = [
        dict(type="rect", x0=0, y0=0, x1=FIELD_LENGTH, y1=FIELD_WIDTH, line=dict(color="#d9f99d", width=2)),
        dict(type="line", x0=FIELD_LENGTH / 2, y0=0, x1=FIELD_LENGTH / 2, y1=FIELD_WIDTH, line=dict(color="#d9f99d", width=1)),
        dict(type="circle", x0=FIELD_LENGTH / 2 - 9.15, y0=FIELD_WIDTH / 2 - 9.15, x1=FIELD_LENGTH / 2 + 9.15, y1=FIELD_WIDTH / 2 + 9.15, line=dict(color="#d9f99d", width=1)),
        dict(type="rect", x0=0, y0=13.84, x1=16.5, y1=54.16, line=dict(color="#d9f99d", width=1)),
        dict(type="rect", x0=88.5, y0=13.84, x1=105, y1=54.16, line=dict(color="#d9f99d", width=1)),
        dict(type="rect", x0=0, y0=24.84, x1=5.5, y1=43.16, line=dict(color="#d9f99d", width=1)),
        dict(type="rect", x0=99.5, y0=24.84, x1=105, y1=43.16, line=dict(color="#d9f99d", width=1)),
    ]
    fig.update_layout(shapes=shapes)
    return fig


def scatter_pitch(points: pd.DataFrame, title: str, color: str | None = "team") -> go.Figure:
    fig = empty_pitch(title)
    if points.empty:
        fig.add_annotation(text="No tracked coordinates available", x=52.5, y=34, showarrow=False, font=dict(color="#cbd5e1"))
        return fig
    if color and color in points.columns:
        for name, group in points.groupby(color):
            fig.add_trace(go.Scattergl(x=group["x"], y=group["y"], mode="markers", name=str(name), marker=dict(size=5, color=TEAM_COLORS.get(str(name), "#e2e8f0"), opacity=0.42)))
    else:
        fig.add_trace(go.Scattergl(x=points["x"], y=points["y"], mode="markers", marker=dict(size=5, color="#facc15", opacity=0.42), name="Ball"))
    return fig


def density_pitch(points: pd.DataFrame, title: str) -> go.Figure:
    fig = empty_pitch(title)
    if points.empty:
        fig.add_annotation(text="No tracked coordinates available", x=52.5, y=34, showarrow=False, font=dict(color="#cbd5e1"))
        return fig
    fig.add_trace(go.Histogram2dContour(x=points["x"], y=points["y"], colorscale="Turbo", contours=dict(coloring="heatmap"), opacity=0.72, showscale=False, ncontours=12))
    return fig


def ball_trajectory(points: pd.DataFrame) -> go.Figure:
    fig = empty_pitch("Complete Ball Trajectory")
    if points.empty:
        fig.add_annotation(text="No ball trajectory artifact available", x=52.5, y=34, showarrow=False, font=dict(color="#cbd5e1"))
        return fig
    fig.add_trace(go.Scatter(x=points["x"], y=points["y"], mode="lines+markers", line=dict(color="#facc15", width=2), marker=dict(size=4), name="Ball"))
    return fig


def possession_pie(team_df: pd.DataFrame) -> go.Figure:
    source = team_df[team_df["possession_pct"] > 0] if "possession_pct" in team_df else pd.DataFrame()
    if source.empty:
        source = pd.DataFrame({"team": ["Unavailable"], "possession_pct": [1]})
    return px.pie(source, names="team", values="possession_pct", hole=0.45, template="plotly_dark", color="team", color_discrete_map=TEAM_COLORS)


def comparison_bar(team_df: pd.DataFrame, metric: str, title: str) -> go.Figure:
    fig = px.bar(team_df, x="team", y=metric, color="team", title=title, template="plotly_dark", color_discrete_map=TEAM_COLORS, text_auto=".2s")
    fig.update_layout(showlegend=False, margin=dict(l=10, r=10, t=45, b=10), paper_bgcolor="rgba(0,0,0,0)")
    return fig


def speed_line(ball: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if not ball.empty and "speed_kmh" in ball.columns:
        fig.add_trace(go.Scatter(x=ball["frame"], y=ball["speed_kmh"], mode="lines", line=dict(color="#facc15", width=2), name="Ball speed"))
    fig.update_layout(title="Ball Speed", template="plotly_dark", height=320, margin=dict(l=10, r=10, t=45, b=10), paper_bgcolor="rgba(0,0,0,0)", xaxis_title="Frame", yaxis_title="km/h")
    return fig
