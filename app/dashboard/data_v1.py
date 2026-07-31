from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

ROOT_OUTPUT_DIR = Path("outputs")


@dataclass(frozen=True)
class MatchArtifacts:
    output_dir: Path
    analytics: dict[str, Any]
    players: pd.DataFrame
    telemetry: dict[str, Any]
    ball: pd.DataFrame

    @property
    def annotated_video(self) -> Path | None:
        for name in ("final_analytics_demo.mp4", "annotated_video.mp4", "tracking.mp4", "team_classification.mp4"):
            candidate = self.output_dir / name
            if candidate.exists() and candidate.stat().st_size > 1000:
                return candidate
        return None

    @property
    def heatmap_image(self) -> Path | None:
        candidate = self.output_dir / "heatmap.png"
        return candidate if candidate.exists() else None


def latest_output_dir(root: Path = ROOT_OUTPUT_DIR) -> Path:
    if not root.exists():
        return root
    candidates = [root]
    candidates.extend(p for p in root.iterdir() if p.is_dir())
    scored = []
    for candidate in candidates:
        artifacts = [candidate / "analytics.json", candidate / "player_statistics.csv"]
        existing = [p for p in artifacts if p.exists()]
        if existing:
            scored.append((max(p.stat().st_mtime for p in existing), candidate))
    if not scored:
        return root
    return sorted(scored, key=lambda item: item[0], reverse=True)[0][1]


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_players(output_dir: Path, analytics: dict[str, Any]) -> pd.DataFrame:
    csv_path = output_dir / "player_statistics.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
    else:
        df = pd.DataFrame(analytics.get("player_statistics", []))

    if df.empty:
        return pd.DataFrame(
            columns=[
                "player_id",
                "team",
                "distance_m",
                "avg_speed_kmh",
                "max_speed_kmh",
                "frames_tracked",
                "possession_frames",
            ]
        )

    rename_map = {
        "track_id": "player_id",
        "team_id": "team",
        "total_distance_meters": "distance_m",
        "total_distance_m": "distance_m",
    }
    df = df.rename(columns=rename_map)
    for column in ("player_id", "team", "distance_m", "avg_speed_kmh", "max_speed_kmh", "frames_tracked", "possession_frames"):
        if column not in df.columns:
            df[column] = 0
    df["team"] = df["team"].apply(format_team)
    return df


def load_ball(output_dir: Path) -> pd.DataFrame:
    candidates = [
        output_dir / "ball_trajectory.csv",
        output_dir / "ball_detection" / "ball_trajectory.csv",
        ROOT_OUTPUT_DIR / "ball_detection" / "ball_trajectory.csv",
    ]
    for path in candidates:
        if path.exists():
            return pd.read_csv(path)
    return pd.DataFrame(columns=["frame", "center_x", "center_y", "speed_kmh"])


def load_match_artifacts(output_dir: Path | None = None) -> MatchArtifacts:
    resolved = output_dir or latest_output_dir()
    analytics = load_json(resolved / "analytics.json")
    return MatchArtifacts(
        output_dir=resolved,
        analytics=analytics,
        players=load_players(resolved, analytics),
        telemetry=load_json(resolved / "tracking_telemetry.json"),
        ball=load_ball(resolved),
    )


def format_team(value: Any) -> str:
    if pd.isna(value):
        return "Unknown"
    text = str(value)
    if text in {"0", "0.0"}:
        return "Team A"
    if text in {"1", "1.0"}:
        return "Team B"
    if text.lower().startswith("team"):
        return text
    return f"Team {text}"


def fps_from_artifacts(artifacts: MatchArtifacts) -> float:
    return float(artifacts.analytics.get("match_info", {}).get("fps") or 25.0)
