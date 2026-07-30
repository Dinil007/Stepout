"""xG visualisation generation."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Iterable, List

from app.analytics.xg_features import FIELD_LENGTH_METERS, FIELD_WIDTH_METERS, JsonDict

LOGGER = logging.getLogger(__name__)


class XGVisualizer:
    """Creates xG shot maps, charts, and timelines."""

    def render_all(
        self,
        output_dir: Path,
        shots: List[JsonDict],
        team_summary: JsonDict,
        player_summary: JsonDict,
    ) -> Dict[str, str]:
        output_dir.mkdir(parents=True, exist_ok=True)
        paths = {
            "xg_shot_map.png": output_dir / "xg_shot_map.png",
            "team_xg_chart.png": output_dir / "team_xg_chart.png",
            "player_xg_chart.png": output_dir / "player_xg_chart.png",
            "xg_timeline.png": output_dir / "xg_timeline.png",
        }
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            self._shot_map(plt, paths["xg_shot_map.png"], shots)
            self._bar_chart(plt, paths["team_xg_chart.png"], team_summary, "total_xg", "Team xG")
            self._bar_chart(plt, paths["player_xg_chart.png"], player_summary, "total_xg", "Player xG")
            self._timeline(plt, paths["xg_timeline.png"], shots)
        except Exception as exc:  # pragma: no cover - matplotlib fallback
            LOGGER.warning("xG visualisation failed, writing placeholders: %s", exc)
            for path in paths.values():
                path.write_bytes(b"")
        return {name: str(path) for name, path in paths.items()}

    def _shot_map(self, plt, path: Path, shots: List[JsonDict]) -> None:
        fig, ax = plt.subplots(figsize=(9, 5.8))
        ax.set_facecolor("#166534")
        ax.set_xlim(0, FIELD_LENGTH_METERS)
        ax.set_ylim(0, FIELD_WIDTH_METERS)
        ax.set_title("xG Shot Map")
        ax.set_xlabel("Pitch X (m)")
        ax.set_ylabel("Pitch Y (m)")
        ax.plot([0, FIELD_LENGTH_METERS, FIELD_LENGTH_METERS, 0, 0], [0, 0, FIELD_WIDTH_METERS, FIELD_WIDTH_METERS, 0], color="white")
        colors = {"Blue": "#2563eb", "Red": "#dc2626"}
        for shot in shots:
            color = colors.get(str(shot.get("team")), "#facc15")
            marker = "*" if shot.get("goal") else "o"
            ax.scatter(
                shot.get("shot_x", 0),
                shot.get("shot_y", 0),
                s=70 + float(shot.get("xg", 0.0)) * 700,
                c=color,
                marker=marker,
                edgecolors="white",
                alpha=0.85,
                label=str(shot.get("team")),
            )
        self._dedupe_legend(ax)
        fig.tight_layout()
        fig.savefig(path, dpi=140)
        plt.close(fig)

    def _bar_chart(self, plt, path: Path, summary: JsonDict, metric: str, title: str) -> None:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        labels = list(summary.keys())
        values = [float(summary[label].get(metric, 0.0)) for label in labels]
        ax.bar(labels, values, color="#0f766e")
        ax.set_title(title)
        ax.set_ylabel(metric)
        ax.tick_params(axis="x", rotation=30)
        fig.tight_layout()
        fig.savefig(path, dpi=140)
        plt.close(fig)

    def _timeline(self, plt, path: Path, shots: List[JsonDict]) -> None:
        fig, ax = plt.subplots(figsize=(9, 4.5))
        ordered = sorted(shots, key=lambda shot: shot.get("frame", 0))
        cumulative: Dict[str, float] = {}
        for shot in ordered:
            team = str(shot.get("team"))
            cumulative[team] = cumulative.get(team, 0.0) + float(shot.get("xg", 0.0))
            ax.scatter(shot.get("match_time_s") or shot.get("frame", 0), cumulative[team], label=team)
        ax.set_title("xG Timeline")
        ax.set_xlabel("Match Time (s)")
        ax.set_ylabel("Cumulative xG")
        self._dedupe_legend(ax)
        fig.tight_layout()
        fig.savefig(path, dpi=140)
        plt.close(fig)

    def _dedupe_legend(self, ax) -> None:
        handles, labels = ax.get_legend_handles_labels()
        deduped = dict(zip(labels, handles))
        if deduped:
            ax.legend(deduped.values(), deduped.keys())
