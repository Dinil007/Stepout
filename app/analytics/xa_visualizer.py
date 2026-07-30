"""xA visualisation generation."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List

from app.analytics.xa_features import FIELD_LENGTH_METERS, FIELD_WIDTH_METERS, JsonDict

LOGGER = logging.getLogger(__name__)


class XAVisualizer:
    """Creates xA pass maps, charts, and timelines."""

    def render_all(
        self,
        output_dir: Path,
        xa_passes: List[JsonDict],
        team_summary: JsonDict,
        player_summary: JsonDict,
    ) -> Dict[str, str]:
        output_dir.mkdir(parents=True, exist_ok=True)
        paths = {
            "xa_pass_map.png": output_dir / "xa_pass_map.png",
            "team_xa_chart.png": output_dir / "team_xa_chart.png",
            "player_xa_chart.png": output_dir / "player_xa_chart.png",
            "xa_timeline.png": output_dir / "xa_timeline.png",
        }
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            self._pass_map(plt, paths["xa_pass_map.png"], xa_passes)
            self._bar_chart(plt, paths["team_xa_chart.png"], team_summary, "total_xa", "Team xA")
            self._bar_chart(plt, paths["player_xa_chart.png"], player_summary, "total_xa", "Player xA")
            self._timeline(plt, paths["xa_timeline.png"], xa_passes)
        except Exception as exc:
            LOGGER.warning("xA visualisation failed, writing placeholders: %s", exc)
            for path in paths.values():
                path.write_bytes(b"")
        return {name: str(path) for name, path in paths.items()}

    def _pass_map(self, plt, path: Path, xa_passes: List[JsonDict]) -> None:
        fig, ax = plt.subplots(figsize=(9, 5.8))
        ax.set_facecolor("#166534")
        ax.set_xlim(0, FIELD_LENGTH_METERS)
        ax.set_ylim(0, FIELD_WIDTH_METERS)
        ax.set_title("xA Pass Map")
        ax.set_xlabel("Pitch X (m)")
        ax.set_ylabel("Pitch Y (m)")
        ax.plot(
            [0, FIELD_LENGTH_METERS, FIELD_LENGTH_METERS, 0, 0],
            [0, 0, FIELD_WIDTH_METERS, FIELD_WIDTH_METERS, 0],
            color="white",
        )
        colours = {"Blue": "#2563eb", "Red": "#dc2626"}
        for pa in xa_passes:
            team = str(pa.get("team", "Unknown"))
            colour = colours.get(team, "#facc15")
            xa = float(pa.get("xA", 0.0))
            linewidth = max(0.5, xa * 8.0)
            ax.annotate(
                "",
                xy=(pa.get("pass_end_x", 0), pa.get("pass_end_y", 0)),
                xytext=(pa.get("pass_start_x", 0), pa.get("pass_start_y", 0)),
                arrowprops=dict(
                    arrowstyle="->",
                    color=colour,
                    lw=linewidth,
                    alpha=0.7,
                ),
            )
            ax.scatter(
                pa.get("pass_end_x", 0),
                pa.get("pass_end_y", 0),
                s=30 + xa * 300,
                c=colour,
                marker="o",
                edgecolors="white",
                alpha=0.85,
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

    def _timeline(self, plt, path: Path, xa_passes: List[JsonDict]) -> None:
        fig, ax = plt.subplots(figsize=(9, 4.5))
        ordered = sorted(xa_passes, key=lambda p: p.get("frame", 0))
        cumulative: Dict[str, float] = {}
        for pa in ordered:
            team = str(pa.get("team", "Unknown"))
            cumulative[team] = cumulative.get(team, 0.0) + float(pa.get("xA", 0.0))
            ax.scatter(
                pa.get("pass_end_frame") or pa.get("pass_start_frame", 0),
                cumulative[team],
                label=team,
            )
        ax.set_title("xA Timeline")
        ax.set_xlabel("Frame")
        ax.set_ylabel("Cumulative xA")
        self._dedupe_legend(ax)
        fig.tight_layout()
        fig.savefig(path, dpi=140)
        plt.close(fig)

    def _dedupe_legend(self, ax) -> None:
        handles, labels = ax.get_legend_handles_labels()
        deduped = dict(zip(labels, handles))
        if deduped:
            ax.legend(deduped.values(), deduped.keys())