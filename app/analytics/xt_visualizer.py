"""xT visualisation generation."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List

from app.analytics.xt_grid import XTGrid
from app.analytics.xt_features import FIELD_LENGTH_METERS, FIELD_WIDTH_METERS, JsonDict

LOGGER = logging.getLogger(__name__)


class XTVisualizer:
    """Creates xT heatmaps, charts, timelines, and threat flow diagrams."""

    def __init__(self, grid: XTGrid) -> None:
        self.grid = grid

    def render_all(
        self,
        output_dir: Path,
        xt_actions: List[JsonDict],
        team_summary: JsonDict,
        player_summary: JsonDict,
    ) -> Dict[str, str]:
        output_dir.mkdir(parents=True, exist_ok=True)
        paths = {
            "xt_heatmap.png": output_dir / "xt_heatmap.png",
            "player_xt_chart.png": output_dir / "player_xt_chart.png",
            "team_xt_chart.png": output_dir / "team_xt_chart.png",
            "xt_timeline.png": output_dir / "xt_timeline.png",
            "threat_flow.png": output_dir / "threat_flow.png",
        }
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            self._heatmap(plt, paths["xt_heatmap.png"])
            self._bar_chart(plt, paths["team_xt_chart.png"], team_summary, "total_xt", "Team xT")
            self._bar_chart(plt, paths["player_xt_chart.png"], player_summary, "total_xt", "Player xT")
            self._timeline(plt, paths["xt_timeline.png"], xt_actions)
            self._threat_flow(plt, paths["threat_flow.png"], xt_actions)
        except Exception as exc:
            LOGGER.warning("xT visualisation failed, writing placeholders: %s", exc)
            for path in paths.values():
                path.write_bytes(b"")
        return {name: str(path) for name, path in paths.items()}

    def _heatmap(self, plt, path: Path) -> None:
        fig, ax = plt.subplots(figsize=(9, 5.8))
        ax.set_facecolor("#166534")
        ax.set_xlim(0, FIELD_LENGTH_METERS)
        ax.set_ylim(0, FIELD_WIDTH_METERS)
        ax.set_title("xT Pitch Heatmap")
        ax.set_xlabel("Pitch X (m)")
        ax.set_ylabel("Pitch Y (m)")
        ax.plot(
            [0, FIELD_LENGTH_METERS, FIELD_LENGTH_METERS, 0, 0],
            [0, 0, FIELD_WIDTH_METERS, FIELD_WIDTH_METERS, 0],
            color="white",
        )
        import numpy as np
        data = np.array(self.grid.matrix)
        extent = [0, FIELD_LENGTH_METERS, 0, FIELD_WIDTH_METERS]
        im = ax.imshow(data, extent=extent, origin="lower", cmap="hot", alpha=0.7, aspect="auto")
        plt.colorbar(im, ax=ax, label="xT Value")
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

    def _timeline(self, plt, path: Path, xt_actions: List[JsonDict]) -> None:
        fig, ax = plt.subplots(figsize=(9, 4.5))
        ordered = sorted(xt_actions, key=lambda a: a.get("event_id", 0))
        cumulative: Dict[str, float] = {}
        for action in ordered:
            team = str(action.get("team", "Unknown"))
            cumulative[team] = cumulative.get(team, 0.0) + float(action.get("xt_added", 0.0))
            ax.scatter(action.get("event_id", 0), cumulative[team], label=team, s=10)
        ax.set_title("xT Timeline")
        ax.set_xlabel("Event Index")
        ax.set_ylabel("Cumulative xT")
        handles, labels = ax.get_legend_handles_labels()
        deduped = dict(zip(labels, handles))
        if deduped:
            ax.legend(deduped.values(), deduped.keys())
        fig.tight_layout()
        fig.savefig(path, dpi=140)
        plt.close(fig)

    def _threat_flow(self, plt, path: Path, xt_actions: List[JsonDict]) -> None:
        fig, ax = plt.subplots(figsize=(9, 5.8))
        ax.set_facecolor("#166534")
        ax.set_xlim(0, FIELD_LENGTH_METERS)
        ax.set_ylim(0, FIELD_WIDTH_METERS)
        ax.set_title("Threat Flow — Positive Actions")
        ax.set_xlabel("Pitch X (m)")
        ax.set_ylabel("Pitch Y (m)")
        ax.plot(
            [0, FIELD_LENGTH_METERS, FIELD_LENGTH_METERS, 0, 0],
            [0, 0, FIELD_WIDTH_METERS, FIELD_WIDTH_METERS, 0],
            color="white",
        )
        colours = {"Blue": "#2563eb", "Red": "#dc2626"}
        for action in xt_actions:
            xt = float(action.get("xt_added", 0.0))
            if xt <= 0:
                continue
            team = str(action.get("team", "Unknown"))
            colour = colours.get(team, "#facc15")
            lw = max(0.5, min(xt * 30.0, 6.0))
            ax.annotate(
                "",
                xy=(action.get("end_x", 0), action.get("end_y", 0)),
                xytext=(action.get("start_x", 0), action.get("start_y", 0)),
                arrowprops=dict(arrowstyle="->", color=colour, lw=lw, alpha=0.6),
            )
        fig.tight_layout()
        fig.savefig(path, dpi=140)
        plt.close(fig)