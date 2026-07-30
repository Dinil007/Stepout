"""Train an Expected Threat grid from labelled event data (future-ready)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from app.analytics.xt_grid import XTGrid


def train_from_events(dataset_path: Path, grid_key: str, model_dir: Path) -> Dict:
    """Learn xT grid from event data using transition probabilities."""
    df = pd.read_csv(dataset_path)
    required = {"start_col", "start_row", "end_col", "end_row", "shot"}
    if not required.issubset(set(df.columns)):
        raise ValueError(f"Dataset must contain columns: {required}")

    grid = XTGrid(grid_key=grid_key)
    rows, cols = grid.rows, grid.cols
    n_cells = rows * cols

    transition_matrix = np.zeros((n_cells, n_cells))
    shot_probability = np.zeros(n_cells)

    for _, row in df.iterrows():
        sc = int(row["start_col"]) * cols + int(row["start_row"])
        ec = int(row["end_col"]) * cols + int(row["end_row"])
        transition_matrix[sc, ec] += 1.0
        if int(row["shot"]) == 1:
            shot_probability[ec] += 1.0

    row_sums = transition_matrix.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1, row_sums)
    transition_matrix = transition_matrix / row_sums

    shot_sums = np.bincount(
        np.arange(n_cells), weights=shot_probability, minlength=n_cells
    )
    total_events = np.bincount(
        np.arange(n_cells), weights=np.ones(n_cells) * df.shape[0], minlength=n_cells
    )
    shot_prob = np.where(total_events > 0, shot_sums / total_events, 0.0)

    xt_values = np.zeros(n_cells)
    for _ in range(100):
        new_xt = shot_prob + (transition_matrix @ xt_values)
        xt_values = new_xt

    learned_matrix = xt_values.reshape(rows, cols).tolist()
    grid.matrix = [[round(float(v), 4) for v in row] for row in learned_matrix]

    model_dir.mkdir(parents=True, exist_ok=True)
    grid.save_to_file(model_dir / "xt_grid.json")

    metrics = {
        "grid": grid_key,
        "rows": rows,
        "cols": cols,
        "cells": n_cells,
        "max_xt": round(float(np.max(xt_values)), 4),
        "mean_xt": round(float(np.mean(xt_values)), 4),
    }
    (model_dir / "xt_model_metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8",
    )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train StepOut xT grid")
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--grid", choices=list(XTGrid.GRID_CONFIGS.keys()), default="12x8")
    parser.add_argument("--model-dir", type=Path, default=Path("models"))
    args = parser.parse_args()
    metrics = train_from_events(args.dataset, args.grid, args.model_dir)
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()