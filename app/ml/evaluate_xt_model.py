"""Evaluate a saved xT grid against event data."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from app.analytics.xt_grid import XTGrid


def evaluate(dataset_path: Path, grid_path: Path) -> dict:
    df = pd.read_csv(dataset_path)
    grid = XTGrid(grid_key="12x8")
    grid.load_from_file(grid_path)

    correct = 0
    total = len(df)
    for _, row in df.iterrows():
        start_xt = grid.get_xt(int(row["start_col"]), int(row["start_row"]))
        end_xt = grid.get_xt(int(row["end_col"]), int(row["end_row"]))
        predicted_positive = end_xt > start_xt
        actual_positive = bool(int(row.get("goal_scored", 0)))
        if predicted_positive == actual_positive:
            correct += 1

    return {
        "accuracy": round(correct / total, 4) if total else 0.0,
        "total_events": total,
        "grid": grid.grid_key,
        "rows": grid.rows,
        "cols": grid.cols,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate StepOut xT grid")
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--grid-path", type=Path, default=Path("models/xt_grid.json"))
    args = parser.parse_args()
    print(json.dumps(evaluate(args.dataset, args.grid_path), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()