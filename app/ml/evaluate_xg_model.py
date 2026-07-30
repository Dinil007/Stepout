"""Evaluate a saved xG model against a labelled dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

from app.analytics.xg_features import XGFeatureExtractor


def evaluate(dataset_path: Path, model_path: Path, scaler_path: Path) -> Dict[str, float]:
    df = pd.read_csv(dataset_path)
    feature_columns = [column for column in XGFeatureExtractor.FEATURE_COLUMNS if column in df.columns]
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    x = scaler.transform(df[feature_columns].fillna(0.0))
    y = df["goal"].astype(int)
    probabilities = model.predict_proba(x)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    return {
        "accuracy": round(float(accuracy_score(y, predictions)), 4),
        "precision": round(float(precision_score(y, predictions, zero_division=0)), 4),
        "recall": round(float(recall_score(y, predictions, zero_division=0)), 4),
        "f1": round(float(f1_score(y, predictions, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y, probabilities)), 4) if y.nunique() > 1 else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate StepOut xG model")
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--model-path", type=Path, default=Path("models/xg_model.pkl"))
    parser.add_argument("--scaler-path", type=Path, default=Path("models/scaler.pkl"))
    args = parser.parse_args()
    print(json.dumps(evaluate(args.dataset, args.model_path, args.scaler_path), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
