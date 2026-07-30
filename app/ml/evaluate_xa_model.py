"""Evaluate a saved xA model against a labelled dataset."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

from app.analytics.xa_features import XAFeatureExtractor


def evaluate(dataset_path: Path, model_path: Path, scaler_path: Path) -> dict:
    df = pd.read_csv(dataset_path)
    feature_columns = [col for col in XAFeatureExtractor.FEATURE_COLUMNS if col in df.columns]
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    x = scaler.transform(df[feature_columns].fillna(0.0))
    y = df["assist"].astype(int)
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
    parser = argparse.ArgumentParser(description="Evaluate StepOut xA model")
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--model-path", type=Path, default=Path("models/xa_model.pkl"))
    parser.add_argument("--scaler-path", type=Path, default=Path("models/xa_scaler.pkl"))
    args = parser.parse_args()
    print(json.dumps(evaluate(args.dataset, args.model_path, args.scaler_path), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()