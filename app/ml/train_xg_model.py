"""Train an expected goals model from a labelled shot dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from app.analytics.xg_features import XGFeatureExtractor


def train(dataset_path: Path, model_type: str, model_dir: Path) -> Dict[str, float]:
    df = pd.read_csv(dataset_path)
    feature_columns = [column for column in XGFeatureExtractor.FEATURE_COLUMNS if column in df.columns]
    if "goal" not in df.columns:
        raise ValueError("Dataset must include a binary 'goal' column")
    if not feature_columns:
        raise ValueError("Dataset does not contain xG feature columns")

    x = df[feature_columns].fillna(0.0)
    y = df["goal"].astype(int)
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y if y.nunique() > 1 else None,
    )

    estimator = _estimator(model_type)
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)
    estimator.fit(x_train_scaled, y_train)
    probabilities = estimator.predict_proba(x_test_scaled)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    metrics = {
        "accuracy": round(float(accuracy_score(y_test, predictions)), 4),
        "precision": round(float(precision_score(y_test, predictions, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, predictions, zero_division=0)), 4),
        "f1": round(float(f1_score(y_test, predictions, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, probabilities)), 4) if y_test.nunique() > 1 else 0.0,
        "cross_validation_accuracy": round(float(cross_val_score(estimator, scaler.transform(x), y, cv=min(5, len(df))).mean()), 4) if len(df) >= 2 else 0.0,
    }

    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(estimator, model_dir / "xg_model.pkl")
    joblib.dump(scaler, model_dir / "scaler.pkl")
    (model_dir / "xg_model_metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return metrics


def _estimator(model_type: str):
    if model_type == "logistic_regression":
        return LogisticRegression(max_iter=1000, class_weight="balanced")
    if model_type == "random_forest":
        return RandomForestClassifier(n_estimators=250, random_state=42, class_weight="balanced")
    if model_type == "xgboost":
        try:
            from xgboost import XGBClassifier
        except ImportError as exc:
            raise RuntimeError("xgboost is optional and is not installed") from exc
        return XGBClassifier(eval_metric="logloss", random_state=42)
    raise ValueError(f"Unsupported model type: {model_type}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train StepOut xG model")
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument(
        "--model-type",
        choices=["logistic_regression", "random_forest", "xgboost"],
        default="logistic_regression",
    )
    parser.add_argument("--model-dir", type=Path, default=Path("models"))
    args = parser.parse_args()
    metrics = train(args.dataset, args.model_type, args.model_dir)
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
