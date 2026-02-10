"""
Time-series aware cross-validation for occupancy prediction.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Dict, Any

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from preprocess import prepare_dataset
from train_random_forest import build_model as build_rf
from train_logistic import build_model as build_logreg
from train_dummy import build_model as build_dummy


MODEL_BUILDERS = {
    "rf": build_rf,
    "logreg": build_logreg,
    "dummy": build_dummy,
}


def _evaluate_fold(y_true, y_pred) -> Dict[str, float]:
    """
    Compute fold metrics. Uses class=1 as the positive class (occupied=1).
    zero_division=0 prevents warnings when a model predicts no positives.
    """
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_1": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "recall_1": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "f1_1": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "support": int(len(y_true)),
    }


def run_cross_validation(
    models: List[str] | None = None,
    n_splits: int = 5,
    out_folds_path: str = "results/metrics_cv_folds.csv",
    out_summary_path: str = "results/metrics_cv.csv",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run TimeSeriesSplit cross-validation for selected models and save results.

    Returns:
        folds_df: per-fold metrics
        summary_df: mean/std aggregated per model
    """
    if models is None:
        models = ["rf", "logreg", "dummy"]

    # Validate model keys early (fail fast, less pain later).
    unknown = [m for m in models if m not in MODEL_BUILDERS]
    if unknown:
        raise ValueError(f"Unknown model(s): {unknown}. Available: {list(MODEL_BUILDERS.keys())}")

    # Load full dataset (already in chronological order in the CSV).
    X, y = prepare_dataset()

    # TimeSeriesSplit expects order preserved. We do NOT shuffle.
    tscv = TimeSeriesSplit(n_splits=n_splits)

    records: List[Dict[str, Any]] = []

    for model_key in models:
        build_fn = MODEL_BUILDERS[model_key]

        for fold, (train_idx, test_idx) in enumerate(tscv.split(X), start=1):
            # Works for both pandas and numpy containers.
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx] if hasattr(X, "iloc") else (X[train_idx], X[test_idx])
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx] if hasattr(y, "iloc") else (y[train_idx], y[test_idx])

            model = build_fn()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            metrics = _evaluate_fold(y_test, y_pred)

            records.append({
                "model": model_key,
                "fold": fold,
                **metrics,
            })

    folds_df = pd.DataFrame(records)

    summary_df = (
        folds_df
        .groupby("model")[["accuracy", "precision_1", "recall_1", "f1_1"]]
        .agg(["mean", "std"])
        .reset_index()
    )

    # Ensure output dir exists
    os.makedirs(os.path.dirname(out_folds_path), exist_ok=True)

    folds_df.to_csv(out_folds_path, index=False)
    summary_df.to_csv(out_summary_path, index=False)

    print("\nCross-validation summary (TimeSeriesSplit):")
    print(summary_df)

    return folds_df, summary_df


def main() -> None:
    run_cross_validation()


if __name__ == "__main__":
    main()