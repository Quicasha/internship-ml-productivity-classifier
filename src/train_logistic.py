"""
Logistic Regression training + evaluation (holdout split).

- Can be executed standalone
- Used by run.py and compare_models.py

Outputs:
- prints confusion matrix + classification report
- can return Metrics + predictions for aggregation
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from preprocess import prepare_dataset
from metrics import pretty_print, compute_metrics, Metrics

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)


def build_model(random_state: int = 42) -> Pipeline:
    """
    Build Logistic Regression pipeline with scaling.

    Why scaling:
    - Logistic Regression is sensitive to feature magnitude.
    - StandardScaler stabilizes optimization and makes coefficients comparable.
    """
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    solver="liblinear",
                    max_iter=2000,
                    random_state=random_state,
                ),
            ),
        ]
    )


def train_holdout(
    test_size: float = 0.2,
    random_state: int = 42,
    return_preds: bool = False,
) -> Tuple[Metrics, pd.Series, pd.Series] | Metrics:
    """
    Train on a holdout split and evaluate.

    If return_preds=True, returns:
        (metrics, y_true, y_pred)
    Otherwise returns:
        metrics
    """
    X, y = prepare_dataset()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    model = build_model(random_state=random_state)
    model.fit(X_train, y_train)
    y_pred = pd.Series(model.predict(X_test), index=y_test.index)

    print("\n=== LogisticRegression (holdout) ===")
    pretty_print(y_test, y_pred)

    m = compute_metrics(y_test, y_pred)

    if return_preds:
        return m, y_test, y_pred
    return m


def main() -> None:
    m = train_holdout()
    out_path = RESULTS_DIR / "metrics_logreg_holdout.csv"
    pd.DataFrame([m.to_row("LogisticRegression")]).to_csv(out_path, index=False)
    print(f"\nSaved: {out_path.as_posix()}")


if __name__ == "__main__":
    main()