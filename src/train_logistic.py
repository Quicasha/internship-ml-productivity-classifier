"""
Logistic Regression training + evaluation (holdout split).

Used by:
- run.py (via build_model)
- can be executed standalone
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Any

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from preprocess import prepare_dataset
from metrics import pretty_print, compute_metrics

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)


MODEL_NAME = "logreg"


def build_model(random_state: int = 42) -> Pipeline:
    """
    Build a Logistic Regression pipeline with scaling.

    Why StandardScaler:
    - Logistic regression is sensitive to feature magnitudes
    - Scaling makes optimization stable and coefficients comparable

    Solver notes:
    - liblinear: good for small/medium datasets and binary classification
    - max_iter: bump it to avoid non-convergence
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


def train_and_eval(test_size: float = 0.2, random_state: int = 42) -> Dict[str, Any]:
    """
    Train on a holdout split and return metrics as a dict.
    """
    X_train, X_test, y_train, y_test = prepare_dataset(
        test_size=test_size,
        random_state=random_state,
    )

    model = build_model(random_state=random_state)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print("\n=== LogisticRegression (holdout) ===")
    pretty_print(y_test, y_pred)

    row = compute_metrics(y_test, y_pred).to_row(model_name="LogisticRegression")
    return row


def main() -> None:
    row = train_and_eval()
    out_path = RESULTS_DIR / "metrics_logreg_holdout.csv"
    pd.DataFrame([row]).to_csv(out_path, index=False)
    print(f"\nSaved: {out_path.as_posix()}")


if __name__ == "__main__":
    main()
