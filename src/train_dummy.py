"""
Dummy baseline training + evaluation (holdout split).

Used by:
- run.py (via build_model)
- can be executed standalone
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Any

import pandas as pd
from sklearn.dummy import DummyClassifier

from preprocess import prepare_dataset
from metrics import pretty_print, compute_metrics

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)


MODEL_NAME = "dummy"


def build_model(random_state: int = 42) -> DummyClassifier:
    """
    Build a simple baseline model.

    strategy="most_frequent" means:
    - predict the majority class for every sample
    - gives a baseline accuracy close to class imbalance ratio
    """
    return DummyClassifier(strategy="most_frequent", random_state=random_state)


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

    print("\n=== DummyClassifier (most_frequent baseline) ===")
    pretty_print(y_test, y_pred)

    row = compute_metrics(y_test, y_pred).to_row(model_name="DummyMostFrequent")
    return row


def main() -> None:
    row = train_and_eval()
    out_path = RESULTS_DIR / "metrics_dummy_holdout.csv"
    pd.DataFrame([row]).to_csv(out_path, index=False)
    print(f"\nSaved: {out_path.as_posix()}")


if __name__ == "__main__":
    main()
