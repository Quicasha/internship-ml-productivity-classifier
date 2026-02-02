from __future__ import annotations

from typing import Tuple, Union

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from preprocess import prepare_dataset
from metrics import compute_metrics


def train_logistic(
    random_state: int = 42,
    test_size: float = 0.2,
    C: float = 1.0,
    max_iter: int = 2000,
    return_preds: bool = False,
) -> Union["object", Tuple["object", np.ndarray, np.ndarray]]:
    """
    Train and evaluate Logistic Regression on a stratified train/test split.

    Args:
        random_state: Reproducibility seed.
        test_size: Proportion of dataset reserved for testing.
        C: Inverse regularization strength (smaller C => stronger regularization).
        max_iter: Max solver iterations to ensure convergence.
        return_preds: If True, return (metrics, y_test, y_pred).

    Returns:
        metrics OR (metrics, y_test, y_pred) if return_preds=True
    """
    ds = prepare_dataset()

    X_train, X_test, y_train, y_test = train_test_split(
        ds.X,
        ds.y,
        test_size=test_size,
        shuffle=True,
        stratify=ds.y,
        random_state=random_state,
    )

    # Pipeline prevents leakage: scaler is fit on train only
    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    C=C,
                    max_iter=max_iter,
                    random_state=random_state,
                    solver="lbfgs",
                ),
            ),
        ]
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    metrics = compute_metrics(y_test, y_pred)

    if return_preds:
        return metrics, np.asarray(y_test), np.asarray(y_pred)
    return metrics


if __name__ == "__main__":
    m = train_logistic(return_preds=False)
    print(f"Accuracy: {m.accuracy:.6f}")
