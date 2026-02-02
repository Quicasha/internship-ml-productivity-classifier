from __future__ import annotations

from typing import Tuple, Union

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from preprocess import prepare_dataset
from metrics import compute_metrics


def train_random_forest(
    random_state: int = 42,
    test_size: float = 0.2,
    n_estimators: int = 300,
    max_depth: int | None = None,
    return_preds: bool = False,
) -> Union["object", Tuple["object", np.ndarray, np.ndarray]]:
    """
    Train and evaluate RandomForest on a stratified train/test split.

    Args:
        random_state: Reproducibility seed.
        test_size: Proportion of dataset reserved for testing.
        n_estimators: Number of trees in the forest.
        max_depth: Optional depth limit to control overfitting.
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

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    metrics = compute_metrics(y_test, y_pred)

    if return_preds:
        return metrics, np.asarray(y_test), np.asarray(y_pred)
    return metrics


if __name__ == "__main__":
    # Keep script runnable for quick manual checks
    m = train_random_forest(return_preds=False)
    print(f"Accuracy: {m.accuracy:.6f}")