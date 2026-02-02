import pandas as pd
import numpy as np

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from preprocess import prepare_data
from train_random_forest import build_model as build_rf
from train_logistic import build_model as build_logreg


def evaluate_cv(model_name: str, n_splits: int = 5):
    """
    Perform time-series aware cross-validation.
    This prevents data leakage by ensuring training data
    always precedes validation data in time.
    """

    # Load and preprocess full dataset (no random split here)
    X, y = prepare_data(return_full=True)

    tscv = TimeSeriesSplit(n_splits=n_splits)

    records = []

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X), start=1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        if model_name == "rf":
            model = build_rf()
        elif model_name == "logreg":
            model = build_logreg()
        else:
            raise ValueError("Unsupported model")

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        records.append({
            "model": model_name,
            "fold": fold,
            "accuracy": accuracy_score(y_test, y_pred),
            "precision_1": precision_score(y_test, y_pred, pos_label=1),
            "recall_1": recall_score(y_test, y_pred, pos_label=1),
            "f1_1": f1_score(y_test, y_pred, pos_label=1),
            "support": len(y_test),
        })

    return pd.DataFrame(records)


def run():
    all_results = []

    for model in ["rf", "logreg"]:
        df = evaluate_cv(model)
        all_results.append(df)

    results = pd.concat(all_results, ignore_index=True)

    # Aggregate statistics per model
    summary = (
        results
        .groupby("model")[["accuracy", "precision_1", "recall_1", "f1_1"]]
        .agg(["mean", "std"])
        .reset_index()
    )

    results.to_csv("results/metrics_cv_folds.csv", index=False)
    summary.to_csv("results/metrics_cv.csv", index=False)

    print("\nCross-validation summary:")
    print(summary)


if __name__ == "__main__":
    run()