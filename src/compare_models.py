"""
Unified model comparison on the same holdout split.

Output:
results/model_comparison.csv
"""

from __future__ import annotations

from pathlib import Path
import pandas as pd

from metrics import pretty_print

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)
OUT_PATH = RESULTS_DIR / "model_comparison.csv"


def main() -> None:
    from train_random_forest import train_holdout as rf_train
    from train_logistic import train_holdout as logreg_train
    from train_dummy import train_holdout as dummy_train

    rows = []

    print("\n=== RandomForest ===")
    rf_m, rf_y_true, rf_y_pred = rf_train(return_preds=True)
    pretty_print(rf_y_true, rf_y_pred)
    rows.append(rf_m.to_row("RandomForest"))

    print("\n=== LogisticRegression ===")
    lr_m, lr_y_true, lr_y_pred = logreg_train(return_preds=True)
    pretty_print(lr_y_true, lr_y_pred)
    rows.append(lr_m.to_row("LogisticRegression"))

    print("\n=== DummyMostFrequent ===")
    dm_m, dm_y_true, dm_y_pred = dummy_train(return_preds=True)
    pretty_print(dm_y_true, dm_y_pred)
    rows.append(dm_m.to_row("DummyMostFrequent"))

    pd.DataFrame(rows).to_csv(OUT_PATH, index=False)
    print(f"\nSaved results to: {OUT_PATH.as_posix()}")


if __name__ == "__main__":
    main()