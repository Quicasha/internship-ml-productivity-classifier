from __future__ import annotations

from pathlib import Path

import pandas as pd

from metrics import compute_metrics, pretty_print


RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

OUT_PATH = RESULTS_DIR / "model_comparison.csv"


def _row(model_name: str, m):
    d = m.to_row(model_name)
    return d


def main():
    # Import here to avoid circular imports if scripts evolve
    from train_dummy import train_dummy
    from train_logistic import train_logistic
    from train_random_forest import train_random_forest

    rows = []

    print("\n=== RandomForest ===")
    rf_metrics, rf_y_true, rf_y_pred = train_random_forest(return_preds=True)
    pretty_print(rf_y_true, rf_y_pred)
    rows.append(_row("RandomForest", rf_metrics))

    print("\n=== LogisticRegression ===")
    lr_metrics, lr_y_true, lr_y_pred = train_logistic(return_preds=True)
    pretty_print(lr_y_true, lr_y_pred)
    rows.append(_row("LogisticRegression", lr_metrics))

    print("\n=== Dummy (most_frequent baseline) ===")
    dummy_metrics = train_dummy()
    rows.append(_row("DummyMostFrequent", dummy_metrics))

    df_out = pd.DataFrame(rows)
    df_out.to_csv(OUT_PATH, index=False)

    print(f"\nSaved results to: {OUT_PATH}")


if __name__ == "__main__":
    main()
