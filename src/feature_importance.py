"""
Random Forest feature importance visualization.

Output:
results/feature_importance.png
"""

from __future__ import annotations

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from preprocess import prepare_dataset, FEATURES
from train_random_forest import build_model

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)
OUT_PATH = RESULTS_DIR / "feature_importance.png"


def main() -> None:
    X, y = prepare_dataset()

    # Holdout split only for fitting a representative model
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = build_model(random_state=42)
    model.fit(X_train, y_train)

    importances = model.feature_importances_
    fi = pd.DataFrame({"feature": FEATURES, "importance": importances}).sort_values("importance", ascending=False)

    print("\nFeature importance (RandomForest):")
    for _, row in fi.iterrows():
        print(f"{row['feature']:>14}: {row['importance']:.6f}")

    plt.figure()
    plt.bar(fi["feature"], fi["importance"])
    plt.xticks(rotation=45, ha="right")
    plt.title("Feature Importance (RandomForest)")
    plt.tight_layout()
    plt.savefig(OUT_PATH)
    print(f"\nSaved: {OUT_PATH.as_posix()}")


if __name__ == "__main__":
    main()