import os
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


DATA_PATH = "data/occupancy.csv"
OUTPUT_PATH = "results/ablation_test.png"
RANDOM_STATE = 42


def train_and_eval(df: pd.DataFrame, features: list[str], target: str = "Occupancy") -> float:
    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=300,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    return accuracy_score(y_test, y_pred)


def main() -> None:
    df = pd.read_csv(DATA_PATH)

    # Drop non-feature column(s)
    if "date" in df.columns:
        df = df.drop(columns=["date"])

    features = ["Temperature", "Humidity", "Light", "CO2", "HumidityRatio"]

    # Ensure output folder exists
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    results = []

    baseline_acc = train_and_eval(df, features)
    results.append(("All", baseline_acc))

    for drop_feature in features:
        reduced = [f for f in features if f != drop_feature]
        acc = train_and_eval(df, reduced)
        results.append((f"-{drop_feature}", acc))

    # Print results to terminal (so it's usable in report text)
    print("Ablation test (Random Forest) - Accuracy\n")
    for label, acc in results:
        print(f"{label:>12}: {acc:.6f}")

    # Plot
    labels = [r[0] for r in results]
    scores = [r[1] for r in results]

    plt.figure()
    plt.bar(labels, scores)
    plt.ylim(0.0, 1.0)
    plt.xticks(rotation=45, ha="right")
    plt.title("Ablation Test Accuracy (Random Forest)")
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH)

    print(f"\nSaved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()