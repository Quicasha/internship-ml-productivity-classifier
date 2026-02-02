import pandas as pd

from train_random_forest import train_random_forest
from train_logistic import train_logistic


def compare_models():
    rows = []

    rf_metrics = train_random_forest()
    rows.append(rf_metrics.to_row("RandomForest"))

    log_metrics = train_logistic()
    rows.append(log_metrics.to_row("LogisticRegression"))

    df = pd.DataFrame(rows)
    df.to_csv("results/model_comparison.csv", index=False)

    print("\nSaved results to results/model_comparison.csv")
    print(df)


if __name__ == "__main__":
    compare_models()