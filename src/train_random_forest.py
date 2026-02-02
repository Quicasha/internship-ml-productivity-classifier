import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from preprocess import prepare_data
from metrics import compute_metrics, pretty_print


def train_random_forest(random_state: int = 42):
    X, y = prepare_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=random_state,
        n_jobs=-1
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    pretty_print(y_test, y_pred)
    metrics = compute_metrics(y_test, y_pred)

    return metrics


if __name__ == "__main__":
    m = train_random_forest()
    print("\nAccuracy:", m.accuracy)