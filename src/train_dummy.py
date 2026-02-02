from __future__ import annotations

from sklearn.dummy import DummyClassifier
from sklearn.model_selection import train_test_split

from preprocess import prepare_dataset
from metrics import pretty_print, compute_metrics


def train_dummy(random_state: int = 42):
    ds = prepare_dataset()

    # Stratify keeps class balance similar in train/test
    X_train, X_test, y_train, y_test = train_test_split(
        ds.X, ds.y, test_size=0.2, shuffle=True, stratify=ds.y, random_state=random_state
    )

    model = DummyClassifier(strategy="most_frequent")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    # Print detailed report (same style as other scripts)
    pretty_print(y_test, y_pred)

    return compute_metrics(y_test, y_pred)


if __name__ == "__main__":
    metrics = train_dummy()
    print(f"\nBaseline (Dummy most_frequent) accuracy: {metrics.accuracy:.6f}")
