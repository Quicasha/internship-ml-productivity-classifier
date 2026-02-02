from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from preprocess import prepare_data
from metrics import compute_metrics, pretty_print


def train_logistic(random_state: int = 42):
    X, y = prepare_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=random_state
        ))
    ])

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    pretty_print(y_test, y_pred)
    metrics = compute_metrics(y_test, y_pred)

    return metrics


if __name__ == "__main__":
    m = train_logistic()
    print("\nAccuracy:", m.accuracy)