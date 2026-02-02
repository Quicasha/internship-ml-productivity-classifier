import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from preprocess import prepare_data
from metrics import pretty_print


def build_model(random_state: int = 42) -> Pipeline:
    """
    Build Logistic Regression model with proper feature scaling.
    Scaling is mandatory for linear models to behave correctly.
    """
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "logreg",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=random_state,
                ),
            ),
        ]
    )


def train_logistic() -> None:
    """
    Train + evaluate Logistic Regression using a standard train/test split.
    """
    X_train, X_test, y_train, y_test = prepare_data(return_full=False)

    model = build_model()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("Accuracy:", float(np.mean(y_pred == y_test)))
    pretty_print(y_test, y_pred)


if __name__ == "__main__":
    train_logistic()