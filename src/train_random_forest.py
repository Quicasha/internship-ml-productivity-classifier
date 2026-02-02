import numpy as np
from sklearn.ensemble import RandomForestClassifier

from preprocess import prepare_data
from metrics import pretty_print


def build_model(random_state: int = 42) -> RandomForestClassifier:
    """
    Build a Random Forest classifier with sane defaults.
    Keeping this as a function lets other scripts (CV, realtime sim)
    reuse the exact same model config.
    """
    return RandomForestClassifier(
        n_estimators=300,
        random_state=random_state,
        n_jobs=-1,
        class_weight="balanced",
    )


def train_random_forest() -> None:
    """
    Train + evaluate Random Forest using a standard train/test split.
    Prints metrics to console.
    """
    X_train, X_test, y_train, y_test = prepare_data(return_full=False)

    model = build_model()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("Accuracy:", float(np.mean(y_pred == y_test)))
    pretty_print(y_test, y_pred)


if __name__ == "__main__":
    train_random_forest()