import pandas as pd
from sklearn.model_selection import train_test_split

from load_data import load_data  # <-- THIS WAS MISSING

DATA_PATH = "data/occupancy.csv"

FEATURES = [
    "Temperature",
    "Humidity",
    "Light",
    "CO2",
    "HumidityRatio",
]

TARGET = "Occupancy"


def prepare_data(return_full: bool = False):
    """
    Prepare features and target variable for modeling.

    If return_full=True:
        Returns full feature matrix X and target vector y
        without performing a train/test split.
        Used for time-series cross-validation.

    If return_full=False:
        Returns X_train, X_test, y_train, y_test
        using a standard random split.
    """

    df = load_data()

    X = df[FEATURES]
    y = df[TARGET]

    if return_full:
        return X, y

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    return X_train, X_test, y_train, y_test