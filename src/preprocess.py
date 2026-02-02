"""
Data loading and preprocessing utilities.

This separation is deliberate and required for proper ML evaluation.
"""

import pandas as pd

DATA_PATH = "data/occupancy.csv"

FEATURES = [
    "Temperature",
    "Humidity",
    "Light",
    "CO2",
    "HumidityRatio",
]

TARGET = "Occupancy"


def load_data() -> pd.DataFrame:
    """
    Load raw occupancy dataset from CSV.

    Returns
    -------
    pd.DataFrame
        Raw dataset with all columns.
    """
    df = pd.read_csv(DATA_PATH)

    # Basic sanity check
    expected_cols = FEATURES + [TARGET]
    missing = set(expected_cols) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    return df


def prepare_dataset():
    """
    Prepare full dataset for modeling.

    IMPORTANT:
    - No train/test split here
    - No randomness here
    - No model-specific logic here

    Returns
    -------
    X : pd.DataFrame
        Feature matrix
    y : pd.Series
        Target vector
    """
    df = load_data()

    X = df[FEATURES].copy()
    y = df[TARGET].copy()

    return X, y
