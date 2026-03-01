"""
Data loading and preprocessing utilities.

Design rules:
- This module ONLY loads data and prepares X/y (no splitting).
- No randomness here (no train/test split).
- Model-specific logic must live in train_*.py or run.py.

Expected dataset:
data/occupancy.csv with columns:
Temperature, Humidity, Light, CO2, HumidityRatio, Occupancy
"""

from __future__ import annotations

from pathlib import Path
import pandas as pd

DATA_PATH = Path("data") / "occupancy.csv"

FEATURES = [
    "Temperature",
    "Humidity",
    "Light",
    "CO2",
    "HumidityRatio",
]

TARGET = "Occupancy"


def load_data(path: Path | str = DATA_PATH) -> pd.DataFrame:
    """
    Load raw occupancy dataset from CSV.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path.as_posix()}")

    df = pd.read_csv(path)

    # Optional: some versions contain a time column. We ignore it by default.
    # (Models in this repo use only sensor features listed in FEATURES.)
    return df


def prepare_dataset(path: Path | str = DATA_PATH) -> tuple[pd.DataFrame, pd.Series]:
    """
    Prepare full dataset for modeling.

    Returns:
        X: feature matrix
        y: target vector
    """
    df = load_data(path)

    required = set(FEATURES + [TARGET])
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}. Found: {list(df.columns)}")

    X = df[FEATURES].copy()
    y = df[TARGET].copy()

    return X, y