from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import pandas as pd


# Project-relative dataset path (works when scripts are run from repo root)
DATA_PATH = Path("data") / "occupancy.csv"

# Columns used as model inputs
FEATURES = ["Temperature", "Humidity", "Light", "CO2", "HumidityRatio"]

# Target column (binary label: 0 = unoccupied, 1 = occupied)
TARGET = "Occupancy"


@dataclass(frozen=True)
class Dataset:
    """Container for ML-ready data."""
    X: pd.DataFrame
    y: pd.Series
    df: pd.DataFrame


def load_raw_data(path: Path = DATA_PATH) -> pd.DataFrame:
    """
    Load the raw CSV file.

    Notes:
    - The dataset has an 'id' column which is just a row identifier.
    - The 'date' column is a timestamp; we keep it in df for potential time-based logic,
      but models typically use numeric sensor features defined in FEATURES.
    """
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at: {path.resolve()}")

    df = pd.read_csv(path)

    # Defensive cleanup: strip whitespace from column names if present
    df.columns = [c.strip() for c in df.columns]

    return df


def prepare_dataset(path: Path = DATA_PATH) -> Dataset:
    """
    Return ML-ready (X, y) plus the full raw dataframe.

    This is the single source of truth for:
    - which features are used
    - which column is the target
    - where the dataset is loaded from
    """
    df = load_raw_data(path)

    missing = [c for c in FEATURES + [TARGET] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in CSV: {missing}")

    X = df[FEATURES].copy()
    y = df[TARGET].copy()

    return Dataset(X=X, y=y, df=df)