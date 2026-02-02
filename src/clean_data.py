import pandas as pd
import numpy as np

# --------------------------------------------------
# Paths
# --------------------------------------------------
DATA_PATH = "data/occupancy.csv"
OUT_PATH = "data/occupancy_clean.csv"

# Expected logical structure of the dataset
EXPECTED_COLS = [
    "id",
    "date",
    "Temperature",
    "Humidity",
    "Light",
    "CO2",
    "HumidityRatio",
    "Occupancy",
]


def load_raw(path: str) -> pd.DataFrame:
    """
    Load raw occupancy dataset and fix column misalignment issues.

    IMPORTANT:
    The original dataset sometimes comes with a shifted schema:
    - First column named 'date' actually contains an ID (1, 2, 3, ...)
    - Second column 'Temperature' actually contains timestamp strings

    This function detects that case and realigns columns correctly.
    """
    df = pd.read_csv(path)
    cols = list(df.columns)

    # Case: misaligned columns (ID is incorrectly named 'date')
    if cols[:2] == ["date", "Temperature"]:
        # Rename first two columns to their real meaning
        df = df.rename(columns={"date": "id", "Temperature": "date"})

        # Shift remaining sensor columns to correct names
        df = df.rename(
            columns={
                "Humidity": "Temperature",
                "Light": "Humidity",
                "CO2": "Light",
                "HumidityRatio": "CO2",
                "Occupancy": "HumidityRatio",
            }
        )

        # In this layout, Occupancy is stored in the last column
        # If HumidityRatio contains only {0,1}, it is actually Occupancy
        if "HumidityRatio" in df.columns and df["HumidityRatio"].isin([0, 1]).all():
            df = df.rename(columns={"HumidityRatio": "Occupancy"})
            df["HumidityRatio"] = np.nan

    # Fallback: if CSV has 8 columns but wrong headers, force expected schema
    if df.shape[1] == 8 and "Occupancy" not in df.columns:
        df.columns = EXPECTED_COLS

    return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert timestamp to datetime and extract time-based features.

    These features make the model more realistic for real-world usage:
    - hour: hour of day
    - dayofweek: weekday index (0=Mon, 6=Sun)
    - is_weekend: binary weekend flag
    - hour_sin / hour_cos: cyclical encoding of hour
    """
    df = df.copy()

    # Parse timestamp
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # Drop rows where timestamp could not be parsed
    df = df.dropna(subset=["date"])

    # Basic time features
    df["hour"] = df["date"].dt.hour
    df["dayofweek"] = df["date"].dt.dayofweek
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)

    # Cyclical encoding for hour (important for ML models)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

    return df


def main():
    """
    Main cleaning pipeline:
    1. Load raw CSV
    2. Fix schema issues
    3. Convert numeric columns
    4. Add time-based features
    5. Save cleaned dataset
    """
    df = load_raw(DATA_PATH)

    # Ensure target column exists
    if "Occupancy" not in df.columns:
        raise ValueError(f"Target column 'Occupancy' not found. Columns: {list(df.columns)}")

    # Convert sensor columns to numeric
    numeric_cols = [
        "id",
        "Temperature",
        "Humidity",
        "Light",
        "CO2",
        "HumidityRatio",
        "Occupancy",
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Add time-based features
    df = add_time_features(df)

    # Define final column order
    base_cols = [
        "id",
        "date",
        "Temperature",
        "Humidity",
        "Light",
        "CO2",
        "HumidityRatio",
        "Occupancy",
    ]
    time_cols = ["hour", "dayofweek", "is_weekend", "hour_sin", "hour_cos"]

    final_cols = [c for c in base_cols if c in df.columns] + time_cols
    df = df[final_cols].dropna()

    # Save cleaned dataset
    df.to_c_