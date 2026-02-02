import pandas as pd

DATA_PATH = "data/occupancy.csv"


def load_data() -> pd.DataFrame:
    """
    Load raw occupancy dataset from disk.
    Returns a pandas DataFrame without any preprocessing.
    """
    return pd.read_csv(DATA_PATH)
