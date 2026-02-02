import pandas as pd
from sklearn.model_selection import train_test_split

DATA_PATH = "data/occupancy.csv"
FEATURES = ["Temperature", "Humidity", "Light", "CO2", "HumidityRatio"]
TARGET = "Occupancy"


def prepare_data():
    df = pd.read_csv(DATA_PATH)

    X = df[FEATURES]
    y = df[TARGET]

    return X, y