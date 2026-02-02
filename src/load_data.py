import pandas as pd

# Load dataset
DATA_PATH = "data/occupancy.csv"

df = pd.read_csv(DATA_PATH)

# Basic inspection
print("Dataset shape:", df.shape)
print("\nColumns:")
print(df.columns)

print("\nFirst 5 rows:")
print(df.head())