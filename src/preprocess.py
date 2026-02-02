import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load data
DATA_PATH = "data/occupancy.csv"
df = pd.read_csv(DATA_PATH)

# Drop non-numeric / non-useful column
df = df.drop(columns=["date"])

# Features and target
X = df.drop(columns=["Occupancy"])
y = df["Occupancy"]

# Train / test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Standardization
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Train shape:", X_train_scaled.shape)
print("Test shape:", X_test_scaled.shape)