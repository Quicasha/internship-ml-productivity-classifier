import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

DATA_PATH = "data/occupancy.csv"
WINDOW = 5  # minutes

df = pd.read_csv(DATA_PATH)
df = df.drop(columns=["date"])

# Create sliding window features
features = ["Temperature", "Humidity", "Light", "CO2", "HumidityRatio"]
target = "Occupancy"

rows = []
labels = []

for i in range(WINDOW - 1, len(df)):
    window_df = df.iloc[i - WINDOW + 1 : i + 1]

    row = {}
    for f in features:
        row[f"{f}_mean"] = window_df[f].mean()
        row[f"{f}_std"] = window_df[f].std()
        row[f"{f}_min"] = window_df[f].min()
        row[f"{f}_max"] = window_df[f].max()
        row[f"{f}_delta"] = window_df[f].iloc[-1] - window_df[f].iloc[0]

    rows.append(row)
    labels.append(df[target].iloc[i])

X = pd.DataFrame(rows)
y = pd.Series(labels)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Realtime simulation (window=5) Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification report:")
print(classification_report(y_test, y_pred))