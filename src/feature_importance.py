import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

DATA_PATH = "data/occupancy.csv"
df = pd.read_csv(DATA_PATH)

df = df.drop(columns=["date"])

X = df.drop(columns=["Occupancy"])
y = df["Occupancy"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)

print("Feature importance:")
print(importances)

plt.figure()
importances.plot(kind="bar")
plt.title("Random Forest Feature Importance")
plt.tight_layout()
plt.savefig("results/feature_importance.png")
print("\nSaved: results/feature_importance.png")