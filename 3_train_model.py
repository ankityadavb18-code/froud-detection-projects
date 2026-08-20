import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    average_precision_score,
)
import matplotlib.pyplot as plt
import joblib
import os

data = pd.read_csv("data/transactions.csv")

FEATURES = [
    "amount",
    "hour_of_day",
    "transactions_last_hour",
    "distance_from_home_km",
    "is_foreign_transaction",
    "account_age_days",
]

X = data[FEATURES]
y = data["is_fraud"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=8,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

print(classification_report(y_test, y_pred, target_names=["Normal", "Fraud"]))
print(confusion_matrix(y_test, y_pred))
print(f"PR-AUC: {average_precision_score(y_test, y_proba):.3f}")

os.makedirs("models", exist_ok=True)
joblib.dump({"model": model, "features": FEATURES}, "models/fraud_model.pkl")