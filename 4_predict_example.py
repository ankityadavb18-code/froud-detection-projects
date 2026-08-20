import joblib
import pandas as pd

saved = joblib.load("models/fraud_model.pkl")
model = saved["model"]
FEATURES = saved["features"]

normal_example = {
    "amount": 45.20,
    "hour_of_day": 13,
    "transactions_last_hour": 1,
    "distance_from_home_km": 3.5,
    "is_foreign_transaction": 0,
    "account_age_days": 800,
}

suspicious_example = {
    "amount": 980.00,
    "hour_of_day": 3,
    "transactions_last_hour": 6,
    "distance_from_home_km": 450.0,
    "is_foreign_transaction": 1,
    "account_age_days": 20,
}

examples = {
    "Normal-looking transaction": normal_example,
    "Suspicious transaction": suspicious_example,
}

for label, txn in examples.items():
    txn_df = pd.DataFrame([txn])[FEATURES]
    prediction = model.predict(txn_df)[0]
    fraud_probability = model.predict_proba(txn_df)[0][1]
    verdict = "🚨 FRAUD" if prediction == 1 else "✅ Normal"
    print(f"{label}: {verdict} ({fraud_probability * 100:.1f}%)")