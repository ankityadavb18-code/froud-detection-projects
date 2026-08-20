import numpy as np
import pandas as pd
import os

np.random.seed(42)

NUM_NORMAL_TRANSACTIONS = 9800
NUM_FRAUD_TRANSACTIONS = 200

print("Generating synthetic transaction data...")

normal_data = pd.DataFrame({
    "amount": np.random.gamma(shape=2.0, scale=40, size=NUM_NORMAL_TRANSACTIONS),
    "hour_of_day": np.random.normal(loc=14, scale=4, size=NUM_NORMAL_TRANSACTIONS).clip(0, 23),
    "transactions_last_hour": np.random.poisson(lam=1, size=NUM_NORMAL_TRANSACTIONS),
    "distance_from_home_km": np.random.exponential(scale=5, size=NUM_NORMAL_TRANSACTIONS),
    "is_foreign_transaction": np.random.choice([0, 1], size=NUM_NORMAL_TRANSACTIONS, p=[0.95, 0.05]),
    "account_age_days": np.random.randint(30, 3000, size=NUM_NORMAL_TRANSACTIONS),
    "is_fraud": 0
})

fraud_data = pd.DataFrame({
    "amount": np.random.gamma(shape=3.0, scale=150, size=NUM_FRAUD_TRANSACTIONS),
    "hour_of_day": np.random.normal(loc=3, scale=3, size=NUM_FRAUD_TRANSACTIONS).clip(0, 23),
    "transactions_last_hour": np.random.poisson(lam=5, size=NUM_FRAUD_TRANSACTIONS),
    "distance_from_home_km": np.random.exponential(scale=300, size=NUM_FRAUD_TRANSACTIONS),
    "is_foreign_transaction": np.random.choice([0, 1], size=NUM_FRAUD_TRANSACTIONS, p=[0.4, 0.6]),
    "account_age_days": np.random.randint(1, 400, size=NUM_FRAUD_TRANSACTIONS),
    "is_fraud": 1
})

data = pd.concat([normal_data, fraud_data], ignore_index=True)
data = data.sample(frac=1, random_state=42).reset_index(drop=True)

data["amount"] = data["amount"].round(2)
data["hour_of_day"] = data["hour_of_day"].round(0).astype(int)
data["distance_from_home_km"] = data["distance_from_home_km"].round(1)

os.makedirs("data", exist_ok=True)
data.to_csv("data/transactions.csv", index=False)

print(f"✅ Done! Saved {len(data)} transactions to 'data/transactions.csv'")