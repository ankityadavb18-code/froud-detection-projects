import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

data = pd.read_csv("data/transactions.csv")

print(data.head())
print(data.info())
print(data.describe())

fraud_counts = data["is_fraud"].value_counts()
fraud_percent = data["is_fraud"].value_counts(normalize=True) * 100

os.makedirs("outputs", exist_ok=True)

plt.figure(figsize=(5, 4))
sns.countplot(x="is_fraud", data=data)
plt.title("Number of Normal vs Fraud Transactions")
plt.savefig("outputs/class_balance.png")
plt.close()

plt.figure(figsize=(6, 4))
sns.boxplot(x="is_fraud", y="amount", data=data)
plt.title("Transaction Amount: Normal vs Fraud")
plt.savefig("outputs/amount_by_fraud.png")
plt.close()

plt.figure(figsize=(7, 5))
sns.heatmap(data.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Between Features")
plt.savefig("outputs/correlation_heatmap.png")
plt.close()