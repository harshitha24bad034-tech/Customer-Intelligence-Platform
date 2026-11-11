import pandas as pd

data = pd.read_csv("data/raw/Telco-Customer-Churn.csv")

data["TotalCharges"] = pd.to_numeric(data["TotalCharges"], errors="coerce")

print("Missing Values Before Cleaning:")
print(data.isnull().sum())

data["TotalCharges"] = data["TotalCharges"].fillna(0)

print("\nMissing Values After Cleaning:")
print(data.isnull().sum())

print("\nUpdated Data Types:")
print(data.dtypes)

data.to_csv("data/processed/cleaned_customer_data.csv", index=False)

