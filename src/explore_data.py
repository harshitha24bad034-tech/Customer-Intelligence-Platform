import pandas as pd

data = pd.read_csv("data/raw/Telco-Customer-Churn.csv")

print("Dataset Shape:")
print(data.shape)

print("\nFirst 5 Rows:")
print(data.head())

print("\nDataset Information:")
data.info()

print("\nMissing Values:")
print(data.isnull().sum())


print("Statistical Summary:")
print(data.describe())

print("\nChurn Distribution:")
print(data["Churn"].value_counts())

print("\nUnique Values:")
for column in data.columns:
    print(column, ":", data[column].nunique())

print("\nTotal Charges Data Type:")
print(data["TotalCharges"].dtype)

print("\nBlank Total Charges:")
print((data["TotalCharges"].str.strip() == "").sum())