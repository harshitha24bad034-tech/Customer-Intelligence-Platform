import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("data/processed/cleaned_customer_data.csv")

churn_counts = data["Churn"].value_counts()

print("Customer Churn Count:")
print(churn_counts)
'''
print("\nCustomer Churn Percentage:")
print(data["Churn"].value_counts(normalize=True) * 100)

plt.figure(figsize=(7, 5))

plt.bar(churn_counts.index, churn_counts.values)

plt.title("Customer Churn Distribution")
plt.xlabel("Churn Status")
plt.ylabel("Number of Customers")

plt.show()

#customer tenure vs churn rate
data["TenureGroup"] = pd.cut(
    data["tenure"],
    bins=[-1, 12, 24, 36, 48, 60, 72],
    labels=["0-12", "13-24", "25-36", "37-48", "49-60", "61-72"]
)

churn_rate = data.groupby("TenureGroup", observed=False)["Churn"].apply(
    lambda x: (x == "Yes").mean() * 100
)

print("\nChurn Rate by Tenure Group:")
print(churn_rate)

plt.figure(figsize=(8, 5))

plt.bar(churn_rate.index.astype(str), churn_rate.values)

plt.title("Customer Churn Rate by Tenure")
plt.xlabel("Customer Tenure (Months)")
plt.ylabel("Churn Rate (%)")

plt.show()

#contract type vs churn rate
contract_churn = data.groupby("Contract")["Churn"].apply(
    lambda x: (x == "Yes").mean() * 100
)

print("\nChurn Rate by Contract Type:")
print(contract_churn)

plt.figure(figsize=(8, 5))

plt.bar(contract_churn.index, contract_churn.values)

plt.title("Customer Churn Rate by Contract Type")
plt.xlabel("Contract Type")
plt.ylabel("Churn Rate (%)")

plt.show()

#montly charges vs churn rate
monthly_charges = data.groupby("Churn")["MonthlyCharges"].mean()

print("\nAverage Monthly Charges by Churn:")
print(monthly_charges)

plt.figure(figsize=(7, 5))

plt.bar(monthly_charges.index, monthly_charges.values)

plt.title("Average Monthly Charges by Churn Status")
plt.xlabel("Churn Status")
plt.ylabel("Average Monthly Charges")

plt.show()

#internet service vs churn rate
internet_churn = data.groupby("InternetService")["Churn"].apply(
    lambda x: (x == "Yes").mean() * 100
)

print("\nChurn Rate by Internet Service:")
print(internet_churn)

plt.figure(figsize=(8, 5))

plt.bar(internet_churn.index, internet_churn.values)

plt.title("Customer Churn Rate by Internet Service")
plt.xlabel("Internet Service")
plt.ylabel("Churn Rate (%)")

plt.show()
'''
#technical support vs churn rate
support_churn = data.groupby("TechSupport")["Churn"].apply(
    lambda x: (x == "Yes").mean() * 100
)

print("\nChurn Rate by Technical Support:")
print(support_churn)