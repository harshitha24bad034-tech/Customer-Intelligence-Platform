import streamlit as st
import pandas as pd
import joblib

# PAGE CONFIGURATION

st.set_page_config(
    page_title="Customer Intelligence Platform",
    page_icon="📊",
    layout="wide"
)

# CUSTOM STYLING

st.markdown("""
<style>

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

h1 {
    color: #4DA8DA;
    font-size: 36px;
}

h2, h3 {
    color: #4DA8DA;
}

[data-testid="stMetric"] {
    background-color: #1E293B;
    padding: 20px;
    border-radius: 12px;
    border: 1px solid #334155;
}

[data-testid="stMetricValue"] {
    color: #4DA8DA;
}

.stButton > button {
    border-radius: 8px;
    font-weight: bold;
}

</style>
""", unsafe_allow_html=True)


# LOAD DATA AND MODEL

data = pd.read_csv("data/processed/cleaned_customer_data.csv")

model = joblib.load("models/churn_prediction_pipeline.pkl")


# DASHBOARD TITLE

st.title("Customer Intelligence & Predictive Analytics Platform")

st.write(
    "An interactive platform to analyse customer behaviour "
    "and predict customer churn using Machine Learning."
)


# CUSTOMER OVERVIEW

st.divider()

st.header("Customer Overview")

total_customers = len(data)

churned_customers = (data["Churn"] == "Yes").sum()

retained_customers = (data["Churn"] == "No").sum()

churn_rate = (churned_customers / total_customers) * 100

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Customers", total_customers)

col2.metric("Churned Customers", churned_customers)

col3.metric("Retained Customers", retained_customers)

col4.metric("Churn Rate", f"{churn_rate:.2f}%")


# CHURN DISTRIBUTION

st.subheader("Customer Churn Distribution")

churn_distribution = data["Churn"].value_counts()

st.bar_chart(churn_distribution)


# BUSINESS INSIGHTS

st.divider()

st.header("Business Insights & Analytics")

st.write(
    "Explore customer churn patterns to identify "
    "important business insights."
)


# 1. CHURN BY CONTRACT TYPE

st.subheader("1. Churn Rate by Contract Type")

contract_churn = data.groupby("Contract")["Churn"].apply(
    lambda x: (x == "Yes").mean() * 100
)

st.bar_chart(contract_churn)


# 2. CHURN BY TENURE

st.subheader("2. Churn Rate by Customer Tenure")

data["TenureGroup"] = pd.cut(
    data["tenure"],
    bins=[-1, 12, 24, 36, 48, 60, 72],
    labels=["0-12", "13-24", "25-36", "37-48", "49-60", "61-72"]
)

tenure_churn = data.groupby(
    "TenureGroup",
    observed=False
)["Churn"].apply(
    lambda x: (x == "Yes").mean() * 100
)

st.bar_chart(tenure_churn)


# 3. CHURN BY INTERNET SERVICE

st.subheader("3. Churn Rate by Internet Service")

internet_churn = data.groupby("InternetService")["Churn"].apply(
    lambda x: (x == "Yes").mean() * 100
)

st.bar_chart(internet_churn)


# 4. AVERAGE MONTHLY CHARGES

st.subheader("4. Average Monthly Charges by Churn Status")

average_charges = data.groupby("Churn")["MonthlyCharges"].mean()

st.bar_chart(average_charges)


# 5. CHURN BY TECHNICAL SUPPORT

st.subheader("5. Churn Rate by Technical Support")

support_churn = data.groupby("TechSupport")["Churn"].apply(
    lambda x: (x == "Yes").mean() * 100
)

st.bar_chart(support_churn)


# CUSTOMER CHURN PREDICTION

st.divider()

st.header("Customer Churn Prediction")

st.write(
    "Enter customer details to predict their likelihood "
    "of leaving the company."
)

with st.form("churn_prediction_form"):

    col1, col2, col3 = st.columns(3)

    with col1:

        gender = st.selectbox("Gender", ["Male", "Female"])

        senior_citizen = st.selectbox("Senior Citizen", [0, 1])

        partner = st.selectbox("Has Partner?", ["Yes", "No"])

        dependents = st.selectbox("Has Dependents?", ["Yes", "No"])

        tenure = st.number_input(
            "Tenure (Months)",
            0,
            72,
            12
        )

        phone_service = st.selectbox(
            "Phone Service",
            ["Yes", "No"]
        )

    with col2:

        multiple_lines = st.selectbox(
            "Multiple Lines",
            ["Yes", "No", "No phone service"]
        )

        internet_service = st.selectbox(
            "Internet Service",
            ["DSL", "Fiber optic", "No"]
        )

        online_security = st.selectbox(
            "Online Security",
            ["Yes", "No", "No internet service"]
        )

        online_backup = st.selectbox(
            "Online Backup",
            ["Yes", "No", "No internet service"]
        )

        device_protection = st.selectbox(
            "Device Protection",
            ["Yes", "No", "No internet service"]
        )

        tech_support = st.selectbox(
            "Tech Support",
            ["Yes", "No", "No internet service"]
        )

    with col3:

        streaming_tv = st.selectbox(
            "Streaming TV",
            ["Yes", "No", "No internet service"]
        )

        streaming_movies = st.selectbox(
            "Streaming Movies",
            ["Yes", "No", "No internet service"]
        )

        contract = st.selectbox(
            "Contract",
            ["Month-to-month", "One year", "Two year"]
        )

        paperless_billing = st.selectbox(
            "Paperless Billing",
            ["Yes", "No"]
        )

        payment_method = st.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)"
            ]
        )

        monthly_charges = st.number_input(
            "Monthly Charges",
            min_value=0.0,
            value=70.0
        )

    total_charges = st.number_input(
        "Total Charges",
        min_value=0.0,
        value=840.0
    )

    predict_button = st.form_submit_button("Predict Churn")


# PREDICTION RESULT

if predict_button:

    customer = pd.DataFrame([{
        "gender": gender,
        "SeniorCitizen": senior_citizen,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges
    }])

    prediction = model.predict(customer)[0]

    probability = model.predict_proba(customer)[0][1] * 100

    st.subheader("Prediction Result")

    if prediction == 1:

        st.error("Customer is predicted to churn.")

    else:

        st.success("Customer is predicted to stay.")

    st.metric(
        "Churn Probability",
        f"{probability:.2f}%"
    )

    st.progress(float(probability / 100))




st.divider()

st.caption(
    "Customer Intelligence & Predictive Analytics Platform | "
    "Built using Python, Scikit-learn and Streamlit"
)