# Customer Intelligence & Predictive Analytics Platform

An end-to-end machine learning project that analyses telecom customer behaviour, identifies churn patterns, and predicts the likelihood of customer churn through an interactive Streamlit dashboard.

## 1. Problem Statement

Customer churn occurs when customers stop using a company's services. Identifying customers who are likely to leave helps businesses understand customer behaviour and take steps to improve customer retention.

This project uses machine learning to analyse customer data and predict potential churn.

## 2. Project Objectives

- Analyse customer information and behaviour.
- Identify patterns associated with customer churn.
- Compare machine learning classification models.
- Predict the likelihood of customer churn.
- Present business insights through an interactive dashboard.

## 3. Technologies Used

- **Programming:** Python
- **Data Analysis:** Pandas, NumPy
- **Data Visualization:** Matplotlib, Seaborn
- **Machine Learning:** Scikit-learn
- **Model Storage:** Joblib
- **Web Application:** Streamlit

## 4. Dataset

**Dataset:** IBM Telco Customer Churn Dataset

- Total records: 7,043
- Total original columns: 21
- Target variable: Churn
- Prediction classes: Yes and No

The dataset contains customer information such as tenure, contract type, internet service, monthly charges, and other service details.

## 5. Project Workflow

1. **Data Exploration:** Examined the dataset and understood customer characteristics.
2. **Data Cleaning:** Handled missing values and prepared the data for analysis.
3. **Exploratory Data Analysis:** Identified churn patterns using charts and visualizations.
4. **Data Preparation:** Separated features and target variables and performed an 80:20 train-test split.
5. **Preprocessing:** Applied StandardScaler to numerical features and OneHotEncoder to categorical features.
6. **Model Training:** Trained and compared Logistic Regression and Random Forest.
7. **Model Evaluation:** Evaluated models using accuracy, precision, recall, F1-score, and ROC-AUC.
8. **Deployment:** Built an interactive Streamlit dashboard for customer analytics and churn prediction.

## 6. Model Performance

Two classification models were evaluated on the held-out test dataset.

| Metric | Logistic Regression | Random Forest |
|---|---:|---:|
| Accuracy | 80.55% | 78.28% |
| Precision (Churn) | 66% | 62% |
| Recall (Churn) | 56% | 47% |
| F1-score (Churn) | 60% | 54% |
| ROC-AUC | 0.8421 | 0.8232 |

**Selected Model: Logistic Regression**

Logistic Regression achieved better results across the reported evaluation metrics and was selected for the final prediction pipeline.

## 7. Key Business Insights

The exploratory analysis identified several patterns:

- Customers with month-to-month contracts showed a higher churn rate.
- Customers with shorter tenure were more likely to churn.
- Customers with higher average monthly charges showed a higher churn rate.
- Customers without technical support showed a higher churn rate.
- Fiber optic customers showed a higher churn rate than other internet service groups.

These findings represent associations observed in the dataset, not proof of cause and effect.

## 8. Application Features

- Customer overview with total, retained, and churned customer counts.
- Interactive churn distribution visualization.
- Churn analysis based on contract type, tenure, and internet service.
- Technical support and monthly charge analysis.
- Interactive customer information form.
- Individual churn prediction with estimated probability.

## 9. Project Structure

```text
Customer Intelligence Platform/
│
├── data/
│   ├── raw/
│   │   └── Telco-Customer-Churn.csv
│   └── processed/
│       └── cleaned_customer_data.csv
│
├── src/
│   ├── explore_data.py
│   ├── clean_data.py
│   ├── eda.py
│   └── prepare_data.py
│
├── models/
│   └── churn_prediction_pipeline.pkl
│
├── notebooks/
│
├── app.py
├── requirements.txt
└── README.md
```

## 10. Installation and Execution

### Step 1: Clone the repository

```bash
git clone <your-github-repository-link>
```

### Step 2: Navigate to the project folder

```bash
cd "Customer Intelligence Platform"
```

### Step 3: Install dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Run the application

```bash
streamlit run app.py
```

The application will open in your browser.

## 11. Limitations

- The model is trained using a specific telecom customer dataset.
- Prediction probabilities are estimates and are not guaranteed outcomes.
- Model performance may vary with new or different customer data.
- The project is a demonstration application and has not been deployed as a production business system.

## 12. Future Improvements

- Experiment with additional machine learning models.
- Improve churn detection using model validation and threshold analysis.
- Add more customer segmentation insights.
- Explore deployment through a cloud platform.

## Author

**Harshitha S K**  
B.Tech – Artificial Intelligence and Data Science  
Kumaraguru College of Technology