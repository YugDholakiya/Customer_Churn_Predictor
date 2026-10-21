# 📊 Customer Churn Predictor

[![Python](https://img.shields.io/badge/Python-3.x-blue)](https://www.python.org/)
[![Scikit-learn](https://img.shields.io/badge/ML-Scikit--Learn-orange)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/App-Streamlit-red)](https://streamlit.io/)
[![Live App](https://img.shields.io/badge/Live%20App-Streamlit-success?logo=streamlit)](https://customerchurnpredictor-7dzaawv2mb6p8uvarp3n9c.streamlit.app/)

Built with **Python, Scikit-learn, Pandas, SHAP, Plotly, and Streamlit**, this project delivers an **end-to-end machine learning system:**

**Raw Dataset → Data Cleaning → Feature Engineering → Train-Test Split → Model Training → Model Evaluation → Model Explainability → Model Deployment (Streamlit App).**

---

## 🌐 Live Application

🔗 **Try the deployed application**

https://customerchurnpredictor-7dzaawv2mb6p8uvarp3n9c.streamlit.app/

This interactive dashboard allows users to input telecom customer attributes and instantly predict the **probability of churn** alongside **feature-level SHAP explanations**.

---

## 🚀 Project Overview

- **Objective:** Predict whether a telecom customer is likely to churn.
- **Dataset:** [IBM Telco Customer Churn Dataset (Kaggle)](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- **ML Pipeline:** Data cleaning, feature engineering, model training, evaluation, explainability
- **Deployment:** Interactive **Streamlit ML dashboard**.
- **Model:** XGBoost Classifier
- **ROC-AUC Score:** **0.832**
- **Accuracy:** **0.787**

---

## 📊 Dataset Information

Dataset Source:

> IBM Telco Customer Churn Dataset (Kaggle)

**Dataset Size**

| Metric | Value |
|------|------|
Rows | **7043**
Columns | **21**

---

## ⚙️ Machine Learning Pipeline

```
            Raw Dataset
                ↓
           Data Cleaning
                ↓
        Feature Engineering
                ↓
         Feature Encoding
                ↓
     Train/Test Split (80/20)
                ↓
          Model Training
                ↓
         Model Evaluation
                ↓
        Model Explainability
                ↓
     Streamlit Web Application
```

---

## 🧹 Data Cleaning

Key preprocessing steps performed:

- Removed non-predictive identifier columns (`customerID`)
- Standardized column names to **snake_case**
- Converted categorical values to consistent formats
- Fixed incorrect data types (`total_charges` converted to numeric)
- Converted target variable **churn → binary (0/1)**

---

## 🏗 Feature Engineering

Additional features were engineered to capture customer behavior:

### 1️⃣ Tenure Group

Customers grouped into lifecycle stages:
- `0-12 months`
- `12-24 months`
- `24-48 months`
- `48+ months`

### 2️⃣ Services Count

Counts number of subscribed optional services:
- `services_count = sum(service features)`
- Captures overall customer engagement level.

### 3️⃣ Average Monthly Spend

- `avg_monthly_spend = total_charges / tenure`
- Discovers spending velocity and tenure anomalies.

---

## 🔢 Feature Encoding

### Binary Encoding
- `Yes → 1`
- `No → 0`

Applied to:
- partner, dependents, phone_service, online_security, online_backup, device_protection, tech_support, streaming_tv, streaming_movies, multiple_lines, paperless_billing

### One-Hot Encoding
Applied to multi-category features:
- `internet_service`, `contract`, `payment_method`, `tenure_group`

---

## 🏆 Production Model Performance

**XGBoost Classifier**

| Metric | Score |
|------|------|
Accuracy | **0.787**
ROC-AUC | **0.832**

---

## 🔍 Model Explainability

SHAP (SHapley Additive exPlanations) values provide local and global explainability for each customer prediction.

Key influencing features:
- Contract type (Month-to-month vs. Annual)
- Tenure length
- Internet service type (Fiber optic vs. DSL)
- Monthly charges
- Technical support availability

---

## 🖥 Streamlit Dashboard

The trained ML model is deployed as an **interactive web application** using Streamlit.

### Dashboard Features

- Interactive **customer attribute inputs**
- Real-time **churn probability gauge**
- **Risk classification (Low / Medium / High)**
- **SHAP-based prediction explanation chart**
- **Tailored customer retention recommendations**
- Clean and responsive **UI layout**

---

## 📊 Risk Classification

| Probability | Risk Level |
|-------------|------------|
0 – 40% | Low Risk
40 – 70% | Medium Risk
70 – 100% | High Risk

---

## 🗂 Project Structure

```bash
customer-churn-prediction-system
│
├── data/
│   ├── raw_data/
│   │   └── raw_telco_churn_data.csv
│   └── transformed_data/
│
├── saved_models/
│   ├── best_churn_prediction_model.pkl
│   └── training_columns.pkl
│
├── src/
│   └── train_model.py
│
├── reports/
│   └── figures/
│
├── streamlit/
│   ├── app.py
│   ├── components/
│   │   └── prediction_ui.py
│   ├── saved_models/
│   │   ├── best_churn_prediction_model.pkl
│   │   └── training_columns.pkl
│   └── utils/
│       ├── model_loader.py
│       ├── preprocessing.py
│       └── prediction.py
│
├── requirements.txt
├── requirements-dev.txt
└── README.md
```

---

## 🚀 Run Locally

Clone repository:

```bash
git clone <your-repository-url>
```

Navigate to project directory:

```bash
cd customer-churn-prediction-system
```

Create Python Virtual Environment:

```bash
python -m venv .venv
```

Install dependencies:

```bash
pip install -r requirements-dev.txt
```

Run App Locally:

```bash
streamlit run streamlit/app.py
```

---

## ☁ Deployment

The application is deployed on **Streamlit Community Cloud**:

1. Push your project to your GitHub repository
2. Connect the repository to **Streamlit Community Cloud**
3. Set the app entry point to:
```bash
streamlit/app.py
```
