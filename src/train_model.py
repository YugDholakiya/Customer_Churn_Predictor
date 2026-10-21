"""
Customer Churn Prediction - End-to-End Model Training Pipeline
Author: Customer Churn Predictor Team
License: MIT
Description: Trains, evaluates, and exports a production-ready churn prediction model.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score
from xgboost import XGBClassifier


def load_and_clean_data(csv_path: Path) -> pd.DataFrame:
    """Loads raw Telco data, handles missing values, and drops identifier columns."""
    df = pd.read_csv(csv_path)

    # Standardize column naming
    df.columns = [c.strip() for c in df.columns]

    # Convert TotalCharges to numeric, dropping missing records
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.dropna(subset=["TotalCharges"]).copy()

    # Drop non-predictive customerID column if present
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Transforms raw features into machine learning features."""
    data = df.copy()

    # Target variable mapping
    if "Churn" in data.columns:
        data["churn"] = (data["Churn"].str.strip().str.lower() == "yes").astype(int)
        data = data.drop(columns=["Churn"])

    # Binary mappings
    binary_cols = [
        "Partner", "Dependents", "PhoneService", "PaperlessBilling",
        "OnlineSecurity", "OnlineBackup", "DeviceProtection",
        "TechSupport", "StreamingTV", "StreamingMovies", "MultipleLines"
    ]
    for col in binary_cols:
        if col in data.columns:
            data[col.lower()] = data[col].apply(lambda v: 1 if str(v).lower() == "yes" else 0)
            data = data.drop(columns=[col])

    if "gender" in data.columns:
        data["gender"] = (data["gender"].str.lower() == "male").astype(int)

    if "SeniorCitizen" in data.columns:
        data["senior_citizen"] = data["SeniorCitizen"].astype(int)
        data = data.drop(columns=["SeniorCitizen"])

    if "tenure" in data.columns:
        data["tenure"] = data["tenure"].astype(int)
        # Tenure group cohort
        data["tenure_group"] = pd.cut(
            data["tenure"],
            bins=[-1, 12, 24, 48, 100],
            labels=["0-12_months", "12-24_months", "24-48_months", "48+_months"]
        )

    # Derive engagement count
    services = ["online_security", "online_backup", "device_protection", "tech_support", "streaming_tv", "streaming_movies"]
    existing_services = [s for s in services if s in data.columns]
    data["services_count"] = data[existing_services].sum(axis=1)

    # Rename continuous charges
    if "MonthlyCharges" in data.columns:
        data["monthly_charges"] = data["MonthlyCharges"].astype(float)
        data = data.drop(columns=["MonthlyCharges"])

    if "TotalCharges" in data.columns:
        data["total_charges"] = data["TotalCharges"].astype(float)
        data = data.drop(columns=["TotalCharges"])

    # Derived average spend
    safe_tenure = data["tenure"].replace(0, 1)
    data["avg_monthly_spend"] = data["total_charges"] / safe_tenure

    # Standardize categorical column names
    cat_rename = {
        "InternetService": "internet_service",
        "Contract": "contract",
        "PaymentMethod": "payment_method"
    }
    for old_c, new_c in cat_rename.items():
        if old_c in data.columns:
            data[new_c] = data[old_c]
            data = data.drop(columns=[old_c])

    # One-hot encode remaining categoricals
    cat_cols = [c for c in ["internet_service", "contract", "payment_method", "tenure_group"] if c in data.columns]
    data = pd.get_dummies(data, columns=cat_cols, dtype=int)

    return data


def run_pipeline():
    base_dir = Path(__file__).resolve().parents[1]
    raw_path = base_dir / "data" / "raw_data" / "raw_telco_churn_data.csv"
    output_dir = base_dir / "saved_models"
    streamlit_models_dir = base_dir / "streamlit" / "saved_models"

    output_dir.mkdir(parents=True, exist_ok=True)
    streamlit_models_dir.mkdir(parents=True, exist_ok=True)

    print("Step 1: Loading raw dataset...")
    raw_df = load_and_clean_data(raw_path)

    print("Step 2: Engineering features...")
    processed_df = engineer_features(raw_df)

    y = processed_df["churn"]
    X = processed_df.drop(columns=["churn"])
    training_cols = list(X.columns)

    print("Step 3: Splitting dataset (80/20 stratified)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    print("Step 4: Training XGBoost Classifier...")
    model = XGBClassifier(
        n_estimators=150,
        max_depth=4,
        learning_rate=0.08,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric="logloss"
    )
    model.fit(X_train, y_train)

    print("Step 5: Evaluating model...")
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_proba)
    print(f"Accuracy : {acc:.4f}")
    print(f"ROC-AUC  : {roc:.4f}")
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    print("Step 6: Exporting model artifacts...")
    for target_dir in [output_dir, streamlit_models_dir]:
        joblib.dump(model, target_dir / "best_churn_prediction_model.pkl")
        joblib.dump(training_cols, target_dir / "training_columns.pkl")

    print("Pipeline executed successfully!")


if __name__ == "__main__":
    run_pipeline()
