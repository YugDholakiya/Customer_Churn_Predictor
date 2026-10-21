import numpy as np
import pandas as pd


def get_tenure_group(tenure: int) -> str:
    """Classifies customer tenure into lifecycle stages."""
    if tenure <= 12:
        return "0-12_months"
    elif tenure <= 24:
        return "12-24_months"
    elif tenure <= 48:
        return "24-48_months"
    return "48+_months"


def preprocess_input(user_input: dict, training_columns: list) -> pd.DataFrame:
    """Transforms a single customer input dictionary into aligned model features."""
    df = pd.DataFrame([user_input])

    # Binary features mapping (Yes -> 1, otherwise 0)
    binary_fields = [
        "partner", "dependents", "phone_service", "paperless_billing",
        "online_security", "online_backup", "device_protection",
        "tech_support", "streaming_tv", "streaming_movies", "multiple_lines"
    ]
    for field in binary_fields:
        if field in df.columns:
            df[field] = (df[field].astype(str).str.lower() == "yes").astype(int)

    if "gender" in df.columns:
        df["gender"] = (df["gender"].astype(str).str.lower() == "male").astype(int)

    # Derived lifecycle and engagement features
    tenure_val = int(df["tenure"].iloc[0])
    df["tenure_group"] = get_tenure_group(tenure_val)

    service_fields = [
        "online_security", "online_backup", "device_protection",
        "tech_support", "streaming_tv", "streaming_movies"
    ]
    df["services_count"] = df[service_fields].sum(axis=1)

    # Average monthly spend
    safe_tenure = max(tenure_val, 1)
    df["avg_monthly_spend"] = df["total_charges"].astype(float) / safe_tenure

    # One-hot encode categorical features
    df = pd.get_dummies(df)

    # Align exactly with the model's training columns
    df = df.reindex(columns=training_columns, fill_value=0)
    return df
