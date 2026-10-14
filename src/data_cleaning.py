
"""
NOTE | Data Cleaning Script
--------------------------------------------------------------------------------

This script performs data cleaning on the raw Telco Customer Churn dataset.

Steps Performed:
1. Import required libraries
2. Load raw dataset
3. Verify dataset structure
4. Standardize column names
5. Remove identifier column
6. Fix incorrect data types
7. Handle missing values
8. Remove duplicate records
9. Reset row indexing
10. Verify dataset columns and shape
11. Save the cleaned dataset for feature engineering

"""

# ==========================================================
# 1. Import Required Libraries
# ==========================================================

import pandas as pd


# ==========================================================
# 2. Load Raw Dataset
# ==========================================================

# Path to raw dataset
RAW_DATA_PATH = "data/raw_data/raw_telco_churn_data.csv"

# Load dataset
df = pd.read_csv(RAW_DATA_PATH)

print("\nRaw Dataset Loaded Successfully.")


# ==========================================================
# 3. Verify Dataset Structure
# ==========================================================

print("\nDataset Shape (Rows, Columns):")
print(df.shape)

print("\nDataset Columns:")
print(df.columns)

print("\nDataset Data Types:")
print(df.dtypes)


# ==========================================================
# 4. Standardize Column Names (Convert to snake_case)
# ==========================================================

df.rename(columns={
    "customerID": "customer_id",
    "gender": "gender",
    "SeniorCitizen": "senior_citizen",
    "Partner": "partner",
    "Dependents": "dependents",
    "tenure": "tenure",
    "PhoneService": "phone_service",
    "MultipleLines": "multiple_lines",
    "InternetService": "internet_service",
    "OnlineSecurity": "online_security",
    "OnlineBackup": "online_backup",
    "DeviceProtection": "device_protection",
    "TechSupport": "tech_support",
    "StreamingTV": "streaming_tv",
    "StreamingMovies": "streaming_movies",
    "Contract": "contract",
    "PaperlessBilling": "paperless_billing",
    "PaymentMethod": "payment_method",
    "MonthlyCharges": "monthly_charges",
    "TotalCharges": "total_charges",
    "Churn": "churn"
}, inplace=True)

print("\nColumn names standardized to snake_case : \n")
print(df.columns)
print()


# ==========================================================
# 5. Remove Identifier Column
# ==========================================================

# customer_id is only an identifier and not useful for ML models
df.drop("customer_id", axis=1, inplace=True)

print("\n'customer_id' column removed.")


# ==========================================================
# 6. Fix Incorrect Data Types
# ==========================================================

# Convert total_charges from object to numeric
df["total_charges"] = pd.to_numeric(df["total_charges"], errors="coerce")

print("\n'total_charges' converted to numeric data type.")


# ==========================================================
# 7. Handle Missing Values
# ==========================================================

# Remove rows with missing values
df = df.dropna()

print("\nMissing values handled (rows with NaN removed).")


# ==========================================================
# 8. Remove Duplicate Records
# ==========================================================

# Remove duplicate rows if any
df = df.drop_duplicates()

print("\nDuplicate rows removed.")


# ==========================================================
# 9. Reset Row Indexing
# ==========================================================

# Reset index after removing rows
df.reset_index(drop=True, inplace=True)

print("\nRow indexing reseted.")


# ==========================================================
# 10. Verify Dataset Columns and Shape
# ==========================================================

print("\nFinal Dataset Shape:")
print(df.shape)

print("\nFinal Dataset Columns:")
print(df.columns)


# ==========================================================
# 11. Save cleaned Dataset
# ==========================================================

OUTPUT_DATA_PATH = "data/transformed_data/cleaned_data.csv"

df.to_csv(OUTPUT_DATA_PATH, index=False)

print("\nPreprocessed / cleaned dataset saved successfully.")
print(f"Saved at: {OUTPUT_DATA_PATH}\n")


################################################################################