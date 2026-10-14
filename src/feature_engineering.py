
# ==========================================================
# Feature Engineering Script
# ==========================================================
# This script performs feature engineering on the cleaned
# telecom churn dataset.
#
# Steps performed:
# 1. Load cleaned dataset
# 2. Verify dataset structure
# 3. Convert target variable (churn)
# 4. Encode binary categorical features
# 5. Encode multi-category features (One-Hot Encoding)
# 6. Create engineered features that may improve
#    machine learning model performance.
# 7. Verify feature engineering results
# 8. Check dataset columns and shape
# 9. Save transformed dataset
# ==========================================================


# ==========================================================
# 1. Import Required Libraries
# ==========================================================

import pandas as pd
import numpy as np
from pathlib import Path


# ==========================================================
# 2. Load Preprocessed Dataset
# ==========================================================

print("\nLoading cleaned dataset...")

INPUT_DATA_PATH = Path("data/transformed_data/cleaned_data.csv")

df = pd.read_csv(INPUT_DATA_PATH)

print("\nDataset loaded successfully.")


# ==========================================================
# 3. Verify Dataset Structure
# ==========================================================

print("\nDataset Shape:", df.shape)
print("\nDataset Columns:")
print(df.columns)


# ==========================================================
# 4. Convert Target Variable (Churn)
# ==========================================================

# Convert churn column to binary
# Yes -> 1
# No  -> 0

print("\nConverting target variable 'churn' to numeric format...")

df["churn"] = df["churn"].map({
    "Yes": 1,
    "No": 0
})

print("\nTarget variable converted.")


# ==========================================================
# 5. Encode Binary Categorical Features
# ==========================================================

print("\nEncoding binary categorical features...")

binary_columns = [
    "partner",
    "dependents",
    "phone_service",
    "paperless_billing",
    "online_security",
    "online_backup",
    "device_protection",
    "tech_support",
    "streaming_tv",
    "streaming_movies",
    "multiple_lines"
]

for col in binary_columns:
    df[col] = df[col].map({
        "Yes": 1,
        "No": 0,
        "No internet service": 0,
        "No phone service": 0
    })

print("\nBinary categorical features encoded.")


# ==========================================================
# Encode Gender Column
# ==========================================================

print("\nEncoding 'gender' column...")

df["gender"] = df["gender"].map({
    "Male": 1,
    "Female": 0
})

print("\nGender column encoded.")


# ==========================================================
# 6. Encode Multi-Category Features (One-Hot Encoding)
# ==========================================================

print("\nApplying One-Hot Encoding on Multi-Category Features...")

multi_category_columns = [
    "internet_service",
    "contract",
    "payment_method"
]

df = pd.get_dummies(df, columns=multi_category_columns, dtype=int)

print("\nOne-Hot Encoding completed.")


# ==========================================================
# 7. Creating Engineered Features
# ==========================================================

print("\nCreating engineered features...")


# ----------------------------------------------------------
# a) Tenure Group Feature
# ----------------------------------------------------------
# Group customers into lifecycle stages by tenure length.

print("\nCreating 'tenure_group' feature...")

def create_tenure_group(tenure):

    if tenure <= 12:
        return "0-12_months"

    elif tenure <= 24:
        return "12-24_months"

    elif tenure <= 48:
        return "24-48_months"

    else:
        return "48+_months"


df["tenure_group"] = df["tenure"].apply(create_tenure_group)

print("\n'tenure_group' feature created.")

print("\nApplying One-Hot Encoding on 'tenure_group' feature...")

# One-hot encode tenure_group
df = pd.get_dummies(df, columns=["tenure_group"], dtype=int)

print("\nEncoded 'tenure_group' feature with One-Hot Encoding.")


# ----------------------------------------------------------
# b) Services Count Feature
# ----------------------------------------------------------
# Count number of subscribed optional services
# Higher engagement usually reduces churn risk.


print("\n\nCalculating service usage count...")
print("\nCreating 'services_count' feature...")

service_columns = [
    "online_security",
    "online_backup",
    "device_protection",
    "tech_support",
    "streaming_tv",
    "streaming_movies"
]

df["services_count"] = df[service_columns].sum(axis=1)

print("\nCreated 'services_count' feature successfully.")



# ----------------------------------------------------------
# c) Average Monthly Spend Feature
# ----------------------------------------------------------
# avg_monthly_spend = total_charges / tenure

print("\n\nCalculating the average of monthly spend...")
print("\nCreating 'avg_monthly_spend' feature...")

df["avg_monthly_spend"] = df["total_charges"] / df["tenure"]

# Handle division by zero if tenure = 0
df["avg_monthly_spend"] = df["avg_monthly_spend"].replace([np.inf, -np.inf], 0)


print("\nCreated 'avg_monthly_spend' feature successfully.")

print("\n\nAll the engineered (derived) features created successfully.")



# ==========================================================
# 8. Check Dataset Columns and Shape after Feature Engg.
# ==========================================================

print("\nFinal Dataset Shape (rows, columns):", df.shape)

print("\nFinal Dataset Columns:\n")
print(df.columns)


# ==========================================================
# 9. Verify Feature Engineering Results
# ==========================================================

print("\nChecking data-types of all the columns in the dataset...\n")

print(df.dtypes)


# ==========================================================
# 10.  Verify New / Transformed Features
# ==========================================================

print("\n\nPreview of the entire dataframe after engineered (derived & encoded) feature's creation:\n")

print(df)


# ==========================================================
# 11. Save Feature Engineered Dataset
# ==========================================================

print("\nSaving transformed dataset after completing the Feature Engineering...")

OUTPUT_DATA_PATH = Path("data/transformed_data/transformed_data.csv")

df.to_csv(OUTPUT_DATA_PATH, index=False)

print(f"Feature engineered (final transformed) dataset saved successfully at {OUTPUT_DATA_PATH}.")


# ==========================================================
# 12. Final Confirmation
# ==========================================================

print("\nFeature Engineering pipeline completed successfully.")


# ==========================================================
# End of Feature Engineering Script
# ==========================================================



'''
NOTE | Why These Features Matter ...


a) Tenure Groups:
    - Instead of using only numeric tenure, we capture customer lifecycle stages.
    - Example insight:
        - New customers churn more frequently.

b) Services Count:
    - Customers using more services usually show higher engagement.
    - Example:
        - services_count = 1  --->	churn risk : high
        - services_count = 5  --->	churn risk : low

c) Average Monthly Spend:
    - Combines: total_charges and tenure
    - This reveals spending behavior patterns.
    - Example: high spend + low tenure → potential churn risk

'''