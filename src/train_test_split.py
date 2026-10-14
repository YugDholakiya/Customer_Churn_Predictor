# ==========================================================
# Script for Train/Test Dataset Split
# ==========================================================

"""
Purpose:
--------
Prepare and split the transformed dataset into
training and testing datasets for machine learning
model training and evaluation.

Steps Performed:
1. Load feature-engineered (transformed) dataset
2. Verify dataset structure
3. Separate feature variables and target variable
4. Perform Train/Test split
5. Verify dataset shapes
6. Save training and testing datasets
"""

# ==========================================================
# 1. Import Required Libraries
# ==========================================================

import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path


# ==========================================================
# 2. Define File Paths
# ==========================================================

INPUT_PATH = Path("data/transformed_data/transformed_data.csv")

X_TRAIN_PATH = Path("data/model_input/x_train.csv")
X_TEST_PATH = Path("data/model_input/x_test.csv")
Y_TRAIN_PATH = Path("data/model_input/y_train.csv")
Y_TEST_PATH = Path("data/model_input/y_test.csv")


# ==========================================================
# 3. Load Feature Engineered Dataset
# ==========================================================

print("\nLoading feature engineered dataset...")

df = pd.read_csv(INPUT_PATH)

print("\nDataset loaded successfully.")
print("\nDataset Shape:", df.shape)


# ==========================================================
# 4. Verify Dataset Structure
# ==========================================================

print("\nDataset Columns:\n")
print(df.columns)


# ==========================================================
# 5. Separate Features and Target Variable
# ==========================================================

print("\nSeparating feature variables and target variable...")

# Target variable
y = df["churn"]

# Feature variables
x = df.drop("churn", axis=1)

print("\nFeature and target variables separated.")


# ==========================================================
# 6. Perform Train/Test Split
# ==========================================================

print("\nPerforming Train/Test Split...")

X_train, X_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,        # 80% train, 20% test
    random_state=42,      # reproducibility
    stratify=y            # preserve churn class distribution
)

print("\nTrain/Test split completed.")


# ==========================================================
# 7. Verify Training and Testing Dataset Shapes
# ==========================================================

print("\nTraining Dataset Shape (inputs -> x) :", X_train.shape)
print("\nTesting Dataset Shape (inputs -> x) :", X_test.shape)


# ==========================================================
# 8. Preview Training and Testing Datasets
# ==========================================================

print("\nPreview of Training Dataset (inputs/features) :\n")
print(X_train)

print("\nPreview of Training Dataset (output/target) :\n")
print(y_train)

print("\nPreview of Testing Dataset (inputs/features) :\n")
print(X_test)

print("\nPreview of Testing Dataset (output/target) :\n")
print(y_test)


# ==========================================================
# 10. Save Training and Testing Datasets
# ==========================================================

print("\nSaving training and testing datasets...")


X_train.to_csv(X_TRAIN_PATH, index=False)
X_test.to_csv(X_TEST_PATH, index=False)
y_train.to_csv(Y_TRAIN_PATH, index=False)
y_test.to_csv(Y_TEST_PATH, index=False)

print("Datasets saved successfully.")


# ==========================================================
# Final Confirmation
# ==========================================================

print("\nTrain/Test Split completed successfully.")

# ==========================================================
# End of Train-Test Split Script
# ==========================================================