# ==========================================================
# Gradient Boosting Model Training Script
# ==========================================================

"""
Purpose
-------
Train and evaluate a Gradient Boosting model for
telecom customer churn prediction.

Steps
-----
1. Load training and testing datasets
2. Train Gradient Boosting model
3. Generate predictions on test data
4. Evaluate model performance
5. Save trained model for later use
"""

# ==========================================================
# Import Libraries
# ==========================================================

from pathlib import Path
import pandas as pd
import joblib

from sklearn.ensemble import GradientBoostingClassifier

# Import evaluation module
from src.models_evaluation.evaluate_models import evaluate_model


# ==========================================================
# File Paths
# ==========================================================

DATA_DIR = Path("data/model_input")
MODEL_DIR = Path("saved_models")

# Ensure model directory exists
MODEL_DIR.mkdir(parents=True, exist_ok=True)

X_TRAIN_PATH = DATA_DIR / "x_train.csv"
X_TEST_PATH = DATA_DIR / "x_test.csv"

Y_TRAIN_PATH = DATA_DIR / "y_train.csv"
Y_TEST_PATH = DATA_DIR / "y_test.csv"

MODEL_PATH = MODEL_DIR / "04_gradient_boosting_model.pkl"


# ==========================================================
# Load Dataset
# ==========================================================

def load_data():
    """
    Load training and testing datasets.
    """

    print("\nLoading training and testing datasets...")

    X_train = pd.read_csv(X_TRAIN_PATH)
    X_test = pd.read_csv(X_TEST_PATH)

    y_train = pd.read_csv(Y_TRAIN_PATH).squeeze()
    y_test = pd.read_csv(Y_TEST_PATH).squeeze()

    print("Datasets loaded successfully.")

    print(f"X_train shape : {X_train.shape}")
    print(f"X_test shape  : {X_test.shape}")

    return X_train, X_test, y_train, y_test


# ==========================================================
# Train Gradient Boosting Model
# ==========================================================

def train_gradient_boosting():
    """
    Train Gradient Boosting model and evaluate performance.
    """

    print("\n==================================================")
    print("Training Gradient Boosting Model")
    print("==================================================")

    # ------------------------------------------------------
    # Load Data
    # ------------------------------------------------------

    X_train, X_test, y_train, y_test = load_data()

    # ------------------------------------------------------
    # Initialize Model
    # ------------------------------------------------------

    model = GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        random_state=42
    )

    print("\nFitting Gradient Boosting model...")

    # ------------------------------------------------------
    # Train Model
    # ------------------------------------------------------

    model.fit(X_train, y_train)

    print("Model training completed.")

    # ------------------------------------------------------
    # Generate Predictions
    # ------------------------------------------------------

    print("\nGenerating predictions...")

    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    # ------------------------------------------------------
    # Evaluate Model
    # ------------------------------------------------------

    metrics = evaluate_model(
        model_name="gradient_boosting",
        y_test=y_test,
        y_pred=y_pred,
        y_pred_proba=y_pred_proba
    )

    # ------------------------------------------------------
    # Save Trained Model
    # ------------------------------------------------------

    print("\nSaving trained model...")

    joblib.dump(model, MODEL_PATH)

    print(f"Model saved successfully at: {MODEL_PATH}")

    return model, metrics


# ==========================================================
# End of '04_gradient_boosting_model.py' Script.
# ==========================================================