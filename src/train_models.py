# ==========================================================
# Model Training Pipeline
# ==========================================================

"""
Purpose
-------
Run the full machine learning experimentation pipeline
for telecom customer churn prediction.

Steps
-----
1. Train multiple models
2. Evaluate model performance
3. Compare evaluation metrics
4. Select best performing model
5. Save best model for deployment
"""

# ==========================================================
# Import Libraries
# ==========================================================

from pathlib import Path
import pandas as pd
import joblib


# ==========================================================
# Import Model Training Functions
# ==========================================================

from src.models_training.logistic_regression_model import train_logistic_regression
from src.models_training.decision_tree_model import train_decision_tree
from src.models_training.random_forest_model import train_random_forest
from src.models_training.gradient_boosting_model import train_gradient_boosting
from src.models_training.xgboost_model import train_xgboost


# ==========================================================
# Paths
# ==========================================================

MODEL_DIR = Path("saved_models")
REPORT_DIR = Path("reports")

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

BEST_MODEL_PATH = MODEL_DIR / "best_churn__prediction_model.pkl"
COMPARISON_PATH = REPORT_DIR / "model_comparison.csv"


# ==========================================================
# Run Training Pipeline
# ==========================================================

def run_training_pipeline():

    print("\n==================================================")
    print("Starting Machine Learning Training Pipeline")
    print("==================================================")

    results = []
    models = {}

    # ------------------------------------------------------
    # Logistic Regression
    # ------------------------------------------------------

    print("\nTraining Logistic Regression...\n")

    model, metrics = train_logistic_regression()

    metrics["model_name"] = "logistic_regression"

    results.append(metrics)

    models["logistic_regression"] = model


    # ------------------------------------------------------
    # Decision Tree
    # ------------------------------------------------------

    print("\nTraining Decision Tree...\n")

    model, metrics = train_decision_tree()

    metrics["model_name"] = "decision_tree"

    results.append(metrics)

    models["decision_tree"] = model


    # ------------------------------------------------------
    # Random Forest
    # ------------------------------------------------------

    print("\nTraining Random Forest...\n")

    model, metrics = train_random_forest()

    metrics["model_name"] = "random_forest"

    results.append(metrics)

    models["random_forest"] = model


    # ------------------------------------------------------
    # Gradient Boosting
    # ------------------------------------------------------

    print("\nTraining Gradient Boosting...\n")

    model, metrics = train_gradient_boosting()

    metrics["model_name"] = "gradient_boosting"

    results.append(metrics)

    models["gradient_boosting"] = model


    # ------------------------------------------------------
    # XGBoost
    # ------------------------------------------------------

    print("\nTraining XGBoost...\n")

    model, metrics = train_xgboost()

    metrics["model_name"] = "xgboost"

    results.append(metrics)

    models["xgboost"] = model


    # ======================================================
    # Model Comparison
    # ======================================================

    print("\n==================================================")
    print("Model Performance Comparison")
    print("==================================================")

    comparison_df = pd.DataFrame(results)

    comparison_df = comparison_df.sort_values(
        by="roc_auc",
        ascending=False
    ).reset_index(drop=True)

    print("\nModel Comparison Table:\n")
    print(comparison_df)


    # ======================================================
    # Save Comparison Table
    # ======================================================

    comparison_df.to_csv(COMPARISON_PATH, index=False)

    print(f"\nModel comparison saved at: {COMPARISON_PATH}")


    # ======================================================
    # Select Best Model
    # ======================================================

    best_model_name = comparison_df.iloc[0]["model_name"]

    print("\n==================================================")
    print(f"Best Model Selected: {best_model_name}")
    print("==================================================")


    best_model = models[best_model_name]


    # ======================================================
    # Save Best Model
    # ======================================================

    print("\nSaving best model...")

    joblib.dump(best_model, BEST_MODEL_PATH)

    print(f"Best model saved at: {BEST_MODEL_PATH}")


    print("\nTraining pipeline completed successfully.")


# ==========================================================
# Run Script
# ==========================================================

if __name__ == "__main__":

    run_training_pipeline()


# ==========================================================
# End of Script
# ==========================================================