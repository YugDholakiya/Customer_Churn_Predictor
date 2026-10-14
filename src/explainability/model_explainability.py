# ==========================================================
# Model Explainability Module
# ==========================================================

"""
This module performs model explainability analysis for the
customer churn prediction system.

Explainability Techniques Used:
1. Feature Importance (Global explanation)
2. SHAP Summary Plot (Global feature impact)

Outputs generated:
- reports/figures/feature_importance.png
- reports/figures/shap_summary_plot.png
"""

# ==========================================================
# Import Libraries
# ==========================================================

import joblib
import pandas as pd
import shap
import matplotlib.pyplot as plt

from pathlib import Path

# ==========================================================
# Detect Project Root Directory
# ==========================================================

"""
Current file location:
src/explainability/model_explainability.py

Project root is two levels up.
"""

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# ==========================================================
# Define Paths Using pathlib
# ==========================================================

MODEL_PATH = PROJECT_ROOT / "saved_models" / "best_churn_prediction_model.pkl"

X_TRAIN_PATH = PROJECT_ROOT / "data" / "model_input" / "x_train.csv"

FIGURE_DIR = PROJECT_ROOT / "reports" / "figures"

# Create figure directory if it doesn't exist
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

# ==========================================================
# Load Model and Training Data
# ==========================================================

print("Loading trained model...")

model = joblib.load(MODEL_PATH)

print("Loading training dataset...")

X_train = pd.read_csv(X_TRAIN_PATH)

print("Model and data loaded successfully.")

# ==========================================================
# Feature Importance
# ==========================================================

print("\nGenerating feature importance plot...")

importances = model.feature_importances_

features = X_train.columns

importance_df = pd.DataFrame({
    "feature": features,
    "importance": importances
})

importance_df = importance_df.sort_values(
    by="importance",
    ascending=False
)

# Plot Top 15 Features
plt.figure(figsize=(10,6))

plt.barh(
    importance_df["feature"][:15],
    importance_df["importance"][:15]
)

plt.gca().invert_yaxis()

plt.title("Top 15 Feature Importances")

plt.xlabel("Importance Score")

plt.tight_layout()

plt.savefig(FIGURE_DIR / "feature_importance.png")

plt.close()

print("Feature importance plot saved.")

# ==========================================================
# SHAP Explainability
# ==========================================================

print("\nGenerating SHAP summary plot...")

explainer = shap.TreeExplainer(model)

shap_values = explainer.shap_values(X_train)

shap.summary_plot(
    shap_values,
    X_train,
    show=False
)

plt.tight_layout()

plt.savefig(FIGURE_DIR / "shap_summary_plot.png")

plt.close()

print("SHAP summary plot saved.")

# ==========================================================
# Completion Message
# ==========================================================

print("\nExplainability analysis completed successfully.")
