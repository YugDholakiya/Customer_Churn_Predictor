# ==========================================================
# Model Evaluation Module
# ==========================================================

"""
Purpose
-------
Evaluate machine learning classification models using
multiple performance metrics and generate visual evaluation plots.

Metrics Calculated
------------------
- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC Score
- Confusion Matrix

Plots Generated
---------------
- Confusion Matrix Heatmap
- ROC Curve

All generated plots are automatically saved to:

    reports/figures/

This allows easy inclusion of visualizations in
project documentation and GitHub README files.
"""

# ==========================================================
# Import Libraries
# ==========================================================

from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve
)

# ==========================================================
# Define Figure Output Directory
# ==========================================================

FIGURE_DIR = Path("reports/figures")

# Create directory if it does not exist
FIGURE_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================================
# Model Evaluation Function
# ==========================================================

def evaluate_model(model_name, y_test, y_pred, y_pred_proba):
    """
    Evaluate a trained machine learning model.

    Parameters
    ----------
    model_name : str
        Name of the machine learning model.

    y_test : array-like
        True target labels.

    y_pred : array-like
        Predicted class labels.

    y_pred_proba : array-like
        Predicted probabilities for the positive class.

    Returns
    -------
    dict: Dictionary containing evaluation/performance metrics.
    """

    print("\n==================================================")
    print(f"Evaluating Model : {model_name}")
    print("==================================================")

    # ------------------------------------------------------
    # Calculate Evaluation Metrics
    # ------------------------------------------------------

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)

    print(f"\nAccuracy  : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1 Score  : {f1:.4f}")
    print(f"ROC-AUC   : {roc_auc:.4f}")

    # ------------------------------------------------------
    # Confusion Matrix Plot
    # ------------------------------------------------------

    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(6, 4))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["No Churn", "Churn"],
        yticklabels=["No Churn", "Churn"]
    )

    plt.title(f"{model_name} - Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("Actual Label")

    plt.tight_layout()

    cm_path = FIGURE_DIR / f"{model_name}_confusion_matrix.png"
    plt.savefig(cm_path)

    plt.close()

    print(f"\nConfusion matrix saved to: {cm_path}")

    # ------------------------------------------------------
    # ROC Curve Plot
    # ------------------------------------------------------

    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)

    plt.figure(figsize=(6, 4))

    plt.plot(fpr, tpr, label=f"{model_name} (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], linestyle="--")

    plt.xlabel("False Positive Rate (FPR)")
    plt.ylabel("True Positive Rate (TPR)")
    plt.title(f"{model_name} - ROC Curve")

    plt.legend()

    plt.tight_layout()

    roc_path = FIGURE_DIR / f"{model_name}_roc_curve.png"
    plt.savefig(roc_path)

    plt.close()

    print(f"\nROC curve saved to: {roc_path}")

    # ------------------------------------------------------
    # Return Evaluation Metrics
    # ------------------------------------------------------

    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "roc_auc": roc_auc
    }
    print("\nEvaluation Metrics Returned Successfully.")
    return metrics


if  __name__ == '__main__':
    print("The script 'src/models_evaluation/evaluate_models.py' executed successfully.")


# ==========================================================
# End of 'evaluate_models.py' Script.
# ==========================================================