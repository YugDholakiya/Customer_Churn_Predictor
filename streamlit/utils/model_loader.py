from pathlib import Path
import joblib
import streamlit as st


@st.cache_resource(show_spinner=False)
def load_model():
    """Loads and caches the trained prediction model and feature columns."""
    root_dir = Path(__file__).resolve().parents[1]
    model_path = root_dir / "saved_models" / "best_churn_prediction_model.pkl"
    cols_path = root_dir / "saved_models" / "training_columns.pkl"

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found at: {model_path}")
    if not cols_path.exists():
        raise FileNotFoundError(f"Feature column file not found at: {cols_path}")

    model = joblib.load(model_path)
    columns = joblib.load(cols_path)
    return model, columns
