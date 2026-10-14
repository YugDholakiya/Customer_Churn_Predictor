from pathlib import Path
import joblib


ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = ROOT / "saved_models" / "best_churn_prediction_model.pkl"
COLS_PATH = ROOT / "saved_models" / "training_columns.pkl"


def load_model():

    model = joblib.load(MODEL_PATH)

    columns = joblib.load(COLS_PATH)

    return model, columns
