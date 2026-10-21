import streamlit as st
import plotly.graph_objects as go
import shap
import pandas as pd
import numpy as np

from utils.model_loader import load_model
from utils.preprocessing import preprocess_input
from utils.prediction import predict, generate_recommendations
from components.prediction_ui import sidebar_inputs


# ===============================
# PAGE CONFIGURATION
# ===============================
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📊",
    layout="wide"
)

# Custom header styling
st.title("📊 Customer Churn Predictor")
st.markdown(
    """
    **AI-powered customer churn risk assessment and retention optimization.**  
    Enter the customer's account attributes in the sidebar, then click **Evaluate Customer** to analyze churn probability.
    """
)
st.write("")

# ===============================
# LOAD ARTIFACTS
# ===============================
try:
    model, training_columns = load_model()
except Exception as e:
    st.error(f"Failed to load model artifacts: {e}")
    st.stop()

# ===============================
# USER INPUT SIDEBAR
# ===============================
user_input = sidebar_inputs()

# ===============================
# PREDICTION ACTION
# ===============================
col_btn, _ = st.columns([1, 4])
with col_btn:
    run_prediction = st.button("🔍 Evaluate Customer", use_container_width=True)

if run_prediction:
    processed_df = preprocess_input(user_input, training_columns)
    prediction, probability, risk_level = predict(model, processed_df)

    # Overview Metrics Row
    st.markdown("### 📈 Risk Assessment Overview")
    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)

    with metric_col1:
        st.metric(
            label="Prediction Outcome",
            value="Likely to Churn" if prediction == 1 else "Likely to Stay",
            delta="High Risk" if prediction == 1 else "Retained",
            delta_color="inverse" if prediction == 1 else "normal"
        )

    with metric_col2:
        st.metric(
            label="Churn Probability",
            value=f"{probability * 100:.1f}%"
        )

    with metric_col3:
        st.metric(
            label="Risk Classification",
            value=risk_level
        )

    with metric_col4:
        st.metric(
            label="Tenure / Commitment",
            value=f"{user_input['tenure']} mos ({user_input['contract']})"
        )

    st.write("")
    viz_col1, viz_col2 = st.columns([1, 1])

    # Gauge Chart
    with viz_col1:
        st.subheader("🎯 Probability Gauge")
        gauge_color = "#e63946" if probability >= 0.70 else ("#f4a261" if probability >= 0.40 else "#2a9d8f")

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=probability * 100,
            number={'suffix': "%", 'font': {'size': 36}},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1},
                'bar': {'color': gauge_color},
                'steps': [
                    {'range': [0, 40], 'color': "rgba(42, 157, 143, 0.2)"},
                    {'range': [40, 70], 'color': "rgba(244, 162, 97, 0.2)"},
                    {'range': [70, 100], 'color': "rgba(230, 57, 70, 0.2)"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 3},
                    'thickness': 0.75,
                    'value': 70
                }
            }
        ))
        fig.update_layout(height=280, margin=dict(l=20, r=20, t=20, b=20))
        st.plotly_chart(fig, use_container_width=True)

    # Key Drivers & Advice
    with viz_col2:
        st.subheader("💡 Retention Recommendations")
        recommendations = generate_recommendations(user_input, probability)
        for rec in recommendations:
            st.info(f"• {rec}")

    # Model Explainability (SHAP)
    st.subheader("🔍 Prediction Drivers (SHAP Feature Impact)")
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(processed_df)
        shap_arr = np.array(shap_values)

        if shap_arr.ndim == 3:
            shap_arr = shap_arr[:, :, 1]
        elif isinstance(shap_values, list):
            shap_arr = shap_values[1]

        shap_df = pd.DataFrame(shap_arr, columns=processed_df.columns)
        top_features = shap_df.abs().T.sort_values(by=0, ascending=False).head(10)
        top_features.columns = ["Importance Weight"]

        st.bar_chart(top_features)
    except Exception as ex:
        st.warning(f"Could not calculate feature impact visualization: {ex}")
