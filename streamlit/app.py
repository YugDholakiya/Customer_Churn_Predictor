import streamlit as st
import plotly.graph_objects as go
import shap
import pandas as pd
import numpy as np

from utils.model_loader import load_model
from utils.preprocessing import preprocess_input
from utils.prediction import predict
from components.prediction_ui import sidebar_inputs


# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(
    page_title="Customer Churn Prediction",
    layout="wide"
)

st.title("📊 Customer Churn Prediction Dashboard")

st.markdown(
"""
Predict telecom customer churn using a trained machine learning model.

Use the sidebar to enter customer attributes and click **Predict**.
"""
)

# ===============================
# LOAD MODEL
# ===============================
model, training_columns = load_model()

# ===============================
# USER INPUT
# ===============================
user_input = sidebar_inputs()

# ===============================
# PREDICTION BUTTON
# ===============================
if st.button("Predict Churn"):

    processed = preprocess_input(user_input, training_columns)

    prediction, probability, risk = predict(model, processed)

    col1, col2 = st.columns(2)

    # ===============================
    # GAUGE CHART
    # ===============================
    with col1:

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=probability * 100,
            title={'text': "Churn Probability"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "red"},
                'steps': [
                    {'range': [0, 40], 'color': "green"},
                    {'range': [40, 70], 'color': "orange"},
                    {'range': [70, 100], 'color': "red"}
                ]
            }
        ))

        st.plotly_chart(fig, use_container_width=True)

    # ===============================
    # PREDICTION RESULT
    # ===============================
    with col2:

        st.subheader("Prediction Result")

        if prediction == 1:
            st.error("⚠️ Customer likely to churn")
        else:
            st.success("✅ Customer likely to stay")

        st.metric("Risk Category", risk)
        st.metric("Probability", f"{probability:.2%}")

    # ===============================
    # SHAP EXPLANATION
    # ===============================
    st.subheader("Model Explanation")

    explainer = shap.TreeExplainer(model)

    shap_values = explainer.shap_values(processed)

    shap_array = np.array(shap_values)

    # Handle 3D case (samples, features, classes)
    if shap_array.ndim == 3:
        shap_array = shap_array[:, :, 1]

    # Handle list case
    if isinstance(shap_values, list):
        shap_array = shap_values[1]

    shap_df = pd.DataFrame(
        shap_array,
        columns=processed.columns
    )

    st.bar_chart(
        shap_df.abs().T.sort_values(0, ascending=False).head(10)
    )

# ===============================
# FOOTER
# ===============================
st.markdown("---")

st.markdown(
"""
Developed by **Rohit Raut**

GitHub: https://github.com/rohit-1024/customer-churn-prediction-system  
LinkedIn: https://www.linkedin.com/in/rohitraut1024/

© 2026 Customer Churn Prediction System
"""
)