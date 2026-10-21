import pandas as pd


def predict(model, df: pd.DataFrame):
    """Generates churn prediction, calibrated probability, and risk level."""
    prediction = int(model.predict(df)[0])
    probability = float(model.predict_proba(df)[0][1])

    if probability >= 0.70:
        risk_level = "High Risk"
    elif probability >= 0.40:
        risk_level = "Medium Risk"
    else:
        risk_level = "Low Risk"

    return prediction, probability, risk_level


def generate_recommendations(user_input: dict, probability: float) -> list:
    """Generates tailored retention actions based on customer risk and attributes."""
    recommendations = []

    if user_input.get("contract") == "Month-to-month":
        recommendations.append("Offer a discounted annual or two-year contract commitment.")

    if user_input.get("payment_method") == "Electronic check":
        recommendations.append("Encourage automated bank transfer or credit card autopay with a promotional credit.")

    if user_input.get("tech_support") == "No":
        recommendations.append("Include complimentary premium Tech Support for 3 months.")

    if user_input.get("online_security") == "No":
        recommendations.append("Bundle Online Security and Cloud Backup at a reduced rate.")

    if float(user_input.get("monthly_charges", 0)) > 80:
        recommendations.append("Review billing package and propose a tailored cost-saving bundle.")

    if not recommendations:
        recommendations.append("Maintain high service quality and consider enrollment in loyalty rewards.")

    return recommendations
