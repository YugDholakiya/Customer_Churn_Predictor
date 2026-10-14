def predict(model, df):

    prediction = model.predict(df)[0]

    probability = model.predict_proba(df)[0][1]

    if probability >= 0.7:
        risk = "High Risk"

    elif probability >= 0.4:
        risk = "Medium Risk"

    else:
        risk = "Low Risk"

    return prediction, probability, risk
