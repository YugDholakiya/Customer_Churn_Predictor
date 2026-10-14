import streamlit as st


def sidebar_inputs():

    st.sidebar.title("📋 Customer Information")

    gender = st.sidebar.selectbox("Gender",["Male","Female"])

    senior_citizen = st.sidebar.selectbox("Senior Citizen",[0,1])

    partner = st.sidebar.selectbox("Partner",["Yes","No"])

    dependents = st.sidebar.selectbox("Dependents",["Yes","No"])

    tenure = st.sidebar.slider("Tenure (months)",0,72,12)

    phone_service = st.sidebar.selectbox("Phone Service",["Yes","No"])

    multiple_lines = st.sidebar.selectbox(
        "Multiple Lines",
        ["Yes","No","No phone service"]
    )

    internet_service = st.sidebar.selectbox(
        "Internet Service",
        ["DSL","Fiber optic","No"]
    )

    online_security = st.sidebar.selectbox(
        "Online Security",
        ["Yes","No","No internet service"]
    )

    online_backup = st.sidebar.selectbox(
        "Online Backup",
        ["Yes","No","No internet service"]
    )

    device_protection = st.sidebar.selectbox(
        "Device Protection",
        ["Yes","No","No internet service"]
    )

    tech_support = st.sidebar.selectbox(
        "Tech Support",
        ["Yes","No","No internet service"]
    )

    streaming_tv = st.sidebar.selectbox(
        "Streaming TV",
        ["Yes","No","No internet service"]
    )

    streaming_movies = st.sidebar.selectbox(
        "Streaming Movies",
        ["Yes","No","No internet service"]
    )

    contract = st.sidebar.selectbox(
        "Contract",
        ["Month-to-month","One year","Two year"]
    )

    paperless_billing = st.sidebar.selectbox(
        "Paperless Billing",
        ["Yes","No"]
    )

    payment_method = st.sidebar.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ]
    )

    monthly_charges = st.sidebar.slider(
        "Monthly Charges",
        10.0,
        150.0,
        70.0
    )

    total_charges = st.sidebar.slider(
        "Total Charges",
        0.0,
        10000.0,
        2000.0
    )

    return {
        "gender":gender,
        "senior_citizen":senior_citizen,
        "partner":partner,
        "dependents":dependents,
        "tenure":tenure,
        "phone_service":phone_service,
        "multiple_lines":multiple_lines,
        "internet_service":internet_service,
        "online_security":online_security,
        "online_backup":online_backup,
        "device_protection":device_protection,
        "tech_support":tech_support,
        "streaming_tv":streaming_tv,
        "streaming_movies":streaming_movies,
        "contract":contract,
        "paperless_billing":paperless_billing,
        "payment_method":payment_method,
        "monthly_charges":monthly_charges,
        "total_charges":total_charges
    }
