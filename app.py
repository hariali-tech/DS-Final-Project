import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Telco Churn Predictor", page_icon="📊")
st.title("📊 Telco Customer Churn Predictor")
st.write("Enter customer information to predict whether the customer is likely to churn.")

@st.cache_resource
def load_model():
    return joblib.load("telco_churn_model.joblib")

try:
    model = load_model()
except FileNotFoundError:
    st.error("Put telco_churn_model.joblib in the same folder as app.py.")
    st.stop()

with st.form("customer_form"):
    gender = st.selectbox("Gender", ["Male", "Female"])
    senior = st.selectbox("Senior Citizen", [0, 1])
    partner = st.selectbox("Partner", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["Yes", "No"])
    tenure = st.number_input("Tenure (months)", min_value=0, max_value=72, value=12)

    phone = st.selectbox("Phone Service", ["Yes", "No"])
    lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
    internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
    backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
    protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
    support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
    tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
    movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])

    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
    payment = st.selectbox("Payment Method", [
        "Electronic check", "Mailed check",
        "Bank transfer (automatic)", "Credit card (automatic)"
    ])

    monthly = st.number_input("Monthly Charges", min_value=0.0, value=70.0)
    total = st.number_input("Total Charges", min_value=0.0, value=840.0)

    submitted = st.form_submit_button("Predict Churn")

if submitted:
    row = pd.DataFrame([{
        "customerID": "DEMO-001",
        "gender": gender,
        "SeniorCitizen": senior,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone,
        "MultipleLines": lines,
        "InternetService": internet,
        "OnlineSecurity": security,
        "OnlineBackup": backup,
        "DeviceProtection": protection,
        "TechSupport": support,
        "StreamingTV": tv,
        "StreamingMovies": movies,
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment,
        "MonthlyCharges": monthly,
        "TotalCharges": total
    }])

    # Same feature engineering used by the notebook
    row["AvgMonthlySpend"] = row["TotalCharges"] / row["tenure"].replace(0, np.nan)

    service_cols = [
        "PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup",
        "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"
    ]
    row["ServiceCount"] = (row[service_cols] == "Yes").sum(axis=1)

    prediction = model.predict(row)[0]
    probability = model.predict_proba(row)[0, 1]

    if prediction == 1:
        st.error("Prediction: Customer is likely to CHURN")
    else:
        st.success("Prediction: Customer is likely to STAY")

    st.metric("Churn Probability", f"{probability:.2%}")
    st.caption("This is a machine-learning prediction, not a guarantee.")
