import streamlit as st
import pandas as pd
import numpy as np
from joblib import load

# --- Load your single model ---
model = load("model.pkl")

st.title("Customer Churn Prediction App")
st.write("Enter customer details to predict churn probability")

# --- Input fields ---
gender = st.selectbox("Gender", ["Male","Female"])
senior = st.selectbox("Senior Citizen", [0,1])
partner = st.selectbox("Partner", ["Yes","No"])
dependents = st.selectbox("Dependents", ["Yes","No"])
tenure = st.slider("Tenure (months)",0,72)
phoneservice = st.selectbox("Phone Service", ["Yes","No"])
multiplelines = st.selectbox("Multiple Lines", ["Yes","No","No phone service"])
internet = st.selectbox("Internet Service", ["DSL","Fiber optic","No"])
onlinesecurity = st.selectbox("Online Security", ["Yes","No","No internet service"])
onlinebackup = st.selectbox("Online Backup", ["Yes","No","No internet service"])
deviceprotection = st.selectbox("Device Protection", ["Yes","No","No internet service"])
techsupport = st.selectbox("Tech Support", ["Yes","No","No internet service"])
streamingtv = st.selectbox("Streaming TV", ["Yes","No","No internet service"])
streamingmovies = st.selectbox("Streaming Movies", ["Yes","No","No internet service"])
contract = st.selectbox("Contract", ["Month-to-month","One year","Two year"])
paperless = st.selectbox("Paperless Billing", ["Yes","No"])
payment = st.selectbox(
    "Payment Method",
    ["Electronic check","Mailed check","Bank transfer (automatic)","Credit card (automatic)"]
)
monthly = st.number_input("Monthly Charges",0.0,200.0)
total = st.number_input("Total Charges",0.0,10000.0)

# --- Build dataframe ---
data = pd.DataFrame({
    "gender":[gender],
    "SeniorCitizen":[senior],
    "Partner":[partner],
    "Dependents":[dependents],
    "tenure":[tenure],
    "PhoneService":[phoneservice],
    "MultipleLines":[multiplelines],
    "InternetService":[internet],
    "OnlineSecurity":[onlinesecurity],
    "OnlineBackup":[onlinebackup],
    "DeviceProtection":[deviceprotection],
    "TechSupport":[techsupport],
    "StreamingTV":[streamingtv],
    "StreamingMovies":[streamingmovies],
    "Contract":[contract],
    "PaperlessBilling":[paperless],
    "PaymentMethod":[payment],
    "MonthlyCharges":[monthly],
    "TotalCharges":[total]
})

# --- Prediction ---
if st.button("Predict Churn"):
    prob = model.predict(data)[0]
    st.metric("Predicted Value", f"{prob:.2f}")
    st.progress(float(prob))

    if prob > 0.6:
        st.error("High Risk Customer")
    elif prob > 0.3:
        st.warning("Medium Risk Customer")
    else:
        st.success("Low Risk Customer")
