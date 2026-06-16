"""
FRONTEND / GUI
--------------
This file creates the WEBSITE (using Streamlit) where a user can:
1. Enter customer details using simple dropdowns and sliders
2. Click a button
3. See whether the customer will CHURN or NOT, with a probability score

This file does NOT train any model. It only LOADS the model.pkl file
that backend.py already created. This is exactly how real projects work:
- backend.py  = trains and saves the "brain"
- app.py      = the website that uses the "brain" to answer questions

HOW TO RUN THIS FILE:
    streamlit run app.py

This will open a browser window with your website.
"""

import streamlit as st
import pandas as pd
import pickle

# =========================
# Page Setup
# =========================
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📊",
    layout="centered",
)

# =========================
# Load the trained model bundle (model + scaler + encoders)
# =========================
@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as f:
        bundle = pickle.load(f)
    return bundle

bundle = load_model()
model = bundle["model"]
scaler = bundle["scaler"]
encoders = bundle["encoders"]
feature_columns = bundle["feature_columns"]

# =========================
# Title
# =========================
st.title("📊 Customer Churn Prediction App")
st.write(
    "This tool predicts whether a telecom customer is likely to "
    "**Churn (leave)** or **Stay**, based on their account details."
)

st.divider()

# =========================
# Sidebar - About section
# =========================
with st.sidebar:
    st.header("ℹ️ About this project")
    st.write(
        "This app uses a Machine Learning model (Random Forest) "
        "trained on the Telco Customer Churn dataset to predict "
        "customer churn."
    )
    st.write("Built with: Python, scikit-learn, Streamlit")

# =========================
# Input Form
# =========================
st.subheader("Enter Customer Details")

col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["Female", "Male"])
    senior = st.selectbox("Senior Citizen", ["No", "Yes"])
    partner = st.selectbox("Has Partner", ["No", "Yes"])
    dependents = st.selectbox("Has Dependents", ["No", "Yes"])
    tenure = st.slider("Tenure (months)", 0, 72, 12)
    phone_service = st.selectbox("Phone Service", ["No", "Yes"])
    multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
    internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
    online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])

with col2:
    device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
    tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
    streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
    streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    paperless_billing = st.selectbox("Paperless Billing", ["No", "Yes"])
    payment_method = st.selectbox(
        "Payment Method",
        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
    )
    monthly_charges = st.number_input("Monthly Charges ($)", 0.0, 200.0, 70.0)
    total_charges = st.number_input("Total Charges ($)", 0.0, 10000.0, 1000.0)

st.divider()

# =========================
# Predict Button
# =========================
if st.button("🔍 Predict Churn", use_container_width=True):

    # 1. Put all inputs into a single-row dataframe with the SAME
    #    column names used during training.
    input_dict = {
        "gender": gender,
        "SeniorCitizen": 1 if senior == "Yes" else 0,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
    }

    input_df = pd.DataFrame([input_dict])

    # 2. Encode text columns using the SAME encoders used in training
    for col, le in encoders.items():
        if col == "Churn":
            continue  # Churn is the target, not an input
        if col in input_df.columns:
            input_df[col] = le.transform(input_df[col].astype(str))

    # 3. Make sure column order matches training order exactly
    input_df = input_df[feature_columns]

    # 4. Scale the input the SAME way training data was scaled
    input_scaled = scaler.transform(input_df)

    # 5. Predict
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]  # probability of churn

    st.subheader("Result")

    if prediction == 1:
        st.error(f"⚠️ This customer is LIKELY TO CHURN (leave).")
    else:
        st.success(f"✅ This customer is LIKELY TO STAY.")

    st.metric("Churn Probability", f"{probability * 100:.1f}%")
    st.progress(min(max(probability, 0.0), 1.0))
