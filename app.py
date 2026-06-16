"""
FRONTEND / GUI  (Multi-page version)
-------------------------------------
This is the website. It has a sidebar with 3 pages, like a real dashboard:

1. Overview & EDA          -> key stats + charts about the dataset
2. Model Comparison        -> accuracy table for all 5 ML models tested
3. Live Prediction         -> type in customer details, get a prediction

This file does NOT train anything. It only loads model.pkl, which was
created by running backend.py.

HOW TO RUN:
    streamlit run app.py
"""

import streamlit as st
import pandas as pd
import pickle

# =========================
# Page Setup
# =========================
st.set_page_config(
    page_title="Customer Churn Prediction App",
    page_icon="📊",
    layout="wide",
)

# =========================
# Load the trained model bundle
# =========================
@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as f:
        bundle = pickle.load(f)
    return bundle

bundle = load_model()
model = bundle["model"]
best_model_name = bundle["best_model_name"]
scaler = bundle["scaler"]
encoders = bundle["encoders"]
feature_columns = bundle["feature_columns"]
results_df = bundle["results_df"]
eda_stats = bundle["eda_stats"]
segmentation_data = bundle["segmentation_data"]
segment_profile = bundle["segment_profile"]

# =========================
# Sidebar Navigation
# =========================
st.sidebar.title("📊 Telecom Churn Analytics")
st.sidebar.caption("Customer Churn Prediction System")
st.sidebar.divider()

page = st.sidebar.radio(
    "Navigate",
    ["🏠 Overview & EDA", "📈 Model Comparison", "🔮 Live Prediction", "👥 Customer Segmentation"],
)

st.sidebar.divider()
st.sidebar.write(f"**Best Model:** {best_model_name}")
st.sidebar.write(f"**Dataset size:** {eda_stats['total_customers']:,} customers")

# =========================
# PAGE 1: Overview & EDA
# =========================
if page == "🏠 Overview & EDA":
    st.title("📊 Customer Churn — Overview & Exploratory Data Analysis")
    st.write(
        "A telecom company wants to understand which customers are likely "
        "to leave (churn) so it can take action early. This dashboard "
        "explores the Telco Customer Churn dataset, compares Machine "
        "Learning models, and predicts churn for new customers."
    )
    st.divider()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Customers", f"{eda_stats['total_customers']:,}")
    col2.metric("Churn Rate", f"{eda_stats['churn_rate']:.1f}%")
    col3.metric("Avg. Tenure (months)", f"{eda_stats['avg_tenure']:.1f}")
    col4.metric("Avg. Monthly Charges", f"${eda_stats['avg_monthly_charges']:.0f}")

    st.divider()

    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        st.subheader("Churn Distribution")
        churn_df = pd.DataFrame(
            {
                "Status": list(eda_stats["churn_counts"].keys()),
                "Customers": list(eda_stats["churn_counts"].values()),
            }
        )
        st.bar_chart(churn_df.set_index("Status"), color="#FF4B4B")

    with chart_col2:
        st.subheader("Contract Length vs Churn")
        contract_df = pd.DataFrame(eda_stats["contract_churn"])
        st.bar_chart(contract_df)

    st.info(
        "💡 Insight: Customers on **Month-to-month** contracts churn far "
        "more often than customers on One year or Two year contracts."
    )

# =========================
# PAGE 2: Model Comparison
# =========================
elif page == "📈 Model Comparison":
    st.title("📈 Machine Learning Model Comparison")
    st.write(
        "Five different classification models were trained and tested on "
        "the same data. Their performance is compared below."
    )
    st.divider()

    display_df = results_df.copy()
    for col in ["Accuracy", "Precision", "Recall", "F1 Score"]:
        display_df[col] = (display_df[col] * 100).round(2).astype(str) + "%"

    st.dataframe(display_df, use_container_width=True, hide_index=True)

    st.subheader("Accuracy Comparison")
    chart_df = results_df.set_index("Model")[["Accuracy"]]
    st.bar_chart(chart_df)

    st.success(
        f"✅ **{best_model_name}** performed best overall and is the model "
        f"used on the Live Prediction page."
    )

# =========================
# PAGE 3: Live Prediction
# =========================
elif page == "🔮 Live Prediction":
    st.title("🔮 Predict Churn for a Customer")
    st.write(
        "Enter a customer's details below and click **Predict Churn** to "
        f"see the result using the **{best_model_name}** model."
    )
    st.divider()

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

    if st.button("🔍 Predict Churn", use_container_width=True):
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

        for col, le in encoders.items():
            if col == "Churn":
                continue
            if col in input_df.columns:
                input_df[col] = le.transform(input_df[col].astype(str))

        input_df = input_df[feature_columns]
        input_scaled = scaler.transform(input_df)

        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0][1]

        st.subheader("Result")

        if prediction == 1:
            st.error("⚠️ This customer is LIKELY TO CHURN (leave).")
        else:
            st.success("✅ This customer is LIKELY TO STAY.")

        st.metric("Churn Probability", f"{probability * 100:.1f}%")
        st.progress(min(max(probability, 0.0), 1.0))

# =========================
# PAGE 4: Customer Segmentation
# =========================
elif page == "👥 Customer Segmentation":
    st.title("👥 Customer Segmentation")
    st.write(
        "Customers were grouped into 3 segments using **K-Means Clustering**, "
        "based on their tenure, monthly charges, and total charges. This helps "
        "a business understand different types of customers and target them "
        "differently (e.g. high-spend new customers vs long-term low-spend ones)."
    )
    st.divider()

    st.subheader("Segment Profile")
    st.dataframe(segment_profile, use_container_width=True)

    st.divider()

    st.subheader("Segment Visualization (2D projection)")
    st.write(
        "Each point is a customer. Customers in the same segment behave "
        "similarly. The chart below is a simplified 2D view of all the "
        "customer data using PCA (Principal Component Analysis)."
    )

    chart_data = segmentation_data.rename(columns={"PCA1": "x", "PCA2": "y"})
    st.scatter_chart(chart_data, x="x", y="y", color="Segment", size=20)

    st.divider()

    highest_churn_segment = segment_profile["Churn Rate (%)"].idxmax()
    highest_churn_rate = segment_profile["Churn Rate (%)"].max()

    st.warning(
        f"⚠️ Insight: **Segment {highest_churn_segment}** has the highest "
        f"churn rate at **{highest_churn_rate}%** — likely newer customers "
        f"paying higher monthly charges. This group should be prioritized "
        f"for retention offers."
    )
