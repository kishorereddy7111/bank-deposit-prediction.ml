import streamlit as st
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load saved objects
model = joblib.load("bank_model.pkl")
scaler = joblib.load("scaler.pkl")
feature_columns = joblib.load("feature_columns.pkl")

st.set_page_config(page_title="Bank Deposit Dashboard", layout="wide")

st.title("🏦 Bank Deposit Prediction Dashboard")
st.markdown("Interactive ML dashboard with prediction insights")

st.sidebar.header("📥 Enter Customer Details")

# Collect inputs
input_data = {}
for feature in feature_columns:
    input_data[feature] = st.sidebar.number_input(feature, value=0.0)

if st.sidebar.button("🔍 Predict"):

    input_df = pd.DataFrame([input_data])
    input_scaled = scaler.transform(input_df)

    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    # --------------------------
    # Layout Columns
    # --------------------------
    col1, col2 = st.columns(2)

    # Result Section
    with col1:
        st.subheader("Prediction Result")

        if prediction == 1:
            st.success("✅ Customer is likely to Subscribe")
        else:
            st.error("❌ Customer is NOT likely to Subscribe")

        st.metric("Confidence Score", f"{probability*100:.2f}%")
        st.progress(float(probability))

    # Probability Bar Chart
    with col2:
        st.subheader("Prediction Probability Distribution")

        prob_df = pd.DataFrame({
            "Outcome": ["Not Subscribe", "Subscribe"],
            "Probability": [1-probability, probability]
        })

        st.bar_chart(prob_df.set_index("Outcome"))

    # --------------------------
    # Feature Importance Graph
    # --------------------------
    if hasattr(model, "feature_importances_"):
        st.subheader("📊 Feature Importance")

        importance_df = pd.DataFrame({
            "Feature": feature_columns,
            "Importance": model.feature_importances_
        }).sort_values(by="Importance", ascending=False)

        st.bar_chart(importance_df.set_index("Feature"))
