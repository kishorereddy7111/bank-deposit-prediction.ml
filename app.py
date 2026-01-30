import streamlit as st
import joblib
import pandas as pd

# Load model and preprocessing objects
model = joblib.load("bank_model.pkl")
scaler = joblib.load("scaler.pkl")
feature_columns = joblib.load("feature_columns.pkl")

st.set_page_config(page_title="Bank Prediction App")
st.title("🏦 Bank Prediction App")
st.write("Predict customer outcome using a trained ML model")

st.sidebar.header("Enter Input Values")

# Collect inputs
input_data = {}
for feature in feature_columns:
    input_data[feature] = st.sidebar.number_input(feature, value=0.0)

# Predict button
if st.sidebar.button("Predict"):
    input_df = pd.DataFrame([input_data])
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)[0]

    st.subheader("Prediction Result")
    if prediction == 1:
        st.success("✅ Positive Outcome")
    else:
        st.error("❌ Negative Outcome")
