# bank-deposit-prediction.ml
Predicts bank customer deposit subscription using machine learning and Streamlit.

🏦 Bank Deposit Prediction using Machine Learning

 **Live Application:**  
 https://bank-deposit-predictionml-uwedxpkarg3u6bskutbuzz.streamlit.app/

This project predicts whether a customer will subscribe to a bank term deposit using Machine Learning. The model is deployed using Streamlit Cloud.

1. Project Title

Bank Deposit Subscription Prediction using Machine Learning

📌 2. Problem Statement

Banks need to predict whether a customer will subscribe to a term deposit to optimize marketing campaigns and reduce cost.

📌 3. Business Objective

Identify high-probability customers

Reduce marketing expenses

Improve campaign conversion rate

📌 4. Dataset Description

Source: Bank Marketing Dataset

Features: Age, Job, Marital, Education, Balance, Contact, Duration

Target: Deposit Subscription (Yes/No)

📌 5. Project Workflow
Step 1 – Data Collection

Loaded bank.csv dataset

Step 2 – Data Cleaning

Handled missing values

Encoded categorical variables

Feature scaling

Step 3 – Exploratory Data Analysis

Checked class imbalance

Visualized correlations

Distribution plots

Step 4 – Feature Engineering

One-hot encoding

Scaling using StandardScaler

Step 5 – Model Building

Logistic Regression

Random Forest (if used)

Compared accuracy

Step 6 – Model Evaluation

Accuracy

Precision

Recall

F1 Score

Confusion Matrix

Step 7 – Model Deployment

Saved trained model using pickle

Built prediction app using app.py
