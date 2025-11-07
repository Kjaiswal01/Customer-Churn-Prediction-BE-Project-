# app/streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
from textblob import TextBlob
import joblib
import os

# ---------------------------
# 0. Config & paths
# ---------------------------
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))  # Project root
MODELS_DIR = os.path.join(BASE_DIR, "models")

FEATURE_LIST_FILE = os.path.join(MODELS_DIR, "feature_list.pkl")
SCALER_FILE = os.path.join(MODELS_DIR, "scaler.pkl")
LABEL_ENCODERS_FILE = os.path.join(MODELS_DIR, "label_encoders.pkl")
MODEL_FILE = os.path.join(MODELS_DIR, "churn_model.pkl")

# ---------------------------
# 1. Load model and artifacts
# ---------------------------
@st.cache_resource
def load_artifacts():
    if not all(os.path.exists(f) for f in [MODEL_FILE, SCALER_FILE, FEATURE_LIST_FILE, LABEL_ENCODERS_FILE]):
        st.error("One or more model artifact files are missing in the 'models' folder!")
        st.stop()
    model = joblib.load(MODEL_FILE)
    scaler = joblib.load(SCALER_FILE)
    feature_cols = joblib.load(FEATURE_LIST_FILE)
    label_encoders = joblib.load(LABEL_ENCODERS_FILE)
    return model, scaler, feature_cols, label_encoders

model, scaler, feature_cols, label_encoders = load_artifacts()

# ---------------------------
# 2. Helper functions
# ---------------------------
def get_sentiment(feedback):
    try:
        return round(TextBlob(feedback).sentiment.polarity, 3)
    except:
        return 0.0

def encode_input(value, col_name):
    le = label_encoders.get(col_name)
    if le:
        return le.transform([value])[0]
    return value

# ---------------------------
# 3. Streamlit app UI
# ---------------------------
st.title("Customer Churn Prediction App")
st.markdown("Predict if a customer is likely to churn based on their details and feedback.")

# User input form
with st.form(key="churn_form"):
    gender = st.selectbox("Gender", ["Male", "Female"])
    age = st.number_input("Age", min_value=18, max_value=100, value=30)
    tenure = st.number_input("Tenure (months)", min_value=1, max_value=48, value=12)
    subscription = st.selectbox("Subscription Type", ["Basic", "Standard", "Premium"])
    monthly_charges = st.number_input("Monthly Charges", min_value=100, max_value=2000, value=499)
    total_spend = monthly_charges * tenure
    last_interaction_days = st.number_input("Days Since Last Interaction", min_value=0, max_value=365, value=15)
    feedback = st.text_area("Customer Feedback", "Very satisfied with subscription")
    
    submit_btn = st.form_submit_button("Predict Churn")

# ---------------------------
# 4. Prediction
# ---------------------------
if submit_btn:
    # Prepare input
    input_dict = {
        "Gender": encode_input(gender, "Gender"),
        "Age": age,
        "Tenure": tenure,
        "Subscription_Type": encode_input(subscription, "Subscription_Type"),
        "Monthly_Charges": monthly_charges,
        "Total_Spend": total_spend,
        "Last_Interaction_Days": last_interaction_days,
        "Sentiment": get_sentiment(feedback)
    }
    
    input_df = pd.DataFrame([input_dict])
    input_scaled = scaler.transform(input_df[feature_cols])
    
    # Predict
    pred = model.predict(input_scaled)[0]
    prob = model.predict_proba(input_scaled)[0,1]
    
    st.markdown(f"**Prediction:** {'Churn' if pred==1 else 'Stay'}")
    st.markdown(f"**Probability of Churn:** {prob:.2f}")
