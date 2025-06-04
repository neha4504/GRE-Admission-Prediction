import streamlit as st
import pandas as pd
import pickle
import numpy as np
import os

st.title("Graduate Admission Prediction")

# Check for required files
required_files = ['rf_model.pkl', 'xgb_model.pkl', 'scaler.pkl']
for file in required_files:
    if not os.path.exists(file):
        st.error(f"Required file {file} not found. Please ensure all model and scaler files are in the working directory.")
        st.stop()

# Load models and scaler
try:
    rf_model = pickle.load(open('rf_model.pkl', 'rb'))
    xgb_model = pickle.load(open('xgb_model.pkl', 'rb'))
    scaler = pickle.load(open('scaler.pkl', 'rb'))
except FileNotFoundError:
    st.error("Model or scaler file not found. Please upload required files.")
    st.stop()
except Exception as e:
    st.error(f"Error loading models: {str(e)}")
    st.stop()

# Define feature names
feature_names = ['GRE Score', 'TOEFL Score', 'University Rating', 'SOP', 'LOR', 'CGPA', 'Research']

# Input fields
gre = st.number_input("GRE Score (260-340)", 260, 340, value=300)
toefl = st.number_input("TOEFL Score (0-120)", 0, 120, value=100)
univ_rating = st.number_input("University Rating (1-5)", 1, 5, value=3)
sop = st.number_input("SOP Strength (1-5)", 1.0, 5.0, value=3.0)
lor = st.number_input("LOR Strength (1-5)", 1.0, 5.0, value=3.0)
cgpa = st.number_input("CGPA (6-10)", 6.0, 10.0, value=8.0)
research = st.selectbox("Research Experience", [0, 1], index=0)

if st.button("Predict"):
    try:
        user_input = pd.DataFrame([[gre, toefl, univ_rating, sop, lor, cgpa, research]],
                                 columns=feature_names)
        user_scaled = scaler.transform(user_input)

        # Get predictions from Random Forest and XGBoost
        rf_pred = rf_model.predict(user_scaled)[0]
        xgb_pred = xgb_model.predict(user_scaled)[0]

        # Compute mean prediction
        mean_pred = np.mean([rf_pred, xgb_pred])

        # Ensure prediction is within valid range
        mean_pred = np.clip(mean_pred, 0, 1)

        # Display prediction
        st.write(f"Predicted Chance of Admission: {round(mean_pred * 100, 2)}%")
    except Exception as e:
        st.error(f"Error in prediction: {str(e)}")
