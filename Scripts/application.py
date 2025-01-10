import streamlit as st
import joblib
import numpy as np
import pandas as pd


# Load the saved model
model = joblib.load('bankruptcy_prediction_model.pkl')
df = pd.read_csv("data.csv").drop("Bankrupt?", axis =1)

# Define the app title
st.title("Bankruptcy Risk Prediction")

# Sidebar inputs
st.sidebar.header("Input Financial Metrics")
feature_names = df.columns  # Replace with the actual feature names
inputs = {}

for feature in feature_names:
    inputs[feature] = st.sidebar.number_input(f"Enter {feature}:", value=0.0)

# Convert inputs to numpy array
input_values = np.array([inputs[feature] for feature in feature_names]).reshape(1, -1)

# Predict bankruptcy risk
if st.sidebar.button("Predict"):
    prediction_prob = model.predict_proba(input_values)[0]
    prediction_class = model.predict(input_values)[0]

    st.subheader("Prediction Results")
    st.write(f"Predicted Class: {'Bankrupt' if prediction_class == 1 else 'Non-Bankrupt'}")
    st.write(f"Probability of Bankruptcy: {prediction_prob[1]:.2f}")
    st.write(f"Probability of Non-Bankruptcy: {prediction_prob[0]:.2f}")

# Display feature importance
if hasattr(model, 'feature_importances_'):
    st.subheader("Feature Importance")
    feature_importance = model.feature_importances_
    importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importance})
    st.bar_chart(importance_df.set_index('Feature'))
