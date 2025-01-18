import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler

# Load Models
lstm_model = load_model('lstm_tuned_model_ev.h5')
xgb_model = joblib.load('xgboost_tuned_model_ev.pkl')

# Sidebar Inputs
st.sidebar.title("Input Features")
charging_hour = st.sidebar.slider("Charging Hour", 0, 23, 12)
duration = st.sidebar.slider("Duration (minutes)", 0, 300, 120)
day_of_week = st.sidebar.selectbox("Day of the Week", options=["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
time_of_day = st.sidebar.selectbox("Time of Day", options=["Morning", "Afternoon", "Evening", "Night"])
vehicle_type = st.sidebar.selectbox("Vehicle Type", options=["Sedan", "SUV", "Truck"])

# Convert Inputs to DataFrame
input_data = pd.DataFrame({
    'Charging Hour': [charging_hour],
    'Duration (minutes)': [duration],
    'Day of Week': [day_of_week],
    'Time of Day': [time_of_day],
    'Vehicle Type': [vehicle_type]
})

# Preprocessing Inputs
scaler = StandardScaler()
scaled_input = scaler.fit_transform(input_data)

# Predictions
lstm_prediction = lstm_model.predict(scaled_input.reshape((scaled_input.shape[0], scaled_input.shape[1], 1))).flatten()[0]
xgb_prediction = xgb_model.predict(scaled_input).flatten()[0]

# Display Results
st.title("Energy Consumption Prediction")
st.write("### Input Features")
st.write(input_data)

st.write("### Predictions")
st.write(f"LSTM Prediction: {lstm_prediction:.2f} kWh")
st.write(f"XGBoost Prediction: {xgb_prediction:.2f} kWh")

# Visualize Predictions
st.write("### Comparison of Predictions")
st.bar_chart(pd.DataFrame({
    "Model": ["LSTM", "XGBoost"],
    "Predicted Energy (kWh)": [lstm_prediction, xgb_prediction]
}))
