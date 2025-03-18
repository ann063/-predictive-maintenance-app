import streamlit as st
import pandas as pd
import random
import joblib
import numpy as np

# Load dataset
def load_data():
    df = pd.read_csv("predictive_maintenance.csv")
    return df

# Load model and encoders
xgb_model = joblib.load("xgb_model.pkl")
le_type = joblib.load("le_type.pkl")
le_failure = joblib.load("le_failure.pkl")

# Streamlit UI setup
st.title("CAPSTONE PROJECT")
st.subheader("PREDICTIVE MAINTENANCE FOR INDUSTRIAL APPLICATIONS")
st.write("**ANCY SHARMILA D**  ")
st.write("**20MIS0211**  ")

st.markdown("### About this App")
st.write(
    "This application helps predict failures in industrial machinery based on sensor data. "
    "It provides insights into potential failures before they occur, enabling preventive maintenance.")

# Load data
df = load_data()

# Display random readings
st.markdown("### Random Sample Readings")
sample = df.sample(n=1, random_state=random.randint(0, 100))
st.dataframe(sample)

# Prepare features for prediction
sample_features = sample.drop(columns=["UDI", "Product ID", "Failure Type", "Target"])
sample_features["Type"] = le_type.transform(sample_features["Type"])

# Make prediction
prediction = xgb_model.predict(sample_features)
predicted_failure = le_failure.inverse_transform(prediction)[0]
probs = xgb_model.predict_proba(sample_features)[0]
predicted_prob = max(probs) * 100

# Display prediction results
st.markdown("### Failure Prediction")
st.write(f"**Predicted Failure Type:** {predicted_failure}")
st.write(f"**Probability:** {predicted_prob:.2f}%")

# Manual Input for Prediction
st.markdown("### Manual Input for Prediction")
types = list(le_type.classes_)
selected_type = st.selectbox("Select Machine Type", types)
type_encoded = le_type.transform([selected_type])[0]

temperature = st.number_input("Enter Air Temperature (K)", min_value=200.0, max_value=400.0, value=300.0)
process_temp = st.number_input("Enter Process Temperature (K)", min_value=200.0, max_value=500.0, value=310.0)
rotational_speed = st.number_input("Enter Rotational Speed (rpm)", min_value=100.0, max_value=3000.0, value=1500.0)
torque = st.number_input("Enter Torque (Nm)", min_value=0.0, max_value=100.0, value=50.0)
tool_wear = st.number_input("Enter Tool Wear (min)", min_value=0.0, max_value=300.0, value=150.0)

if st.button("Predict Failure Type"):
    input_data = np.array([[type_encoded, temperature, process_temp, rotational_speed, torque, tool_wear]])
    manual_prediction = xgb_model.predict(input_data)
    manual_predicted_failure = le_failure.inverse_transform(manual_prediction)[0]
    manual_probs = xgb_model.predict_proba(input_data)[0]
    manual_predicted_prob = max(manual_probs) * 100
    
    st.write(f"**Predicted Failure Type:** {manual_predicted_failure}")
    st.write(f"**Probability:** {manual_predicted_prob:.2f}%")

