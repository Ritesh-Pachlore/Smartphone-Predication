import streamlit as st
import pickle
import pandas as pd
import math

# Load the model, scaler, label encoders, and feature columns
with open("models/best_model.pkl", 'rb') as model_file:
    model = pickle.load(model_file)

with open("models/scaler.pkl", 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

with open("models/label_encoders.pkl", 'rb') as encoders_file:
    label_encoders = pickle.load(encoders_file)

with open("models/feature_columns.pkl", 'rb') as feature_columns_file:
    feature_columns = pickle.load(feature_columns_file)

# Streamlit UI
st.title("Smartphone Price Prediction")
st.markdown("Estimate smartphone prices based on features")

# Optional: Upload the CSV dataset if necessary for reference
st.markdown("### Optional: Upload 'data1.csv' (if needed for verification)")
uploaded_file = st.file_uploader("Upload data1.csv", type="csv")
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("File uploaded successfully.")

# Input fields
processor = st.selectbox('Select Processor', label_encoders['Processor'].classes_)
camera_details = st.selectbox('Select Camera Details', label_encoders['Camera_details'].classes_)
storage_details = st.selectbox('Select Storage', label_encoders['Storage_details'].classes_)
screen_size = st.selectbox('Select Screen Size', label_encoders['Screen_size'].classes_)
battery_details = st.selectbox('Select Battery', label_encoders['Battery_details'].classes_)

# Function to encode input data
def encode_input(data, column):
    return label_encoders[column].transform([data])[0]

# Encode inputs
input_data = pd.DataFrame({
    'Processor': [encode_input(processor, 'Processor')],
    'Camera_details': [encode_input(camera_details, 'Camera_details')],
    'Storage_details': [encode_input(storage_details, 'Storage_details')],
    'Screen_size': [encode_input(screen_size, 'Screen_size')],
    'Battery_details': [encode_input(battery_details, 'Battery_details')]
})

# Reorder columns and scale the data
input_data = input_data.reindex(columns=feature_columns)
scaled_input = scaler.transform(input_data)

# Prediction
if st.button("Predict Price"):
    predicted_price = model.predict(scaled_input)[0]
    st.write(f"Predicted Price: ₹{predicted_price:.2f}")
