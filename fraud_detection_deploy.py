import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load("C:/Users/AKIN-JOHNSON/Desktop/DevSeal/fraud_detection_model.pkl")

# Load the Label Encoder for the Transaction Type column
label_encoder = joblib.load("C:/Users/AKIN-JOHNSON/Desktop/DevSeal/label_encoder.pkl")

# Load the pre-trained StandardScaler
scaler = joblib.load("C:/Users/AKIN-JOHNSON/Desktop/DevSeal/standard_scaler.pkl")

# Function to preprocess input data
def preprocess_input(data):
    try:
        # Encode Transaction Type using the loaded LabelEncoder
        data["Transaction Type"] = label_encoder.transform(data["Transaction Type"])
    except ValueError:
        st.error("Invalid Transaction Type! Please select a valid option.")
        return None
    
    # Select only relevant features for scaling
    numerical_features = [
        "Transaction Amount", "Account Balance Before Transaction", 
        "Account Balance After Transaction", "Latitude", "Longitude"
    ]
    
    # Apply the pre-trained StandardScaler
    data[numerical_features] = scaler.transform(data[numerical_features])
    
    return data

# Streamlit App
st.title("Fraud Detection Model Deployment")

st.write("""
## Enter Transaction Details
""")

# Input fields (UI includes all fields)
transaction_id = st.number_input("Transaction ID", min_value=0)
time_stamp = st.text_input("Time Stamp (YYYY-MM-DD HH:MM:SS)")
bvn = st.number_input("BVN", min_value=0)
sender_account = st.number_input("Sender's Account Number", min_value=0)
recipient_account = st.number_input("Recipient's Account Number", min_value=0)
transaction_type = st.selectbox("Transaction Type", ["Deposit", "Transfer", "Withdrawal"])
transaction_amount = st.number_input("Transaction Amount", format="%.6f")  
balance_before = st.number_input("Account Balance Before Transaction", format="%.6f")  
balance_after = st.number_input("Account Balance After Transaction", format="%.6f")  
latitude = st.number_input("Latitude", format="%.6f")  
longitude = st.number_input("Longitude", format="%.6f")  
nin = st.number_input("NIN", min_value=0)
ip_address = st.text_input("IP Address")

# Create a DataFrame from the input data (includes all fields)
input_data = pd.DataFrame({
    "Transaction ID": [transaction_id],
    "Time Stamp": [time_stamp],
    "BVN": [bvn],
    "Sender's Account Number": [sender_account],
    "Recipient's Account Number": [recipient_account],
    "Transaction Type": [transaction_type],
    "Transaction Amount": [transaction_amount],
    "Account Balance Before Transaction": [balance_before],
    "Account Balance After Transaction": [balance_after],
    "Latitude": [latitude],
    "Longitude": [longitude],
    "NIN": [nin],
    "IP Address": [ip_address]
})

# Make a copy of input_data to retain all fields in UI
display_data = input_data.copy()

# Drop unnecessary columns before processing
input_data = input_data.drop(columns=['Transaction ID', 'Time Stamp', 'IP Address'], axis=1)

# Preprocess the input data
processed_data = preprocess_input(input_data)

# Display all entered details for transparency
st.write("### Submitted Transaction Details:")
st.dataframe(display_data)

# Make prediction
if st.button("Predict"):
    if processed_data is not None:
        prediction = model.predict(processed_data)
        if prediction[0] == 0:
            st.success("SAFE Transaction.")
        else:
            st.error("FLAG!")
