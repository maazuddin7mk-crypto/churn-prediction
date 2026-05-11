import streamlit as st
import pandas as pd
import pickle
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(BASE_DIR, "..", "models", "churn_model.pkl")

print(model_path)  # Debug

with open(model_path, "rb") as file:
    model = pickle.load(file)
model_columns = pickle.load(open('models/model_columns.pkl', 'rb'))
st.set_page_config(
    page_title="Customer Churn Prediction",
    layout="centered"
)

# Title
st.title("📊 Customer Churn Prediction App")

st.write("""
This machine learning app predicts whether a customer is likely to churn.
""")

st.divider()


# USER INPUTS


gender = st.selectbox(
    "Gender",
    ["Male", "Female"]
)

senior = st.selectbox(
    "Senior Citizen",
    ["Yes", "No"]
)

partner = st.selectbox(
    "Has Partner",
    ["Yes", "No"]
)

dependents = st.selectbox(
    "Has Dependents",
    ["Yes", "No"]
)

tenure = st.slider(
    "Tenure Months",
    0,
    72,
    12
)

phone_service = st.selectbox(
    "Phone Service",
    ["Yes", "No"]
)

multiple_lines = st.selectbox(
    "Multiple Lines",
    ["Yes", "No", "No phone service"]
)

internet_service = st.selectbox(
    "Internet Service",
    ["DSL", "Fiber optic", "No"]
)

online_security = st.selectbox(
    "Online Security",
    ["Yes", "No", "No internet service"]
)

online_backup = st.selectbox(
    "Online Backup",
    ["Yes", "No", "No internet service"]
)

device_protection = st.selectbox(
    "Device Protection",
    ["Yes", "No", "No internet service"]
)

tech_support = st.selectbox(
    "Tech Support",
    ["Yes", "No", "No internet service"]
)

streaming_tv = st.selectbox(
    "Streaming TV",
    ["Yes", "No", "No internet service"]
)

streaming_movies = st.selectbox(
    "Streaming Movies",
    ["Yes", "No", "No internet service"]
)

contract = st.selectbox(
    "Contract Type",
    ["Month-to-month", "One year", "Two year"]
)

paperless = st.selectbox(
    "Paperless Billing",
    ["Yes", "No"]
)

payment_method = st.selectbox(
    "Payment Method",
    [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ]
)

monthly_charges = st.number_input(
    "Monthly Charges",
    min_value=0.0,
    max_value=200.0,
    value=70.0
)

total_charges = st.number_input(
    "Total Charges",
    min_value=0.0,
    value=1000.0
)


# ENCODING INPUT DATA


input_dict = {
    'Tenure Months': tenure,
    'Monthly Charges': monthly_charges,
    'Total Charges': total_charges,

    'Gender_Male': 1 if gender == 'Male' else 0,

    'Senior Citizen_Yes': 1 if senior == 'Yes' else 0,

    'Partner_Yes': 1 if partner == 'Yes' else 0,

    'Dependents_Yes': 1 if dependents == 'Yes' else 0,

    'Phone Service_Yes': 1 if phone_service == 'Yes' else 0,

    'Multiple Lines_Yes': 1 if multiple_lines == 'Yes' else 0,

    'Internet Service_Fiber optic': 1 if internet_service == 'Fiber optic' else 0,
    'Internet Service_No': 1 if internet_service == 'No' else 0,

    'Online Security_Yes': 1 if online_security == 'Yes' else 0,

    'Online Backup_Yes': 1 if online_backup == 'Yes' else 0,

    'Device Protection_Yes': 1 if device_protection == 'Yes' else 0,

    'Tech Support_Yes': 1 if tech_support == 'Yes' else 0,

    'Streaming TV_Yes': 1 if streaming_tv == 'Yes' else 0,

    'Streaming Movies_Yes': 1 if streaming_movies == 'Yes' else 0,

    'Contract_One year': 1 if contract == 'One year' else 0,
    'Contract_Two year': 1 if contract == 'Two year' else 0,

    'Paperless Billing_Yes': 1 if paperless == 'Yes' else 0,

    'Payment Method_Credit card (automatic)': 1 if payment_method == 'Credit card (automatic)' else 0,

    'Payment Method_Electronic check': 1 if payment_method == 'Electronic check' else 0,

    'Payment Method_Mailed check': 1 if payment_method == 'Mailed check' else 0,
}

# Create dataframe
input_df = pd.DataFrame([input_dict])


# PREDICTION


if st.button("Predict Churn"):

    # One-hot encode input
    input_df = pd.get_dummies(input_df)

    # Add missing columns
    for col in model_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    # Keep only training columns
    input_df = input_df[model_columns]

    # Predict
    prediction = model.predict(input_df)[0]

    probability = model.predict_proba(input_df)[0][1]

    st.divider()

    if prediction == 1:
        st.error("⚠️ Customer is likely to churn")
    else:
        st.success("✅ Customer is likely to stay")

    st.write(f"### Churn Probability: {probability:.2%}")