# Customer Churn Prediction Model

A Machine Learning project that predicts whether a customer is likely to churn based on customer demographics, account details, and service usage.

## Project Overview

This project uses the IBM Telco Customer Churn dataset to build a churn prediction model using a Random Forest Classifier.  
The project includes data preprocessing, exploratory data analysis, model training, evaluation, and a Streamlit web application for predictions.

---

## Features

- Data Cleaning & Preprocessing
- Exploratory Data Analysis (EDA)
- Feature Engineering
- Random Forest Classification Model
- Model Evaluation
- Feature Importance Analysis
- Streamlit Web App for Predictions

---

## Tech Stack

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Streamlit

---

## Model Performance

- Model: Random Forest Classifier
- Accuracy: ~80%

---

## Dataset

Dataset used: IBM Telco Customer Churn Dataset

The dataset contains information such as:
- Customer tenure
- Contract type
- Internet service
- Monthly charges
- Payment method
- Churn status

---

## Project Structure

```bash
customer-churn-prediction/
│
├── app/
│   └── streamlit_app.py
│
├── data/
│   └── churn.csv
│
├── models/
│   └── churn_model.pkl
│
├── notebooks/
│   └── churn_analysis.ipynb
│
├── requirements.txt
├── README.md
└── .gitignore
