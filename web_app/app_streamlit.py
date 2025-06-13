import streamlit as st
import pandas as pd
import joblib
import os

# Loading the trained Pipeline
MODELS_PATH = os.path.join(os.path.dirname(__file__), '..', 'trained_models')

try:
    # Load the entire Pipeline
    model_pipeline = joblib.load(os.path.join(MODELS_PATH, 'churn_prediction_pipeline.pkl'))
    st.sidebar.success("Model loaded successfully!")
except FileNotFoundError as e:
    st.sidebar.error(f"Error loading model files: {e}")
    st.sidebar.warning(
        "Please ensure you have run the train_and_save_model.py script and the files are in the 'trained_models' folder.")
    st.stop()

# Setting up the title and introduction for the app
st.title("📊 Bank Customer Churn Prediction")
st.markdown("""
This application uses a machine learning model to predict the probability of customer churn.
Enter customer data in the form below to get a prediction.
""")

st.write("---")  # Decorative separator

# Creating input fields for customer data
st.header("Enter Customer Data:")

# Using st.columns for better field arrangement
col1, col2 = st.columns(2)

with col1:
    credit_score = st.slider("Credit Score", 350, 850, 650)
    geography = st.selectbox("Geography", ['France', 'Germany', 'Spain'])
    gender = st.radio("Gender", ['Male', 'Female'])
    age = st.slider("Age", 18, 92, 30)
    tenure = st.slider("Tenure (years)", 0, 10, 5)

with col2:
    balance = st.number_input("Balance", value=0.0, format="%.2f")
    num_products = st.slider("Number of Products", 1, 4, 1, step=1)
    # Convert 1/0 to "Yes/No" for user convenience
    has_cr_card_input = st.radio("Has Credit Card?", ["Yes", "No"])
    is_active_member_input = st.radio("Is Active Member?", ["Yes", "No"])
    estimated_salary = st.number_input("Estimated Salary", value=50000.00, format="%.2f")

# Convert "Yes/No" to 1/0
has_cr_card = 1 if has_cr_card_input == "Yes" else 0
is_active_member = 1 if is_active_member_input == "Yes" else 0

st.write("---")

# Prediction button
if st.button("Make Prediction"):
    try:
        # Collect data into a dictionary
        single_input_raw = {
            'CreditScore': credit_score,
            'Geography': geography,
            'Gender': gender,
            'Age': age,
            'Tenure': tenure,
            'Balance': balance,
            'NumOfProducts': num_products,
            'HasCrCard': has_cr_card,
            'IsActiveMember': is_active_member,
            'EstimatedSalary': estimated_salary
        }

        # Create a DataFrame from a single row
        input_df_raw = pd.DataFrame([single_input_raw])

        # Add 'HasNoBalance' feature
        input_df_raw['HasNoBalance'] = (input_df_raw['Balance'] == 0).astype(int)

        # Perform prediction
        prediction_proba = model_pipeline.predict_proba(input_df_raw)[:, 1][0]  # Churn probability (class 1)
        prediction_class = model_pipeline.predict(input_df_raw)[0]  # Class: 0 - stays, 1 - churns

        # Displaying results
        st.subheader("💡 Prediction Result:")
        st.info(f"**Churn Probability:** `{prediction_proba:.2f}`")

        if prediction_class == 1:
            st.error("❗ **Conclusion: Customer is likely to CHURN.** Recommended to take action.")
        else:
            st.success("✅ **Conclusion: Customer is likely to STAY.**")

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
        st.write("Please check the input data.")

st.sidebar.markdown("---")
st.sidebar.info("This application is a demonstration of a customer churn prediction model.")