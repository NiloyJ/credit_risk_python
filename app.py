import streamlit as st 
import pandas as pd 
import joblib


model = joblib.load('extra_tree_credit_model.pkl')
# encoders = {col: joblib.load(f"{col}_encoder.okl") for col in ["Sex", "Housing", "Saving accounts", "Checking account"]}
encoders = {col: joblib.load(f"le_{col}.pkl") for col in ["Sex", "Housing", "Saving accounts", "Checking account"]}

st.title("Credit Risk Prediction App")
st.write("This app predicts whether a credit applicant is a good or bad credit risk based on their financial and personal information.")

age = st.number_input("Age", min_value=18, max_value=100, value=30)
sex = st.selectbox("Sex", ["male", "female"])
job = st.number_input("Job (0-3)", min_value=0, max_value=3, value=1)
housing = st.selectbox("Housing", ["own", "rent", "free"])
saving_accounts = st.selectbox("Saving accounts", ["little", "moderate", "rich", "quite rich"])
checking_accounts = st.selectbox("Checking accounts", ["little", "moderate", "rich"])
credit_amount = st.number_input("Credit Amount", min_value=100, value=1000)
duration = st.number_input("Duration (months)", min_value=1, value=12)

input_df = pd.DataFrame({
    "Age": [age],
    "Sex": [encoders["Sex"].transform([sex])[0]],
    "Job": [job], 
    "Housing": [encoders["Housing"].transform([housing])[0]],
    "Saving accounts": [encoders["Saving accounts"].transform([saving_accounts])[0]],
    "Checking account": [encoders["Checking account"].transform([checking_accounts])[0]],
    "Credit amount": [credit_amount],
    "Duration": [duration],
})  

if st.button("Predict Credit Risk"):
    prediction = model.predict(input_df)[0]
    if prediction == 1:
        st.success("The applicant is a GOOD credit risk.")
    else:
        st.error("The applicant is a BAD credit risk.")