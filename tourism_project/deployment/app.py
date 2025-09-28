import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download and load the model

# replace with your repoid
model_path = hf_hub_download(repo_id="ganeshdattatreyan/tourism-package-prediction", filename="best_tourist_product_taken_model_v1.joblib")

model = joblib.load(model_path)

# Streamlit UI for Machine Failure Prediction
st.title("Tourise Product Prediction App")
st.write("""
This application predicts the likelihood of tourists purchasing the product that is marketed.
Please enter the data below to get a prediction.
""")

# User input
Age = st.number_input("Age", min_value=18, max_value=100, step=1)
TypeofContact = st.selectbox("Type of Contact", ["Company Invited", "Self Inquiry"])
CityTier = st.selectbox("City Tier", ["Tier 1", "Tier 2", "Tier 3"])
Occupation = st.selectbox("Occupation", ["Salaried", "Freelancer"])
Gender = st.selectbox("Gender", ["Male", "Female"])
NumberOfPersonVisiting = st.number_input("Number of People Visiting", min_value=1, max_value=10, step=1)
PreferredPropertyStar = st.number_input("Preferred Property Star", min_value=1, max_value=5, step=1)
MaritalStatus = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
NumberOfTrips = st.number_input("Number of Trips", min_value=1, max_value=10, step=1)
Passport = st.selectbox("Passport", ["No", "Yes"])
OwnCar = st.selectbox("Own Car", ["No", "Yes"])
NumberOfChildrenVisiting = st.number_input("Number of Children Visiting", min_value=0, max_value=5, step=1)
Designation = st.selectbox("Designation", ["Executive", "Managerial", "Professional", "Other"])
MonthlyIncome = st.number_input("Monthly Income", min_value=1000, max_value=100000, step=1000)
PitchSatisfactionScore = st.number_input("Pitch Satisfaction Score", min_value=1, max_value=5, step=1)
ProductPitched = st.selectbox("Product Pitched", ["Basic", "Standard", "Advance"])
NumberOfFollowups = st.number_input("Number of Follow-ups", min_value=1, max_value=10, step=1)
DurationOfPitch = st.number_input("Duration of Pitch", min_value=1, max_value=10, step=1)

# Assemble input into DataFrame
input_data = pd.DataFrame([{
    'Age': Age,
    'TypeofContact': TypeofContact,
    'CityTier': CityTier,
    'Occupation': Occupation,
    'Gender': Gender,
    'NumberOfPersonVisiting': NumberOfPersonVisiting,
    'PreferredPropertyStar': PreferredPropertyStar,
    'MaritalStatus': MaritalStatus,
    'NumberOfTrips': NumberOfTrips,
    'Passport': Passport,
    'OwnCar': OwnCar,
    'NumberOfChildrenVisiting': NumberOfChildrenVisiting,
    'Designation': Designation,
    'MonthlyIncome': MonthlyIncome,
    'PitchSatisfactionScore': PitchSatisfactionScore,
    'ProductPitched': ProductPitched,
    'NumberOfFollowups': NumberOfFollowups,
    'DurationOfPitch': DurationOfPitch
}])

if st.button("Predict Product Purchase"):
    prediction = model.predict(input_data)[0]
    result = "Tourism Product Purchased" if prediction == 1 else "Tourism Product Not Purchased"
    st.subheader("Prediction Result:")
    st.success(f"The model predicts: **{result}**")
