import numpy as np
import pandas as pd
import streamlit as st
import pickle

def load_model():
    with open("heart_disease_saved_model.pkl", 'rb') as file:
        data = pickle.load(file)
        return data

data = load_model()
model = data['model']

def predict_page():

    st.header("HEART DISEASE PREDICTION")

    st.write("Enter your details for prediction:")

    # Input fields
    age = st.slider("Age", 0, 100, step=1)
    sex = st.selectbox("Gender", ['Male', 'Female'])
    cp = st.text_input("Chest Pain (0, 1, 2, or 3):")
    trestbps = st.text_input("Resting Blood Pressure (Trestbps):")
    chol = st.text_input("Cholesterol:")
    fbs = st.text_input("Fasting Blood Sugar (0 or 1):")
    restecg = st.text_input("Rest ECG (0, 1, or 2):")
    thalach = st.text_input("Max Heart Rate:")
    exang = st.text_input("Exercise Induced Angina (0 or 1):")
    oldpeak = st.text_input("Oldpeak:")
    slope = st.text_input("Slope (0, 1, or 2):")
    ca = st.text_input("CA (0 to 4):")
    thal = st.text_input("Thal (0 to 3):")

    ok = st.button("Predict")

    if ok:
        try:
            # Input preprocessing
            input_data = pd.DataFrame({
                'age': [age], 
                'sex': [1 if sex == 'Male' else 0], 
                'cp': [float(cp)],
                'trestbps': [float(trestbps)], 
                'chol': [float(chol)], 
                'fbs': [float(fbs)],
                'restecg': [float(restecg)], 
                'thalach': [float(thalach)], 
                'exang': [float(exang)],
                'oldpeak': [float(oldpeak)], 
                'slope': [float(slope)], 
                'ca': [float(ca)], 
                'thal': [float(thal)]
            })

        except ValueError:
            st.error("Please enter valid numeric values for all fields.")
            return    

        # Convert input into numpy array
        input_array = input_data.values

        # Make the prediction
        prediction = model.predict(input_array)

        # Display the result
        if prediction[0] == 0:
            st.success("The person does not have heart disease.")
        else:
            st.warning("The person has heart disease.")

# Run the prediction page function
predict_page()
