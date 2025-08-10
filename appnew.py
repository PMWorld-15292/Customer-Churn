import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle
import os
from tensorflow.keras.models import load_model

# --------------------------
# Cached loading functions
# --------------------------
@st.cache_resource
def load_trained_model():
    return load_model('model.h5')

@st.cache_resource
def load_pickle(file_path):
    with open(file_path, 'rb') as file:
        return pickle.load(file)

# --------------------------
# Check required files
# --------------------------
required_files = [
    'model.h5',
    'label_encoder_gender.pkl',
    'one_hot_encoder_geo.pkl',
    'scaler.pkl'
]

missing_files = [f for f in required_files if not os.path.exists(f)]
if missing_files:
    st.error(f"Missing required files: {', '.join(missing_files)}")
    st.stop()

# --------------------------
# Load resources
# --------------------------
model = load_trained_model()
label_encoder_gender = load_pickle('label_encoder_gender.pkl')
one_hot_encoder_geo = load_pickle('one_hot_encoder_geo.pkl')
scaler = load_pickle('scaler.pkl')

# --------------------------
# Streamlit UI
# --------------------------
st.title('Customer Churn Prediction')

geography = st.selectbox('Geography', one_hot_encoder_geo.categories_[0])
gender = st.selectbox('Gender', label_encoder_gender.classes_)
age = st.slider('Age', 18, 80)
balance = st.number_input('Balance', min_value=0.0, format="%.2f")
credit_score = st.number_input('Credit Score', min_value=0, max_value=1000)
estimated_salary = st.number_input('Estimated Salary', min_value=0.0, format="%.2f")
tenure = st.slider('Tenure', 0, 10)
num_of_products = st.slider('Number Of Products', 0, 5)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])

# --------------------------
# Predict button
# --------------------------
if st.button("Predict Churn"):
    try:
        # Prepare input data
        input_data = pd.DataFrame({
            'CreditScore': [credit_score],
            'Gender': [label_encoder_gender.transform([gender])[0]],
            'Age': [age],
            'Tenure': [tenure],
            'Balance': [balance],
            'NumOfProducts': [num_of_products],
            'HasCrCard': [has_cr_card],
            'IsActiveMember': [is_active_member],
            'EstimatedSalary': [estimated_salary]
        })

        # One-hot encode geography
        geo_encoded = one_hot_encoder_geo.transform([[geography]]).toarray()
        geo_encoded_df = pd.DataFrame(
            geo_encoded,
            columns=one_hot_encoder_geo.get_feature_names_out(['Geography'])
        )

        # Combine with main data
        input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

        # Scale
        input_data_scaled = scaler.transform(input_data)

        # Predict
        prediction = model.predict(input_data_scaled)
        prediction_proba = float(prediction[0][0])

        st.write(f'Churn Probability: {prediction_proba:.2%}')
        if prediction_proba > 0.5:
            st.warning("Customer is likely to CHURN")
        else:
            st.success("Customer is not likely to CHURN.")

    except Exception as e:
        st.error(f"Error during prediction: {e}")
