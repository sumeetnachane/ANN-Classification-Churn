import streamlit as st
import numpy as np
import tensorflow as tf
import pandas as pd
import pickle

# -------------------- PAGE CONFIG --------------------
st.set_page_config(
    page_title="Churn AI",
    page_icon="📊",
    layout="centered"
)

# -------------------- CUSTOM CSS --------------------
st.markdown("""
<style>
.main {
    background: linear-gradient(135deg, #0E1117, #1c1f26);
}
h1 {
    text-align: center;
    color: #00C9A7;
}
h3 {
    text-align: center;
    color: #AAAAAA;
}
.stButton>button {
    background-color: #00C9A7;
    color: black;
    border-radius: 12px;
    height: 3em;
    width: 100%;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# -------------------- LOAD MODEL --------------------
model = tf.keras.models.load_model('model.h5')

with open('label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open('onehot_encoder_geo.pkl', 'rb') as file:
    onehot_encoder_geo = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# -------------------- HEADER --------------------
st.markdown("<h1>💳 Customer Churn Predictor</h1>", unsafe_allow_html=True)
st.markdown("<h3>AI system to predict customer churn risk</h3>", unsafe_allow_html=True)

st.markdown("---")

# -------------------- INPUT SECTION --------------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("👤 Customer Info")
    geography = st.selectbox("🌍 Geography", onehot_encoder_geo.categories_[0])
    gender = st.selectbox("👤 Gender", label_encoder_gender.classes_)
    age = st.slider("🎂 Age", 18, 92)
    tenure = st.slider("📅 Tenure", 0, 10)

with col2:
    st.subheader("💰 Financial Info")
    balance = st.number_input("💰 Balance", value=0.0)
    credit_score = st.number_input("📊 Credit Score", value=600)
    estimated_salary = st.number_input("💵 Estimated Salary", value=50000.0)
    num_of_products = st.slider("📦 Products", 1, 4)

st.markdown("---")

col3, col4 = st.columns(2)
with col3:
    has_cr_card = st.selectbox("💳 Has Credit Card", [0, 1])
with col4:
    is_active_member = st.selectbox("⚡ Active Member", [0, 1])

# -------------------- PREDICT BUTTON --------------------
if st.button("🔍 Predict Churn"):

    # -------- SAME DATA PREP --------
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

    # -------- ONE HOT ENCODING --------
    geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()

    geo_encoded_df = pd.DataFrame(
        geo_encoded,
        columns=onehot_encoder_geo.get_feature_names_out(['Geography'])
    )

    # -------- COMBINE --------
    input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

    # 🔥🔥 CRITICAL FIX (MATCH TRAINING ORDER)
    input_data = input_data.reindex(columns=scaler.feature_names_in_, fill_value=0)

    # -------- SCALING --------
    input_data_scaled = scaler.transform(input_data)

    # -------- PREDICTION --------
    prediction = model.predict(input_data_scaled)
    prediction_probability = prediction[0][0]

    # -------------------- RESULT --------------------
    st.markdown("---")
    st.subheader("📊 Prediction Result")

    st.progress(float(prediction_probability))
    st.write(f"Probability of Churn: **{prediction_probability:.4f}**")

    if prediction_probability > 0.5:
        st.error("⚠️ High Risk! Customer likely to churn")
    else:
        st.success("✅ Safe! Customer not likely to churn")









# import streamlit as st
# import numpy as np
# import tensorflow as tf
# from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
# import pandas as pd
# import pickle

# st.set_page_config(
#     page_title="Churn AI",
#     page_icon="📊",
#     layout="centered"
# )  

# # run in terminal = streamlit run app.py 

# # Load the trained model
# model = tf.keras.models.load_model('model.h5')

# # Load encoders and scaler
# with open('label_encoder_gender.pkl', 'rb') as file:
#   label_encoder_gender = pickle.load(file)

# with open('onehot_encoder_geo.pkl', 'rb') as file:
#   onehot_encoder_geo = pickle.load(file)  

# with open('scaler.pkl', 'rb') as file:
#   scaler = pickle.load(file)  




# ## Streamlit app

# st.title("Customer Churn Prediction")

# # User Input
# geography = st.selectbox("Geography", onehot_encoder_geo.categories_[0])
# gender = st.selectbox('Gender', label_encoder_gender.classes_)
# age = st.slider('Age', 18, 92)
# balance = st.number_input('Balance')
# credit_score =st.number_input('Credit Score')
# estimated_salary = st.number_input('Estimated Salary')
# tenure = st.slider('Tenure', 0, 10)
# num_of_products = st.slider('Number of Products', 1, 4)
# has_cr_card = st.selectbox('Has Credit Card', [0, 1])
# is_active_member = st.selectbox('Is Active Member', [0, 1])

# # Prepare input data
# input_data = pd.DataFrame({
#   'CreditScore': [credit_score],
#   'Gender': [label_encoder_gender.transform([gender])[0]],
#   'Age': [age],
#   'Tenure': [tenure],
#   'Balance': [balance],
#   'NumOfProducts': [num_of_products],
#   'HasCrCard': [has_cr_card],
#   'IsActiveMember': [is_active_member],
#   'EstimatedSalary': [estimated_salary]
# })

# ## One-hot encode 'Geography' column
# geo_encoded = onehot_encoder_geo.transform([[geography]]). toarray() ## 2d formate me convert kiya gaya hai kyunki onehotencoder 2d array return karta hai.
# geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography'])) # Encoded array ko DataFrame me convert kar rahe hain aur Columns ke naam automatically generate ho rahe hain. eg. ['Geography_India', 'Geography_France', 'Geography_Germany']

# ## Combine one-hot encoded columns with input data
# input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

# # Scale the input data
# input_data_scaled = scaler.transform(input_data)

# # Scale the input data
# input_data_scaled = scaler.transform(input_data)

# # Make prediction on Customer Churn Probability
# prediction = model.predict(input_data_scaled)
# prediction_propability = prediction[0][0]

# st.write(f"Churn Probability: {prediction_propability:.2f}")

# if prediction_propability > 0.5:
#   st.write("The customer is likely to churn.")
# else:
#   st.write("The customer is not likely to churn.")
